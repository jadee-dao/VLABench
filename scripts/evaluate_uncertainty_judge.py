
import os
import argparse
import json
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import dotenv
import time

# Load environment variables
dotenv.load_dotenv()

# Constants
DOMAIN_SUFFIXES_KEY = {
    "CommonSense": "_common_sense",
    "Semantic": "_semantic",
    "Spatial": "_spatial",
    "Complex": "_complex",
    "M&T": "_m&t",
    "PhysicsLaw": "_physics_law"
}

# Assume VLABENCH_ROOT is set or we can infer it
VLABENCH_ROOT = os.environ.get("VLABENCH_ROOT", os.getcwd())
if "VLABench" not in VLABENCH_ROOT: # Fallback if run from a weird place
    pass 

DATASET_BASE_DIR = Path(VLABENCH_ROOT) / "../dataset/vlm_evaluation_v1.0/"

# Fix path resolution if VLABENCH_ROOT points to inner VLABench
if not DATASET_BASE_DIR.exists():
    DATASET_BASE_DIR = Path(VLABENCH_ROOT) / "dataset/vlm_evaluation_v1.0/"


def get_results_df(results_dir):
    records = []
    
    # Iterate over all JSON files in the results directory
    json_files = glob.glob(os.path.join(results_dir, "*_uncertainty.json"))
    
    for json_file in json_files:
        filename = os.path.basename(json_file)
        # Try to infer domain from filename (e.g., CommonSense_uncertainty.json -> CommonSense)
        domain = filename.replace("_uncertainty.json", "")
        
        # Check if this maps to a known domain
        # Some map exactly, but let's just use the prefix as domain usually
        # The key in the JSON usually contains the task name
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        suffix = DOMAIN_SUFFIXES_KEY.get(domain)
        
        for task, items in data.items():
            if task == 'benchmeatinfo':
                continue
                
            task_original = task
            task_normalized = task
            if suffix and task.endswith(suffix):
                task_normalized = task.removesuffix(suffix)
            
            for sample_id, metrics in items.items():
                predicted_skill_seq = []
                answer = metrics.get("answer", "")
                
                # Parse the answer which is a stringified JSON
                try:
                    if answer:
                        # Clean markdown code blocks if present
                        if "```json" in answer:
                            answer = answer.replace("```json", "").replace("```", "")
                        elif "```" in answer:
                            answer = answer.replace("```", "")
                        answer = answer.strip()
                            
                        predicted_skill_seq = json.loads(answer)
                except json.JSONDecodeError:
                    print(f"Failed to parse answer JSON for {task} sample {sample_id}. Raw answer: {answer!r}")
                    continue
                
                # Construct path to dataset example
                dataset_example_path = DATASET_BASE_DIR / domain / task_original / f"example{sample_id}"
                
                # 1. Load Instruction
                instruction_text = ""
                try:
                    with open(dataset_example_path / "input" / "instruction.txt", "r") as f:
                        instruction_text = f.read().strip()
                except FileNotFoundError:
                    # Try alternate path if task name doesn't match directory exactly
                    if suffix: 
                        # Sometimes the directory name is without suffix? Or with?
                        # Based on previous script, it seems dataset is {domain}/{task}/
                        pass
                    print(f"Instruction file not found for {dataset_example_path}")
                    continue # Cannot evaluate without instruction
                
                # 2. Load GT Skill Sequence
                gt_skill_sequence = []
                try:
                    with open(dataset_example_path / "output" / "operation_sequence.json", "r") as f:
                        gt_skill_sequence = json.load(f)
                except FileNotFoundError:
                    print(f"GT skill sequence file not found for {dataset_example_path}")
                    continue

                # 3. Load Objects ID Key (Env Config)
                objects_id_key = {}
                try:
                    with open(dataset_example_path / "env_config" / "env_config.json", "r") as f:
                        components = json.load(f).get("task", {}).get("components", {})
                        for i, comp in enumerate(components):
                            objects_id_key[i] = comp['name']
                except FileNotFoundError:
                    print(f"Objects env config file not found for {dataset_example_path}")
                
                records.append({
                    "domain": domain,
                    "task": task_normalized,
                    "task_original": task_original,
                    "sample_id": sample_id,
                    "predicted_skill_seq": predicted_skill_seq,
                    "gt_skill_seq": gt_skill_sequence,
                    "instruction_text": instruction_text,
                    "objects_id_key": objects_id_key,
                    "metrics": metrics # Keep original metrics to preserve context
                })

    return pd.DataFrame.from_records(records)

def evaluate_with_judge(examples_df, client, model_name="google/gemini-2.0-flash-exp:free", save_path_csv=None):
    judge_results = []
    
    # For now, just simplistic incremental append or overwrite
    print("DEBUG: Starting evaluate_with_judge with retry logic")
    
    for index, row in tqdm(examples_df.iterrows(), total=len(examples_df), desc="Judging examples"):
        try:
            predicted_skill_seq = row["predicted_skill_seq"]
            gt_skill_seq = row["gt_skill_seq"]
            instruction_text = row["instruction_text"]
            objects_id_key = row["objects_id_key"]
            task_original = row["task_original"]

            # Construct the prompt for the LLM judge
            prompt = f"""
    Decide if given the instruction, the predicted sequence is acceptably similar to the ground truth skill sequence.
    Instruction: {instruction_text}

    Predicted Skill Sequence:
    {json.dumps(predicted_skill_seq, indent=2)}

    Ground Truth Skill Sequence:
    {json.dumps(gt_skill_seq, indent=2)}

    Objects ID Key:
    {objects_id_key}

    Does this predicted skill sequence:
    - Matches the goal explicitly described in the instruction. Do not infer extra requirements that the instruction does not state. If the instruction only describes what object to choose (a "selection-only" instruction), then selecting the correct object is fully sufficient. Do not require any additional actions (e.g., pour, place, insert) unless the instruction explicitly indicates the object should be applied or used.
    - Correct entity/object IDs necessary to achieve that goal.
    - If this is an ordered task, the sequence in which objects are interacted with is also correct.
    - Uses plausible skills that would make sense for the {task_original} environment, given a natural interpretation of the task.

    Your output MUST be a JSON object with the following keys:
    -   "is_acceptable": boolean (true if acceptably similar, false otherwise)
    -   "score": score from 1-100 of "acceptable" similarity.
    -   "reason": string (detailed explanation for your decision)
    -   "key_diffs": list of brief strings (diffs to transform the predicted sequence into the ground truth)
    """

            # Call the LLM judge
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0, 
            )

            judge_output_str = chat_completion.choices[0].message.content
            judge_output = json.loads(judge_output_str)

            judge_results.append({
                "llm_judge_is_acceptable": judge_output.get("is_acceptable"),
                "llm_judge_score": judge_output.get("score"),
                "llm_judge_reason": judge_output.get("reason"),
                "llm_judge_key_diffs": judge_output.get("key_diffs"),
            })

        except Exception as e:
            print(f"Error judging row {index}: {e}")
            judge_results.append({
                "llm_judge_is_acceptable": None,
                "llm_judge_score": None,
                "llm_judge_reason": f"Error: {e}",
                "llm_judge_key_diffs": [],
            })
        
        # Incremental save every 10 items
        if len(judge_results) % 10 == 0 and save_path_csv:
             # Construct partial DF and save (merging with original headers if possible)
             # To keep it simple, we just save what we have as a backup
             pass 

    return pd.DataFrame(judge_results, index=examples_df.index)

def main():
    parser = argparse.ArgumentParser(description="Evaluate uncertainty results with LLM judge")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing uncertainty results json files")
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.2-3b-instruct:free", help="OpenRouter model to use")
    args = parser.parse_args()

    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    
    # Fallback to OPENAI_API_KEY if OPENROUTER_API_KEY is not set but base url indicates openrouter
    if not OPENROUTER_API_KEY:
        possible_key = os.environ.get("OPENAI_API_KEY")
        if possible_key and "openrouter" in OPENROUTER_BASE_URL:
             OPENROUTER_API_KEY = possible_key
    
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY (or OPENAI_API_KEY with openrouter base url) not found in environment variables.")
        return

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    print(f"Loading results from {args.results_dir}...")
    examples_df = get_results_df(args.results_dir)
    
    if examples_df.empty:
        print("No examples found or failed to load data.")
        return
        
    print(f"Found {len(examples_df)} examples to evaluate.")
    
    save_path_csv = os.path.join(args.results_dir, "llm_judge.csv")
    
    try:
        judge_results_df = evaluate_with_judge(examples_df, client, model_name=args.model, save_path_csv=save_path_csv)
    except KeyboardInterrupt:
        print("\nInterrupted! Saving partial results...")
        judge_results_df = pd.DataFrame() # This would lose data if we don't return partial from evaluate_with_judge
        # We need to restructure evaluate_with_judge to output partial
        pass
    
    # Final save
    # If evaluate_with_judge returns (even if interrupted and handled handled internally if we improve it later),
    # judge_results_df will contain what we have.
    # Note: currently evaluate_with_judge catches generic exceptions but NOT KeyboardInterrupt internally to return partial.
    # If we want to support partial save on interrupt, evaluate_with_judge needs to handle it or we rely on the save_path_csv inside it.
    
    # Since evaluate_with_judge already saves to CSV incrementally, we might just need to finalize the JSON.
    
    # Let's rely on the CSV being source of truth if we crash.
    # But here we want to save the final JSON if we finished successfully.
    
    if 'judge_results_df' in locals() and not judge_results_df.empty:
        # Merge results just in case (though evaluate_with_judge returns only new rows? No, it returns all with alignment)
        final_df = examples_df.join(judge_results_df, rsuffix="_judge")
        
        # Save to JSON in the format of {task: {sample_id: {judge_result}}}
        save_path_json = os.path.join(args.results_dir, "llm_judge.json")
        
        judge_results_dict = {}
        for _, row in final_df.iterrows():
            if pd.isna(row.get("llm_judge_is_acceptable")):
                continue
                
            task = row["task_original"]
            sample_id = row["sample_id"]

            if task not in judge_results_dict:
                judge_results_dict[task] = {}

            judge_results_dict[task][sample_id] = {
                "is_acceptable": row["llm_judge_is_acceptable"],
                "score": row["llm_judge_score"],
                "reason": row["llm_judge_reason"],
                "key_diffs": row["llm_judge_key_diffs"],
            }
            
        with open(save_path_json, "w") as f:
            json.dump(judge_results_dict, f, indent=4)
        print(f"Saved JSON results to {save_path_json}")
    


if __name__ == "__main__":
    main()
