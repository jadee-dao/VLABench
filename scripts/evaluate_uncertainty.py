
import argparse
import json
import os
import time
import traceback
import torch
import numpy as np
from colorama import Fore, Style, init
from VLABench.evaluation.evaluator import VLMEvaluator
from VLABench.evaluation.model.vlm import *
from VLABench.evaluation.uncertainty import UncertaintyEvaluator
from dotenv import load_dotenv

init(autoreset=True)

def initialize_model(model_name, model_path=None, thinking_mode=False, **kwargs):
    # Handle Generic Wrapper instantiation
    if model_name in ["Qwen2_VL_Generic", "Qwen3_VL_Generic"]:
         if not model_path:
             raise ValueError(f"Must provide --model_path for {model_name}")
         
         thinking_system_prompt = None
         if thinking_mode:
             thinking_system_prompt = (
                 "You are a helpful assistant. You are in 'Thinking Mode'. "
                 "Before answering, you must output a reasoning trace content. "
                 "Be concise. "
                 "Your output must follow this format: "
                 "Reasoning: [Your step-by-step reasoning] "
                 "Answer: [Final JSON output]"
             )
         
         if "Qwen3" in model_name or "Thinking" in model_path: # Assuming model_name is intended here instead of vlm_name
              return Qwen3_VL_Generic(
                 model_path=model_path, 
                 system_prompt=thinking_system_prompt,
                 **kwargs
             )
         return Qwen2_VL_Generic(
             model_path=model_path, 
             system_prompt=thinking_system_prompt,
             **kwargs
         )

    cls = globals().get(model_name)
    if cls is None:
        raise ValueError(f"Model '{model_name}' not found in the current namespace.")
    return cls(**kwargs) # Pass kwargs to the dynamically loaded class

def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM Uncertainty Quantification.")
    parser.add_argument("--vlm_name", type=str, default="GPT_4v", help="Name of the model class to instantiate")
    parser.add_argument("--model_path", type=str, default=None, help="Hugging Face model path for Generic wrappers")
    parser.add_argument("--thinking_mode", action="store_true", help="Enable 'Thinking' system prompt for Generic wrappers")
    parser.add_argument("--tasks", nargs='+', default=None, help="Specific tasks to run, default is None")
    parser.add_argument("--n-episodes", type=int, default=5, help="Number of episodes to evaluate per task (keep small for testing)")
    parser.add_argument("--save_dir", type=str, default="uncertainty_results", help="Directory to save evaluation results")
    parser.add_argument("--eval-dimension", type=str, default="M&T", help="Evaluation dimension folder name")
    parser.add_argument("--methods", nargs='+', default=["token_prob", "perplexity", "verbalized", "p_true", "consistency"], 
                        choices=["all", "token_prob", "perplexity", "verbalized", "p_true", "consistency", "probe", "ood_probe"],
                        help="Uncertainty methods to evaluate. Use 'all' to run all available methods.")
    args = parser.parse_args()
    
    # Expand 'all' to full list
    if "all" in args.methods:
        args.methods = ["token_prob", "perplexity", "verbalized", "p_true", "consistency", "probe", "ood_probe"]
        
    return args

class LLMJudgeEvaluator:
    def __init__(self):
        from openai import OpenAI
        import dotenv
        dotenv.load_dotenv()
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"), 
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=30.0
        )
        self.model = "x-ai/grok-4.1-fast"

    def judge(self, instruction_text, predicted_seq, gt_seq, objects_id_key, task_original):
        prompt = f'''
    Decide if given the instruction, the predicted sequence is acceptably similar to the ground truth skill sequence.
    Instruction: {instruction_text}

    Predicted Skill Sequence:
    {json.dumps(predicted_seq, indent=2)}

    Ground Truth Skill Sequence:
    {json.dumps(gt_seq, indent=2)}

    Objects ID Key:
    {objects_id_key}

    Does this predicted skill sequence:
    - Matches the goal explicitly described in the instruction. Do not infer extra requirements that the instruction does not state.
    - Correct entity/object IDs necessary to achieve that goal.
    - If this is an ordered task, the sequence in which objects are interacted with is also correct.
    - Uses plausible skills that would make sense for the {task_original} environment.

    Your output MUST be a JSON object with the following keys:
    -   "is_acceptable": boolean (true if acceptably similar, false otherwise)
    -   "score": score from 1-100 of "acceptable" similarity.
    -   "reason": string (detailed explanation for your decision)
    '''
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            print(f"Judge Error: {e}")
            return {"is_acceptable": False, "score": 0, "reason": str(e)}

def main():
    args = parse_args()
    
    # Pass generic args to initialize_model
    vlm = initialize_model(args.vlm_name, model_path=args.model_path, thinking_mode=args.thinking_mode)
    model_identifier = getattr(vlm, "name", args.vlm_name)
    
    dimension_list = [args.eval_dimension]
    if args.eval_dimension == "all":
        dimension_list = ["CommonSense", "Complex", "M&T", "PhysicsLaw", "Semantic", "Spatial"]
        
    for current_dim in dimension_list:
        print(f"\n=== Evaluating Dimension: {current_dim} ===")
        # Initialize helper (data path depends on dimension)
        # Note: Previous logic assumed VLABENCH_ROOT env var points appropriately such that ../dataset works
        # If VLABENCH_ROOT is /home/jadelynn/VLABench/VLABench, then ../dataset is /home/jadelynn/VLABench/dataset
        # which matches our 'find' result.
        
        data_path = os.path.join(os.getenv("VLABENCH_ROOT"), "../dataset", f"vlm_evaluation_v1.0/{current_dim}")
        
        # Check if dim path exists
        if not os.path.exists(data_path):
             print(f"Dimension path not found: {data_path}. Skipping.")
             continue

        if args.tasks is None:
            try:
                task_list = os.listdir(data_path)
            except FileNotFoundError:
                print(f"Could not list tasks in {data_path}")
                continue
        else:
            task_list = args.tasks
            
        evaluator = VLMEvaluator(
            tasks=task_list, 
            n_episodes=args.n_episodes,
            data_path=data_path,
            save_path=args.save_dir
        )
        
        uq_evaluator = UncertaintyEvaluator(vlm)
        
        results = {}
        # Save file includes dimension name to avoid overwriting
        save_file = os.path.join(args.save_dir, model_identifier, f"{current_dim}_uncertainty.json")
        if os.path.exists(save_file):
            with open(save_file, 'r') as f:
                results = json.load(f)
                
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        
        print(f"Starting Uncertainty Evaluation for {model_identifier} on {len(task_list)} tasks in {current_dim}...")
        
        for task_name in task_list:
            if task_name not in results:
                results[task_name] = {}
                
            task_path = os.path.join(data_path, task_name)
            if not os.path.exists(task_path):
                print(f"Task path not found: {task_path}")
                continue
                
            num_examples = min(args.n_episodes, len(os.listdir(task_path)))
            
            for example_num in range(num_examples):
                str_example_num = str(example_num)
                if str_example_num in results[task_name]:
                    print(f"Skipping {current_dim}/{task_name} example {example_num} (already done)")
                    continue
                    
                print(f"Processing {current_dim}/{task_name} example {example_num}...")
                
                try:
                    # Load input
                    input_pic, input_pic_gt, input_instruction_text = evaluator.load_single_input(task_name, example_num)
                    # Need to construct the input_dict expected by vlm.evaluate
                    # VLMEvaluator.build_input does this but it adds shots. We can use 0-shot.
                    # Actually, simpler to use evaluator.build_input(task_name, example_num, few_shot_num=0)
                    input_dict = evaluator.build_input(task_name, example_num, few_shot_num=0)
                    
                    # 1. Base Generation (Answer)
                    # ---------------------------------------------------------------------
                    print(f"  Running base generation...")
                    start_time_base = time.time()
                    response = vlm.evaluate(input_dict, "en") # Assume English for now
                    end_time_base = time.time()
                    answer_latency = end_time_base - start_time_base
                    
                    previous_answer_text = response.get("origin_output", "")
                    reasoning_trace = ""
                    
                    # Attempt to extract JSON if mixed with text (Thinking Mode)
                    # Look for ```json ... ``` or just the first [ ... ] block
                    try:
                        import re
                        # Pattern for markdown json block
                        json_match = re.search(r"```json\s*(\[.*?\])\s*```", previous_answer_text, re.DOTALL)
                        if not json_match:
                            # Pattern for just brackets if no markdown
                            json_match = re.search(r"(\[.*\])", previous_answer_text, re.DOTALL)
                            
                        if json_match:
                            raw_json = json_match.group(1)
                            # Verify if valid json
                            json.loads(raw_json) 
                            
                            # If valid, split content
                            reasoning_trace = previous_answer_text.replace(raw_json, "").replace("```json", "").replace("```", "").strip()
                            previous_answer_text = raw_json
                    except:
                        pass # Keep original text if extraction fails

                    if "skill_sequence" in response:
                        previous_answer_text = json.dumps(response["skill_sequence"])
                    
                    # 2. Get Uncertainty Scores
                    # ---------------------------------------------------------------------
                    print(f"  Calculating Uncertainty Metrics ({', '.join(args.methods)})...")
                    uncertainty_metrics = uq_evaluator.get_multiple_confidence_scores(
                        vlm, 
                        input_dict, 
                        response, 
                        methods=args.methods
                    )
                    
                    # --- LLM Judge Integration ---
                    try:
                        # Load Ground Truth and Config for Judge
                        # We need full paths or use the evaluator's helpers if available.
                        # Since `VLMEvaluator` doesn't expose easy single-file loaders for GT/Config without building input, 
                        # we rely on manual loading similar to `check_eval_with_llm_judge.py` or use existing paths.
                        
                        # Re-construct paths from data_path + task_name
                        example_dir = os.path.join(data_path, task_name, f"example{example_num}")
                        
                        # Load GT Sequence
                        with open(os.path.join(example_dir, "output", "operation_sequence.json"), "r") as f:
                            gt_seq = json.load(f)
                            
                        # Load Env Config for IDs
                        objects_id_key = {}
                        env_config_path = os.path.join(example_dir, "env_config", "env_config.json")
                        if os.path.exists(env_config_path):
                            with open(env_config_path, "r") as f:
                                components = json.load(f).get("task", {}).get("components", {})
                                for i, comp in enumerate(components):
                                    objects_id_key[i] = comp['name']
                        
                        # Attempt to parse predicted sequence from string
                        try:
                            predicted_seq = json.loads(previous_answer_text)
                        except:
                            predicted_seq = {} # Invalid JSON

                        # Initialize Judge if not exists
                        if 'llm_judge' not in locals():
                            llm_judge = LLMJudgeEvaluator()
                            
                        print(f"  Running LLM Judge for correctness...")
                        judge_result = llm_judge.judge(
                            instruction_text=input_instruction_text,
                            predicted_seq=predicted_seq,
                            gt_seq=gt_seq,
                            objects_id_key=objects_id_key,
                            task_original=task_name
                        )
                        example_results["llm_judge"] = judge_result
                        print(f"  Judge Score: {judge_result.get('score')}")
                        
                    except Exception as e:
                        print(f"  LLM Judge failed: {e}")
                        example_results["llm_judge"] = {"error": str(e)}
                    # -----------------------------
                    
                    if reasoning_trace:
                        example_results["reasoning_trace"] = reasoning_trace
                    
                    if "token_prob" in args.methods:
                        example_results["token_prob"] = uq_evaluator.get_token_prob_confidence(response)
                    
                    if "perplexity" in args.methods:
                        example_results["perplexity"] = uq_evaluator.get_perplexity(response)
                        
                    if "probe" in args.methods:
                        example_results["probe"] = uq_evaluator.get_hidden_state_probe(response)
                        
                    if "ood_probe" in args.methods:
                        example_results["ood_probe"] = uq_evaluator.get_ood_probe(response)
    
                    # 2. Verbalization (New call)
                    if "verbalized" in args.methods:
                        print("  Running verbalization...")
                        t0 = time.time()
                        example_results["verbalized"] = uq_evaluator.get_verbalized_confidence(input_dict, "en")
                        example_results["latency_verbalized"] = time.time() - t0
                        
                    # 3. P(True) (New call)
                    if "p_true" in args.methods:
                        print("  Running P(True)...")
                        t0 = time.time()
                        example_results["p_true"] = uq_evaluator.get_p_true_confidence(input_dict, previous_answer_text, "en")
                        example_results["latency_p_true"] = time.time() - t0
    
                    # 4. Consistency (New calls)
                    if "consistency" in args.methods:
                        print("  Running consistency...")
                        t0 = time.time()
                        example_results["consistency"] = uq_evaluator.get_self_check_consistency(input_dict, "en")
                        example_results["latency_consistency"] = time.time() - t0
                    
                    example_results["latency_total"] = time.time() - start_time_base
                    results[task_name][str_example_num] = example_results
                    
                    # Save periodically
                    with open(save_file, 'w') as f:
                        json.dump(results, f, indent=4)
                        
                except Exception as e:
                    print(f"Error processing {task_name} example {example_num}: {e}")
                    traceback.print_exc()
                
    print("Evaluation complete.")

if __name__ == "__main__":
    load_dotenv()
    main()
