
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union
import re
import copy

class UncertaintyEvaluator:
    """
    Evaluator for Uncertainty Quantification methods in VLMs.
    Wraps a VLM model to perform various uncertainty estimation strategies.
    """
    def __init__(self, vlm_model):
        self.vlm = vlm_model

    def get_token_prob_confidence(self, response_dict: Dict[str, Any]) -> float:
        """
        Calculates confidence based on average token probability.
        Expects response_dict to contain 'logprobs' with 'avg_token_prob' or 'all_logprobs'.
        """
        if "logprobs" in response_dict and response_dict["logprobs"]:
            logprobs_data = response_dict["logprobs"]
            
            # If the model wrapper already provides avg_token_prob
            if "avg_token_prob" in logprobs_data:
                return logprobs_data["avg_token_prob"]
            
            # Otherwise calculate from all_logprobs if available
            if "all_logprobs" in logprobs_data and logprobs_data["all_logprobs"]:
                # Filter out None values
                valid_logprobs = [lp for lp in logprobs_data["all_logprobs"] if lp is not None]
                if valid_logprobs:
                    avg_logprob = np.mean(valid_logprobs)
                    return np.exp(avg_logprob)
        
        return 0.0

    def get_perplexity(self, response_dict: Dict[str, Any]) -> float:
        """
        Calculates perplexity of the generated sequence.
        Perplexity = exp(-1/N * sum(log(p_i)))
        """
        if "logprobs" in response_dict and response_dict["logprobs"]:
            logprobs_data = response_dict["logprobs"]
            
            # If avg_nll is provided (NLL = -1/N * sum(log(p_i)))
            if "avg_nll" in logprobs_data:
                return np.exp(logprobs_data["avg_nll"])
            
            # Calculate from all_logprobs
            if "all_logprobs" in logprobs_data and logprobs_data["all_logprobs"]:
                valid_logprobs = [lp for lp in logprobs_data["all_logprobs"] if lp is not None]
                if valid_logprobs:
                    avg_logprob = np.mean(valid_logprobs)
                    return np.exp(-avg_logprob)

        return 0.0

    def get_verbalized_confidence(self, input_dict: Dict[str, Any], language: str = "en") -> Union[str, float]:
        """
        Prompts the model to verbalize confidence (either linguistic or numerical).
        """
        # Append a query for confidence to the input instruction.
        confidence_prompt = {
            "en": "\n\nAt the very end of your response, after the JSON, please output your confidence score (0-100) in this format: 'ConfidenceScore: <score>'.",
            "zh": "\n\n请在回答的最后（JSON之后）按此格式给出置信度（0-100）：'ConfidenceScore: <score>'。"
        }
        
        modified_input = copy.deepcopy(input_dict)
        if "input_instruction" in modified_input:
             modified_input["input_instruction"] += confidence_prompt.get(language, confidence_prompt["en"])
             
        # Optimization: We don't need hidden states for this check
        if hasattr(self.vlm, "evaluate") and "output_hidden_states" in self.vlm.evaluate.__code__.co_varnames:
             response = self.vlm.evaluate(modified_input, language, output_hidden_states=False)
        else:
             response = self.vlm.evaluate(modified_input, language)
             
        content = response.get("origin_output", "")
        
        # Robust extraction looking for the specific tag
        try:
            match = re.search(r"ConfidenceScore:\s*(\d{1,3})", content)
            if match:
                score = float(match.group(1))
                if 0 <= score <= 100:
                    return score / 100.0
                    
            # Fallback: try finding just the last number if it looks like a score at the end
            # numbers = re.findall(r"(\d{1,3})", content)
            # if numbers:
            #     score = float(numbers[-1])
            #     if 0 <= score <= 100:
            #         return score / 100.0
        except:
            pass
            
        return 0.5 # Default fallback

    def get_p_true_confidence(self, input_dict: Dict[str, Any], previous_answer: str, language: str = "en") -> float:
        """
        Asks the model "Is the following answer correct?" and checks probability of 'True'/'Yes'.
        """
        check_prompt_en = f"\n\nQuestion: {input_dict.get('input_instruction', '')}\nProposed Answer: {previous_answer}\nIs this answer correct? Answer True or False."
        
        check_input = copy.deepcopy(input_dict)
        check_input["input_instruction"] = check_prompt_en
        
        # Optimization: We don't need hidden states for this check
        if hasattr(self.vlm, "evaluate") and "output_hidden_states" in self.vlm.evaluate.__code__.co_varnames:
             response = self.vlm.evaluate(check_input, language, output_hidden_states=False)
        else:
             response = self.vlm.evaluate(check_input, language)
        
        if "logprobs" in response and "all_logprobs" in response["logprobs"]:
            content = response.get("origin_output", "").strip().lower()
            if not content:
                return 0.0
                
            first_logprob = response["logprobs"]["all_logprobs"][0] if response["logprobs"]["all_logprobs"] else -100.0
            prob = np.exp(first_logprob)
            
            if "true" in content[:10]:
                return prob
            elif "false" in content[:10]:
                return 1.0 - prob
                
        return 0.0

    def get_self_check_consistency(self, input_dict: Dict[str, Any], language: str = "en", num_samples: int = 5) -> float:
        """
        Samples multiple responses and computes consistency (Jaccard similarity approximation).
        """
        answers = []
        
        # Optimization: Use evaluate_batch if available (1 pass instead of N passes)
        if hasattr(self.vlm, "evaluate_batch"):
            try:
                batch_results = self.vlm.evaluate_batch(input_dict, language, num_samples=num_samples)
                # Convert to string representation for comparison
                answers = [str(r) for r in batch_results]
            except Exception as e:
                print(f"Batch evaluation failed, falling back to sequential: {e}")
                
        if not answers:
            # Fallback to sequential
            for _ in range(num_samples):
                 # The evaluate method likely calls an API with default temperature (often 1.0 or 0.7).
                 # We assume it produces some diversity.
                 res = self.vlm.evaluate(input_dict, language)
                 ans = res.get("skill_sequence", [])
                 # Convert skill sequence to string for comparison or set of tokens
                 # Skill sequence is a list of dicts/strings. Let's canonicalize it.
                 ans_str = str(ans)
                 answers.append(ans_str)
             
        if not answers:
            return 0.0
            
        # Compute pairwise Jaccard similarity (on character/token n-grams or just exact match?)
        # For simplicity, let's use exact match ratio or simplified Jaccard on bag of words/skills.
        
        # Method: Unigram Jaccard on the string representation
        def get_ngrams(text, n=3):
            return set([text[i:i+n] for i in range(len(text)-n+1)])
            
        score_sum = 0
        pairs = 0
        for i in range(len(answers)):
            for j in range(i+1, len(answers)):
                set1 = get_ngrams(answers[i])
                set2 = get_ngrams(answers[j])
                if not set1 and not set2:
                    jaccard = 1.0
                elif not set1 or not set2:
                    jaccard = 0.0
                else:
                    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                score_sum += jaccard
                pairs += 1
                
        if pairs == 0:
            return 1.0
            
        return score_sum / pairs

    def get_hidden_state_probe(self, response_dict: Dict[str, Any]) -> float:
        """
        Requires hidden states from the model. 
        Currently mostly a placeholder as API wrappers don't return hidden states.
        """
        if "hidden_stats" in response_dict and response_dict["hidden_stats"]:
            # If we had a trained probe, we would run it here.
            # return probe(response_dict["hidden_stats"])
            pass
        return 0.0

    def get_ood_probe(self, response_dict: Dict[str, Any]) -> float:
        """
        Requires hidden states for Contrast-Consistent Search (CCS).
        Placeholder.
        """
        return 0.0
