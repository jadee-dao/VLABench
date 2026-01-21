from VLABench.evaluation.model.vlm.base import *
import torch
import torch.nn.functional as F
import json
import numpy as np

class Qwen2_VL(BaseVLM):
    def __init__(self, model_path="Qwen/Qwen2-VL-7B-Instruct", system_prompt=None, **kwargs) -> None:
        self.model_path = model_path
        super().__init__()
        
        from transformers import AutoModelForVision2Seq, AutoProcessor
        # from modelscope import snapshot_download # Removed modelscope dependency for flexibility, rely on HF local or hub
        
        print(f"Loading Qwen2_VL model from: {model_path}")
        
        # We recommend enabling flash_attention_2 for better acceleration and memory saving
        try:
             self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Failed to load with flash_attention_2 ({e}). Falling back to default.")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.system_prompt = system_prompt
        
        # If pixels are passed in kwargs, handle them? Current Qwen2VL handles resolution dynamically.

    def evaluate(self, input_dict, language, with_CoT=False, output_hidden_states=True):
        from qwen_vl_utils import process_vision_info
        ti_list = get_ti_list(input_dict, language, with_CoT=with_CoT)
        
        content = self.build_prompt_with_tilist(ti_list)

        # Messages containing multiple images and a text query
        messages = []
        if self.system_prompt:
             messages.append({"role": "system", "content": self.system_prompt})
             
        messages.append({
                "role": "user",
                "content": content,
            })

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Determine device from model
        device = self.model.device
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {
            "max_new_tokens": 16384,
            "output_hidden_states": output_hidden_states,
            "return_dict_in_generate": True,
            "do_sample": True, 
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.1, # Explicitly neutral, but ready to throttle if needed
            "eos_token_id": [151645, 151643], # <|im_end|>, <|endoftext|>
            "output_scores": True,
            "streamer": streamer
        }
        
        import threading
        result_container = {}
        def run_generate():
            with torch.no_grad():
                result_container['outputs'] = self.model.generate(**inputs, **generate_kwargs)

        thread = threading.Thread(target=run_generate)
        thread.start()
        
        generated_text = ""
        # print("Generating...", end="", flush=True) # Optional prompt
        for new_text in streamer:
            print(new_text, end="", flush=True)
            generated_text += new_text
        print("\n") # Newline after generation
        
        thread.join()
        outputs = result_container['outputs']
        
        # Wait, if we need `outputs` (specifically scores/hidden states), standard Streamer usage makes it hard to capture the return value from the thread easily without a wrapper.
        # BUT, since we are debugging "Thinking" mode loops, we *really* need to see text.
        # Let's use a mutable container to capture the return value.
        
        result_container = {}
        def run_generate():
            with torch.no_grad():
                result_container['outputs'] = self.model.generate(**inputs, **generate_kwargs)

        thread = threading.Thread(target=run_generate)
        thread.start()
        
        generated_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            generated_text += new_text
        
        thread.join()
        outputs = result_container['outputs']

        generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Extract logprobs manually to avoid import issues
        # scores is tuple of (batch, vocab) tensors
        # generated_ids is (batch, new_tokens)
        
        valid_logprobs = []
        try:
             # Calculate transition scores manually
             # scores[i] corresponds to the i-th generated token
             gen_ids_only = outputs.sequences[:, inputs.input_ids.shape[1]:]
             
             for i, score_tensor in enumerate(outputs.scores if outputs.scores else []):
                 if i >= gen_ids_only.shape[1]: 
                     break
                 
                 # log_softmax
                 log_probs = F.log_softmax(score_tensor, dim=-1) # (batch, vocab)
                 
                 # Gather logprob of the chosen token
                 token_id = gen_ids_only[:, i].unsqueeze(-1) # (batch, 1)
                 token_lp = torch.gather(log_probs, -1, token_id).squeeze(-1) # (batch,)
                 
                 val = token_lp.item()
                 if not np.isinf(val) and not np.isnan(val):
                    valid_logprobs.append(val)
        except Exception as e:
            print(f"Error calculating logprobs: {e}")
        
        # log_probs = transition_scores[0].cpu().numpy()
        # valid_logprobs = [float(lp) for lp in log_probs if not np.isinf(lp) and not np.isnan(lp)]
        
        avg_logprob = np.mean(valid_logprobs) if valid_logprobs else 0.0
        avg_token_prob = np.exp(avg_logprob)
        avg_nll = -avg_logprob

        output = {}
        output["origin_output"] = output_text
        output["logprobs"] = {
            "avg_token_prob": avg_token_prob,
            "avg_nll": avg_nll,
            "all_logprobs": valid_logprobs
        }
        
        try:
            if "```json" in output_text:
                json_data = output_text.split("```json")[1].split("```")[0]
            elif "```" in output_text:
                 json_data = output_text.split("```")[1].split("```")[0]
            else:
                 json_data = output_text
                 
            output["skill_sequence"] = json.loads(json_data)
        except:
            output["format_error"] = "format_error"
            
        # Hidden Stats Extraction (Ported from FastVLM)
        stats = []
        if outputs.hidden_states:
            for i, step_hidden_states in enumerate(outputs.hidden_states):
                # step_hidden_states is a tuple of (num_layers+1) tensors
                token_stats = {
                    "token_idx": i,
                    "layer_stats": []
                }
                
                # Stack layers
                try:
                    raw_stack = torch.stack(step_hidden_states)
                    # Force flatten to (L, D) - collapsing batch/seq dims which should be 1 for generation steps
                    layers_tensor = raw_stack.view(raw_stack.shape[0], -1).float()
                except Exception as e:
                     print(f"Error processing hidden states shape: {e}")
                     continue
                
                # print(f"DEBUG: Layers Tensor shape: {layers_tensor.shape}")
                
                num_layers = layers_tensor.shape[0]
                last_layer = layers_tensor[-1]
                
                for layer_idx in range(num_layers):
                    layer_h = layers_tensor[layer_idx]
                    var = torch.var(layer_h).item()
                    
                    if layer_idx < num_layers - 1:
                        next_layer_h = layers_tensor[layer_idx + 1]
                        sim_next = F.cosine_similarity(layer_h.unsqueeze(0), next_layer_h.unsqueeze(0)).item()
                    else:
                        sim_next = 1.0
                        
                    sim_last = F.cosine_similarity(layer_h.unsqueeze(0), last_layer.unsqueeze(0)).item()
                    
                    token_stats["layer_stats"].append({
                        "layer": layer_idx,
                        "variance": var,
                        "cos_sim_next": sim_next,
                        "cos_sim_last": sim_last
                    })
                stats.append(token_stats)
        
        output["hidden_stats"] = stats
        
        return output
        
    def evaluate_batch(self, input_dict, language, num_samples=5, with_CoT=False):
        """
        Batched evaluation for consistency checks.
        """
        from qwen_vl_utils import process_vision_info
        ti_list = get_ti_list(input_dict, language, with_CoT=with_CoT)
        content = self.build_prompt_with_tilist(ti_list)
        
        messages = []
        if self.system_prompt:
             messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        generate_kwargs = {
            "max_new_tokens": 512,
            "output_hidden_states": False, # save memory for batch
            "return_dict_in_generate": True,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "num_return_sequences": num_samples
        }
        
        with torch.no_grad():
             outputs = self.model.generate(**inputs, **generate_kwargs)
             
        # Decode all sequences
        generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Parse all
        parsed_results = []
        for txt in output_texts:
            try:
                if "```json" in txt:
                    json_data = txt.split("```json")[1].split("```")[0]
                elif "```" in txt:
                     json_data = txt.split("```")[1].split("```")[0]
                else:
                     json_data = txt
                parsed_results.append(json.loads(json_data))
            except:
                parsed_results.append(txt)
                
        return parsed_results

    def build_prompt_with_tilist(self, ti_list):
        content = []
        for ti in ti_list:
            if ti[0] == "text":
                content.append({"type": "text", "text": ti[1]})
            elif ti[0] == "image":
                content.append({"type": "image", "image": ti[1]})
        return content
    
    def get_name(self):
        return f"Qwen2_VL_{self.model_path.split('/')[-1]}"

# Generic wrapper alias if needed, or just usage of Qwen2_VL with args
class Qwen2_VL_Generic(Qwen2_VL):
    def __init__(self, model_path, system_prompt=None, **kwargs):
        super().__init__(model_path=model_path, system_prompt=system_prompt, **kwargs)
        self.name = f"Qwen2_VL_{os.path.basename(model_path)}"

class Qwen3_VL_Generic(Qwen2_VL):
    def __init__(self, model_path, system_prompt=None, **kwargs):
        super().__init__(model_path=model_path, system_prompt=system_prompt, **kwargs)
        self.name = f"Qwen3_VL_{os.path.basename(model_path)}"
    
class Qwen2_VL_7B_Instruct(Qwen2_VL):
    def __init__(self):
        super().__init__(model_path="Qwen/Qwen2.5-VL-7B-Instruct")

class Qwen2_VL_3B_Instruct(Qwen2_VL):
    def __init__(self):
        super().__init__(model_path="Qwen/Qwen2.5-VL-3B-Instruct")
        
class Qwen2_VL_Thinking(Qwen2_VL):
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        system_prompt = "You are a helpful assistant. You must think step-by-step about the problem before providing the final JSON answer. Detail your reasoning process clearly."
        super().__init__(model_path=model_path, system_prompt=system_prompt)