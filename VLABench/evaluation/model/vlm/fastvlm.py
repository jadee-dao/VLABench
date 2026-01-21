from VLABench.evaluation.model.vlm.base import *
import torch
import torch.nn.functional as F
import json
import numpy as np

class FastVLM_1_5B(BaseVLM):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        model_id = "apple/FastVLM-1.5B"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Default sampling iterations for agreement check
        self.sampling_n = 5

    def evaluate(self, input_dict, language, with_CoT=False):
        ti_list = get_ti_list(input_dict, language, with_CoT=with_CoT)
        content = self.build_prompt_with_tilist(ti_list)

        # Basic Message Structure
        messages = [
            {"role": "user", "content": content}
        ]

        # Prepare inputs
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages) # Helper to extract images
        
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # --- 1. Main Generation (Greedy by default) with Hidden States Stats ---
        # We use greedy decoding for the main answer
        
        generate_kwargs = {
            "max_new_tokens": 512,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
            "do_sample": False  # Greedy for the "main" result
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        # Decode output
        generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        output = {}
        output["origin_output"] = output_text
        
        # Parse result
        try:
            # Try parsing JSON block first
            if "```json" in output_text:
                json_str = output_text.split("```json")[1].split("```")[0]
            elif "```" in output_text:
                 json_str = output_text.split("```")[1].split("```")[0]
            else:
                json_str = output_text
                
            parsed = json.loads(json_str)
            if "skill_sequence" in parsed:
                output["skill_sequence"] = parsed["skill_sequence"]
            else:
                 output["skill_sequence"] = parsed # Attempt to use direct list if that's what it is
        except Exception as e:
            output["format_error"] = str(e)

        # --- Calculate Statistics on Hidden States ---
        # outputs.hidden_states is a tuple of length 'generated_len'
        # Each element is a tuple of (num_layers + 1) tensors of shape (batch, 1, hidden_dim)
        
        stats = []
        if outputs.hidden_states:
            # We iterate over each generated token
            for i, step_hidden_states in enumerate(outputs.hidden_states):
                # step_hidden_states is (layer_0, layer_1, ... layer_N)
                # We ignore layer_0 (embeddings) usually? Or keep it? 
                # Request says "variance of each layer". I'll include all layers in the stack.
                
                token_stats = {
                    "token_idx": i,
                    "layer_stats": []
                }
                
                # Convert tuple to list of tensors for easier access
                # stacking -> (num_layers, 1, hidden_dim) -> (num_layers, hidden_dim) (batch=1)
                layers_tensor = torch.stack(step_hidden_states).squeeze(1).squeeze(1).float() # (L, D)
                
                num_layers = layers_tensor.shape[0]
                last_layer = layers_tensor[-1] 
                
                for layer_idx in range(num_layers):
                    layer_h = layers_tensor[layer_idx]
                    
                    # 1. Variance of the layer
                    var = torch.var(layer_h).item()
                    
                    # 2. Cosine Similarity to Next Layer
                    if layer_idx < num_layers - 1:
                        next_layer_h = layers_tensor[layer_idx + 1]
                        sim_next = F.cosine_similarity(layer_h.unsqueeze(0), next_layer_h.unsqueeze(0)).item()
                    else:
                        sim_next = 1.0 # Or None
                        
                    # 3. Cosine Similarity to Actual Output (Last Layer)
                    sim_last = F.cosine_similarity(layer_h.unsqueeze(0), last_layer.unsqueeze(0)).item()
                    
                    token_stats["layer_stats"].append({
                        "layer": layer_idx,
                        "variance": var,
                        "cos_sim_next": sim_next,
                        "cos_sim_last": sim_last
                    })
                stats.append(token_stats)
        
        output["hidden_stats"] = stats

        # --- 2. Multiple Sampling for Agreement ---
        # Run N times with sampling to check stability
        
        sample_results = []
        sample_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7, # Some randomness
            "top_p": 0.9,
        }
        
        for _ in range(self.sampling_n):
             with torch.no_grad():
                s_out = self.model.generate(**inputs, **sample_kwargs)
                s_ids = s_out[:, inputs.input_ids.shape[1]:]
                s_text = self.processor.batch_decode(s_ids, skip_special_tokens=True)[0]
                
                # Parse
                try:
                    if "```json" in s_text:
                        j = s_text.split("```json")[1].split("```")[0]
                    elif "```" in s_text:
                        j = s_text.split("```")[1].split("```")[0]
                    else:
                        j = s_text
                    s_parsed = json.loads(j)
                    if isinstance(s_parsed, dict) and "skill_sequence" in s_parsed:
                        s_seq = s_parsed["skill_sequence"]
                    else:
                        s_seq = s_parsed
                    
                    # Normalized string representation for comparison
                    sample_results.append(json.dumps(s_seq, sort_keys=True))
                except:
                    sample_results.append("error")

        if sample_results:
            # Count agreement
            from collections import Counter
            counts = Counter(sample_results)
            most_common_seq, count = counts.most_common(1)[0]
            agreement_rate = count / len(sample_results)
            output["agreement_rate"] = agreement_rate
            output["agreement_samples"] = sample_results # Debug info
        
        return output

    def build_prompt_with_tilist(self, ti_list):
        # Similar to Qwen2-VL format or FastVLM specific format?
        # Assuming standard ChatML or Qwen dictionary format for processor
        # Qwen-VL uses a list of dicts with 'type' and 'text'/'image'
        content = []
        for ti in ti_list:
            if ti[0] == "text":
                content.append({"type": "text", "text": ti[1]})
            elif ti[0] == "image":
                 content.append({"type": "image", "image": ti[1]})
        return content

    def process_vision_info(self, messages):
        # Helper to extract PIL images
        images = []
        videos = [] # Assuming no video for now
        for msg in messages:
            for content in msg["content"]:
                if content["type"] == "image":
                    images.append(content["image"])
        return images, None if not videos else videos

    def get_name(self):
        return "FastVLM_1_5B"
