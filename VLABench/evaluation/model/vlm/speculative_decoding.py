from VLABench.evaluation.model.vlm.qwen_vl import Qwen2_VL
import torch
import json
import os
from VLABench.evaluation.model.vlm.base import get_ti_list

class Speculative_Decoding(Qwen2_VL):
    def __init__(self, model_path="qwen/Qwen2-VL-7B-Instruct", draft_model_name="Qwen/Qwen2-VL-2B-Instruct", **kwargs) -> None:
        # If kwargs contains draft_model_name (passed from CLI) and it is not None, use it.
        if kwargs.get('draft_model_name'):
            draft_model_name = kwargs['draft_model_name']
        
        # If kwargs contains model_path (passed from CLI) and it is not None, use it.
        if kwargs.get('model_path'):
            model_path = kwargs['model_path']

        super().__init__(model_path=model_path, draft_model_name=draft_model_name, **kwargs)

    def get_name(self):

        print(f"Loading draft model: {draft_model_name}")
        self.draft_model = None
        try:
             # Try to download if not local
            if not os.path.exists(draft_model_name):
                 try:
                    draft_model_dir = snapshot_download(draft_model_name)
                 except:
                    # Fallback or assume it's a HF ID
                    draft_model_dir = draft_model_name
            else:
                draft_model_dir = draft_model_name
            
            self.draft_model = Qwen2VLForConditionalGeneration.from_pretrained(
                draft_model_dir,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            print(f"Draft model {draft_model_name} loaded successfully.")
        except Exception as e:
            print(f"Failed to load draft model {draft_model_name}: {e}")
            raise e

    def evaluate(self, input_dict, language, with_CoT=False):
        from qwen_vl_utils import process_vision_info
        
        ti_list = get_ti_list(input_dict, language, with_CoT=with_CoT)
        content = self.build_prompt_with_tilist(ti_list)

        # Messages containing multiple images and a text query
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference with speculative decoding
        generate_kwargs = {
            "max_new_tokens": 128,
        }
        if self.draft_model:
            generate_kwargs["assistant_model"] = self.draft_model
            
        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        output = {}
        output["origin_output"] = output_text[0]
        try:
            json_data = output_text[0].split("```json")[1].split("```")[0]
            output["skill_sequence"] = json.loads(json_data)
        except:
            output["format_error"] = "format_error"
        return output

    def get_name(self):
        return "Speculative_Decoding"
