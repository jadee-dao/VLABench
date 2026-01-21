
import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from VLABench.evaluation.model.vlm.base import *
from transformers import AutoProcessor
from vllm import LLM

SPECIAL_TOKENS = {
    "<|image_pad|>", "<|vision_start|>", "<|vision_end|>",
    "<|video_pad|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"
}


class Local(BaseVLM):
    """Generic VLM wrapper for vLLM-setup models."""

    CONFIG_FILENAME = "openrouter_models.json"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_alias: str | None = None,
    ) -> None:
        self.api_key = api_key or "EMPTY"
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY (OPENAI_API_KEY) is required to query OpenRouter models.")

        self.base_url = base_url or "http://localhost:8000/v1"

        self.model_id = self._resolve_model_alias(model_alias)
        self._display_name = self._format_display_name(self.model_id)

        super().__init__()

    # ----------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------
    def evaluate(self, input_dict, language, with_CoT=False):
        from VLABench.utils.gpt_utils import build_prompt_with_tilist, generate_resp_with_vllm

        ti_list = get_ti_list(input_dict, language, with_CoT)
        prompt = build_prompt_with_tilist(ti_list)

        completion_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model_id,
            "chat_completion_client": "client.completions.create",
            "processor": AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        }

        content, logprobs = generate_resp_with_vllm(prompt, logprobs=True, **completion_kwargs)
        output = {"origin_output": content}

        try:
            if logprobs:
                # remove all logprobs that are None or correspond to special tokens
                cleaned_logprobs = []
                all_logprobs = []

                for tok in logprobs:
                    if not tok:
                        continue
                    attr = list(tok.values())[0]
                    all_logprobs.append(attr.get("logprob", None))
                    if attr.get("decoded_token") not in SPECIAL_TOKENS:
                        cleaned_logprobs.append(attr["logprob"])

                total_logprob = sum(cleaned_logprobs)
                nll = -total_logprob
                avg_nll = -total_logprob / len(cleaned_logprobs)

                avg_token_prob = float(np.exp(total_logprob / len(cleaned_logprobs)))

                output["logprobs"] = {
                    "all_logprobs": all_logprobs,
                    "nll": nll,
                    "avg_nll": avg_nll,
                    "avg_token_prob": avg_token_prob
                }

            if "```json" in content:
                json_data = content.split("```json")[1].split("```")[0]
            else: # try to parse the whole content as json
                json_data = content
            output["skill_sequence"] = json.loads(json_data)

        except Exception as e:
            output["format_error"] = "format_error"
            print(e)

        return output

    def get_name(self):
        return self._display_name

    # ----------------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------------

    def _resolve_model_alias(self, preferred_alias: str | None) -> str:
        
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        models = client.models.list()

        default_alias = preferred_alias or models.data[0].id

        aliases = [m.id for m in models.data]
        if len(aliases) == 0:
            raise ValueError("No models available from vLLM endpoint.")
        elif len(aliases) > 1:
            print("Available OpenRouter models:")
            for model_id in aliases:
                print(f"  {model_id}")
            selection = input(
                f"Select OpenRouter model alias [default: {default_alias}]: "
            ).strip()
            if selection in aliases:
                return selection
            else:
                print(f"Invalid selection '{selection}'. Using default alias '{default_alias}'.")

        return default_alias

    def _format_display_name(self, alias_key: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z]+", "_", alias_key).strip("_") or "model"
        return sanitized