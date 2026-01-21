
import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple

from VLABench.evaluation.model.vlm.base import *


class OpenRouter(BaseVLM):
    """Generic VLM wrapper for OpenRouter-hosted models."""

    CONFIG_FILENAME = "openrouter_models.json"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_alias: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY (OPENAI_API_KEY) is required to query OpenRouter models.")

        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model_alias, self.model_id = self._resolve_model_alias(model_alias)
        self._display_name = self._format_display_name(self.model_alias)

        super().__init__()

    # ----------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------
    def evaluate(self, input_dict, language, with_CoT=False):
        from VLABench.utils.gpt_utils import build_prompt_with_tilist, query_gpt4_v

        ti_list = get_ti_list(input_dict, language, with_CoT)
        prompt = build_prompt_with_tilist(ti_list)

        default_headers = {}
        referer = os.environ.get("OPENROUTER_SITE_URL")
        title = os.environ.get("OPENROUTER_APP_TITLE")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title

        completion_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model_id,
        }
        if default_headers:
            completion_kwargs["default_headers"] = default_headers

        breakpoint()

        content, logprobs = query_gpt4_v(prompt, logprobs=True, **completion_kwargs)
        output = {"origin_output": content}

        try:
            if logprobs:
                all_logprobs = [tok.logprob for tok in logprobs.content]
                total_logprob = sum(all_logprobs)
                nll = -total_logprob
                avg_nll = -total_logprob / len(all_logprobs)

                output["logprobs"] = {
                    "all_logprobs": all_logprobs,
                    "nll": nll,
                    "avg_nll": avg_nll,
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
    def _resolve_model_alias(self, preferred_alias: str | None) -> Tuple[str, str]:
        config = self._load_config()
        aliases = config.setdefault("aliases", {})
        default_alias = preferred_alias or os.environ.get("OPENROUTER_MODEL_ALIAS") or config.get("default_alias")

        if aliases:
            print("Available OpenRouter models:")
            for alias_key, model_id in aliases.items():
                print(f"  {alias_key}: {model_id}")
            selection = input(
                f"Select OpenRouter model alias [default: {config.get('default_alias') or 'new'}]: "
            ).strip()
            if not selection:
                selection = config.get("default_alias")
            if selection and selection in aliases:
                config["default_alias"] = selection
                self._save_config(config)
                return selection, aliases[selection]
            default_alias = selection

        alias_key, model_id = self._prompt_for_new_alias(default_alias)
        aliases[alias_key] = model_id
        config["default_alias"] = alias_key
        self._save_config(config)
        return alias_key, model_id

    def _ensure_alias(self, alias_key: str, aliases: Dict[str, str], config: Dict) -> Tuple[str, str]:
        if alias_key in aliases:
            config["default_alias"] = alias_key
            self._save_config(config)
            return alias_key, aliases[alias_key]

        model_id = os.environ.get("OPENROUTER_MODEL")
        if not model_id:
            print(f"No stored OpenRouter model for alias '{alias_key}'.")
            model_id = input(f"Enter OpenRouter model identifier for alias '{alias_key}': ").strip()
            while not model_id:
                model_id = input("Model identifier cannot be empty. Enter model identifier: ").strip()
        aliases[alias_key] = model_id
        config["aliases"] = aliases
        config["default_alias"] = alias_key
        self._save_config(config)
        return alias_key, model_id

    def _prompt_for_new_alias(self, suggested_alias: str | None = None) -> Tuple[str, str]:
        alias_prompt = "Enter a name for this OpenRouter model"
        if suggested_alias:
            alias_prompt += f" [{suggested_alias}]"
        alias_prompt += ": "

        alias_key = input(alias_prompt).strip()
        if not alias_key:
            alias_key = suggested_alias or "openrouter_model"

        model_id = input(
            f"Enter the OpenRouter model identifier (e.g., openai/gpt-4o-mini) for '{alias_key}': "
        ).strip()
        while not model_id:
            model_id = input("Model identifier cannot be empty. Enter the identifier: ").strip()
        return alias_key, model_id

    def _format_display_name(self, alias_key: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z]+", "_", alias_key).strip("_") or "model"
        return sanitized

    def _load_config(self) -> Dict:
        path = self._config_path()
        if not path.exists():
            return {"aliases": {}, "default_alias": None}
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"aliases": {}, "default_alias": None}

    def _save_config(self, config: Dict) -> None:
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _config_path(self) -> Path:
        root = os.environ.get("VLABENCH_ROOT")
        if not root:
            raise ValueError("VLABENCH_ROOT must be set to resolve OpenRouter configuration path.")
        return Path(root) / "configs" / self.CONFIG_FILENAME
