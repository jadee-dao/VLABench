
from transformers import AutoProcessor, AutoConfig
import torch

model_id = "apple/FastVLM-1.5B"
print(f"Testing load for {model_id}")

try:
    print("Attempting to load config...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print("Config loaded:", config)
except Exception as e:
    print("Config load failed:", e)

try:
    print("Attempting to load processor...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Processor loaded successfully")
except Exception as e:
    print("Processor load failed:", e)
    import traceback
    traceback.print_exc()
