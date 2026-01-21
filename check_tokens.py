from transformers import AutoProcessor
import os

model_path = "Qwen/Qwen3-VL-4B-Thinking"
try:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"Type: {type(processor)}")
    
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor 

        
    print(f"EOS Token: {tokenizer.eos_token}")
    print(f"EOS Token ID: {tokenizer.eos_token_id}")
    print(f"Pad Token: {tokenizer.pad_token}")
    print(f"Pad Token ID: {tokenizer.pad_token_id}")
    # print(f"Chat Template: {tokenizer.chat_template}")
    print(f"Special Tokens Map: {tokenizer.special_tokens_map}")
except Exception as e:
    print(e)
