
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

try:
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Mock Input Data
    # Qwen2-VL expects list of messages
    # Content is list of dicts with type 'text' or 'image'
    
    # Text-only
    messages_text = [
        {"role": "user", "content": [{"type": "text", "text": "Hello world"}]}
    ]
    
    # Image + Text
    # We need a dummy image. 
    # Create a small blank image and save it or use a URL if internet allowed (likely not relying on it is safer).
    # Or just use a valid local path check.
    # Actually, let's create a temporary white image.
    from PIL import Image
    import os
    img_path = "test_image.jpg"
    Image.new('RGB', (100, 100), color='white').save(img_path)
    
    messages_vision = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": "Describe this image."}
            ]
        }
    ]

    print("\n--- Testing Apply Chat Template ---")
    text = processor.apply_chat_template(messages_vision, tokenize=False, add_generation_prompt=True)
    print("Generated Prompt:")
    print(text)
    
    print("\n--- Testing Process Vision Info ---")
    image_inputs, video_inputs = process_vision_info(messages_vision)
    print(f"Image Inputs Type: {type(image_inputs)}")
    print(f"Image Inputs Len: {len(image_inputs) if image_inputs else 'None'}")
    
    print("\n--- Testing Processor Call ---")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    print("Processor Output Keys:", inputs.keys())
    print("Input IDs Shape:", inputs.input_ids.shape)
    if "pixel_values" in inputs:
        print("Pixel Values Shape:", inputs.pixel_values.shape)
    if "image_grid_thw" in inputs:
         print("Image Grid THW:", inputs.image_grid_thw)
         
    print("\n✅ Processor verification successful! Format appears compatible.")

except Exception as e:
    print(f"\n❌ Processor verification FAILED: {e}")
    import traceback
    traceback.print_exc()
