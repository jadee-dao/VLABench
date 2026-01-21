
import os
import sys
import torch
from PIL import Image
import numpy as np

# Ensure VLABench is in path
sys.path.append(os.getcwd())

# Mock open3d to avoid dependency issues during lightweight testing
from unittest.mock import MagicMock
sys.modules["open3d"] = MagicMock()

from VLABench.evaluation.model.vlm import FastVLM_1_5B

def test_fastvlm():
    print("Initializing FastVLM 1.5B...")
    try:
        model = FastVLM_1_5B()
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    print("Model initialized.")

    # Create dummy image
    img = Image.new('RGB', (224, 224), color = 'red')
    
    # Create dummy input dict
    input_dict = {
        "pre_prompt": "You are a robot.",
        "input_pic": "dummy_path.jpg", # Path handling might need actual file or PIL object
        "input_pic_gt": "dummy_path_gt.jpg",
        "input_instruction": "Describe this image.",
        "shot_output": {}, # Optional depending on usage
    }
    
    # We need to hack the base.py generic 'get_ti_list' which tries to open image paths
    # OR we can manually construct the ti_list and pass it if we were calling internal methods,
    # but evaluate() calls get_ti_list.
    # So we should save the dummy image to a temporary path.
    img.save("dummy_test_image.jpg")
    input_dict["input_pic"] = os.path.abspath("dummy_test_image.jpg")
    input_dict["input_pic_gt"] = os.path.abspath("dummy_test_image.jpg") # Reuse

    print("Running evaluate...")
    try:
        output = model.evaluate(input_dict, language="en")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print("Evaluation successful.")
    print("Output keys:", output.keys())
    
    if "hidden_stats" in output:
        stats = output["hidden_stats"]
        print(f"Stats collected for {len(stats)} tokens.")
        if len(stats) > 0:
            print("Sample stats for first token:", stats[0])
            # Verify structure
            t0 = stats[0]
            assert "layer_stats" in t0
            assert "variance" in t0["layer_stats"][0]
            assert "cos_sim_next" in t0["layer_stats"][0]
    else:
        print("ERROR: 'hidden_stats' missing from output.")

    if "agreement_rate" in output:
        print(f"Agreement Rate: {output['agreement_rate']}")
        print("Samples:", output.get("agreement_samples", []))
    else:
        print("ERROR: 'agreement_rate' missing from output.")

    # Cleanup
    if os.path.exists("dummy_test_image.jpg"):
        os.remove("dummy_test_image.jpg")

if __name__ == "__main__":
    test_fastvlm()
