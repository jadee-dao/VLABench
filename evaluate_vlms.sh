#bin/bash

export OPENAI_API_KEY="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" # replace with your API key
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

vlms=("Gemini" "Claude" "GPT4_v") # add more VLMs here
for vlm in "${vlms[@]}"; do
    echo "Evaluate $vlm"
    python scripts/evaluate_vlm.py \
        --vlm_name $vlm 
done