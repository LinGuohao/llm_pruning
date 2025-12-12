#!/bin/bash

# Evaluate the best individual found by genetic algorithm
# Usage: ./genetic/run_eval_best.sh <path_to_best_individual.json> [gpu_id]

# Check if best_individual.json path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_best_individual.json> [gpu_id]"
    echo ""
    echo "Example:"
    echo "  $0 genetic/checkpoints/20240315-123456/best_individual.json 0"
    echo ""
    echo "If gpu_id is not provided, will use GPU 0 by default"
    exit 1
fi

BEST_INDIVIDUAL_PATH=$1
GPU_ID=${2:-0}  # Default to GPU 0 if not specified

# Check if file exists
if [ ! -f "$BEST_INDIVIDUAL_PATH" ]; then
    echo "Error: File not found: $BEST_INDIVIDUAL_PATH"
    exit 1
fi

# Model path (change this to your actual model path)
MODEL_PATH="/path/to/your/Llama-2-13b-hf"

# Check if MODEL_PATH exists, if not, try to find it from run_evolution.sh
if [ ! -d "$MODEL_PATH" ]; then
    echo "Warning: MODEL_PATH not set correctly in run_eval_best.sh"
    echo "Please edit the script and set MODEL_PATH to your Llama-2-13b model path"
    echo ""
    echo "Trying to extract model path from run_evolution.sh..."

    if [ -f "genetic/run_evolution.sh" ]; then
        EXTRACTED_PATH=$(grep "MODEL_PATH=" genetic/run_evolution.sh | head -1 | cut -d'=' -f2 | tr -d '"')
        if [ -n "$EXTRACTED_PATH" ] && [ -d "$EXTRACTED_PATH" ]; then
            MODEL_PATH=$EXTRACTED_PATH
            echo "Found model path: $MODEL_PATH"
        else
            echo "Could not find valid model path in run_evolution.sh"
            exit 1
        fi
    else
        echo "run_evolution.sh not found"
        exit 1
    fi
fi

echo "=========================================="
echo "Evaluating Best Individual"
echo "=========================================="
echo "Best individual file: $BEST_INDIVIDUAL_PATH"
echo "Model path: $MODEL_PATH"
echo "GPU ID: $GPU_ID"
echo "=========================================="
echo ""

# Run evaluation
python genetic/eval_best_individual.py \
    --model_path "$MODEL_PATH" \
    --best_individual_path "$BEST_INDIVIDUAL_PATH" \
    --gpu_id "$GPU_ID" \
    --seqlen 2048 \
    --eval_samples 128

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
