#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Define model pairs (draft_model main_model)
declare -a model_pairs=(
  "gpt2-medium gpt2-large"
  "distilbert-base-uncased bert-base-uncased"
  "facebook/opt-125m facebook/opt-6.7b"
  "facebook/opt-1.3b facebook/opt-6.7b"
)

# Other search params
batch_sizes=(64 128 256)
num_spec_tokens=(1 4 8)
prompt_lengths=(128 256 512)

# Output base dir
OUTPUT_DIR="benchmark_results_v2"
mkdir -p "$OUTPUT_DIR"

# Begin search over model pairs
for pair in "${model_pairs[@]}"; do
  draft_model=$(echo $pair | awk '{print $1}')
  main_model=$(echo $pair | awk '{print $2}')

  for batch in "${batch_sizes[@]}"; do
    for num_spec in "${num_spec_tokens[@]}"; do
      for prompt_len in "${prompt_lengths[@]}"; do

        echo "Running: Draft=$draft_model | Main=$main_model | Batch=$batch | Spec=$num_spec | PromptLen=$prompt_len"

        python speculative_decode.py \
          --model "$main_model" \
          --draft_model "$draft_model" \
          --num_speculative_tokens "$num_spec" \
          --batch_size "$batch" \
          --prompt_length "$prompt_len" \
          --output_dir "$OUTPUT_DIR"

      done
    done
  done

done