#!/bin/bash

# Estimated VRAM usage per model in GB (approximate; adjust as needed)
declare -A model_mem_map=(
    ["facebook/opt-125m"]=2
    ["facebook/opt-350m"]=3
    ["facebook/opt-1.3b"]=6
    ["facebook/opt-2.7b"]=10
    ["facebook/opt-6.7b"]=20
    ["facebook/opt-13b"]=35
)

# Models to consider
draft_models=("facebook/opt-125m" "facebook/opt-350m" "facebook/opt-1.3b")
main_models=("facebook/opt-2.7b" "facebook/opt-6.7b" "facebook/opt-13b")

# Other search params
# batch_sizes=(1 4 8 16)  # Commented out batch sizes
num_spec_tokens=(1 4 8)
prompt_lengths=(64 128 256)
subset=100  # you can increase for more realistic benchmarking

# Output base dir
OUTPUT_DIR="benchmark_grid_results"
mkdir -p "$OUTPUT_DIR"

# Incompatible with FlashAttention due to head size issues
incompatible_flash_models=("facebook/opt-125m" "facebook/opt-350m" "facebook/opt-2.7b")

# Begin grid search
for draft in "${draft_models[@]}"; do
    for main in "${main_models[@]}"; do

        # Skip models incompatible with FlashAttention
        if [[ " ${incompatible_flash_models[@]} " =~ " $draft " ]]; then
            echo "Skipping incompatible draft model: $draft"
            continue
        fi

        if [[ " ${incompatible_flash_models[@]} " =~ " $main " ]]; then
            echo "Skipping incompatible main model: $main"
            continue
        fi

        # Estimate memory usage
        draft_mem=${model_mem_map[$draft]}
        main_mem=${model_mem_map[$main]}
        total_mem=$((draft_mem + main_mem))

        if (( total_mem > 40 )); then
            echo "Skipping combination $draft + $main (estimated VRAM: ${total_mem}GB > 40GB)"
            continue
        fi

        # for batch in "${batch_sizes[@]}"; do  # Commented out batch loop
            for num_spec in "${num_spec_tokens[@]}"; do
                for prompt_len in "${prompt_lengths[@]}"; do

                    echo "Running: Draft=$draft | Main=$main | Spec=$num_spec | PromptLen=$prompt_len"  # Removed Batch from echo

                    python speculative_decode.py \
                        --model "$main" \
                        --draft_model "$draft" \
                        --num_speculative_tokens "$num_spec" \
                        --prompt_length "$prompt_len" \
                        --subset "$subset" \
                        --output_dir "$OUTPUT_DIR"

                done
            done
        # done  # Commented out batch loop

    done
done
