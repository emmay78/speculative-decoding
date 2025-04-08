from vllm import LLM, SamplingParams
from datasets import load_dataset, DownloadConfig
import time
import random
import argparse
import statistics
import csv
import os
import json
from transformers import AutoTokenizer

CACHE_DIR = "/n/netscratch/idreos_lab/Lab/emyang/cs2241/.cache"


def save_json(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    json_file = os.path.join(output_dir, f"args_{timestamp}.json")
    with open(json_file, "w") as f:
        json.dump(config, f, indent=2)


def measure_metrics(prompts, outputs, start_time, end_time, config, output_dir):
    total_time = end_time - start_time
    avg_time_per_prompt = total_time / len(prompts)

    total_tokens = sum(
        len(output.outputs[0].text.strip().split()) for output in outputs
    )
    tokens_per_second = total_tokens / total_time

    print("--- Metrics ---")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per prompt: {avg_time_per_prompt:.2f} seconds")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "benchmark_results.csv")
    file_exists = os.path.isfile(results_file)
    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "model",
                    "draft_model",
                    "num_speculative_tokens",
                    "subset",
                    "temperature",
                    "top_p",
                    # "batch_size",
                    "prompt_length",
                    "total_time",
                    "avg_time_per_prompt",
                    "total_tokens",
                    "tokens_per_second",
                ]
            )
        writer.writerow(
            [
                config["model"],
                config["draft_model"],
                config["num_speculative_tokens"],
                config["subset"],
                config["temperature"],
                config["top_p"],
                # config["batch_size"],
                config["prompt_length"],
                total_time,
                avg_time_per_prompt,
                total_tokens,
                tokens_per_second,
            ]
        )

    # Save config to JSON
    save_json(config, output_dir)


def main(
    model="facebook/opt-6.7b",
    draft_model="facebook/opt-125m",
    num_speculative_tokens=5,
    subset=None,
    temperature=0.8,
    top_p=0.95,
    output_dir="benchmark_output",
    # batch_size=1,
    prompt_length=100,
):
    random.seed(2241)

    download_config = DownloadConfig(max_retries=5, resume_download=True)
    dataset = load_dataset(
        "tatsu-lab/alpaca",
        cache_dir=CACHE_DIR,
        download_config=download_config,
        split="train",
    )
    dataset = dataset.shuffle(seed=2241)

    tokenizer = AutoTokenizer.from_pretrained(model)

    prompts = []
    for example in dataset:
        tokens = tokenizer.encode(
            example["instruction"], truncation=True, max_length=prompt_length
        )
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        prompts.append(decoded)

    if subset is not None:
        prompts = prompts[:subset]

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p)

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        speculative_config={
            "method": "ngram",
            "model": draft_model,
            "num_speculative_tokens": num_speculative_tokens,
        },
        download_dir=CACHE_DIR,
    )

    start_time = time.time()
    outputs = []
    # for i in range(0, len(prompts), batch_size):
    #     batch_prompts = prompts[i : i + batch_size]
    #     batch_outputs = llm.generate(batch_prompts, sampling_params)
    #     outputs.extend(batch_outputs)
    for prompt in prompts:
        output = llm.generate([prompt], sampling_params)
        outputs.extend(output)
    end_time = time.time()

    config = {
        "model": model,
        "draft_model": draft_model,
        "num_speculative_tokens": num_speculative_tokens,
        "subset": subset,
        "temperature": temperature,
        "top_p": top_p,
        # "batch_size": batch_size,
        "prompt_length": prompt_length,
    }

    measure_metrics(prompts, outputs, start_time, end_time, config, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speculative decoding benchmark.")
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--draft_model", type=str, default="facebook/opt-125m")
    parser.add_argument("--num_speculative_tokens", type=int, default=5)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_dir", type=str, default="benchmark_output")
    # parser.add_argument(
    #     "--batch_size", type=int, default=1, help="Batch size for prompt generation"
    # )
    parser.add_argument(
        "--prompt_length", type=int, default=100, help="Max prompt length (in tokens)"
    )
    args = parser.parse_args()

    main(
        model=args.model,
        draft_model=args.draft_model,
        num_speculative_tokens=args.num_speculative_tokens,
        subset=args.subset,
        temperature=args.temperature,
        top_p=args.top_p,
        output_dir=args.output_dir,
        # batch_size=args.batch_size,
        prompt_length=args.prompt_length,
    )
