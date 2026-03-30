#!/usr/bin/env python3
"""
Benchmark script for vLLM API
Measures: throughput, latency, memory usage, token generation speed
"""

import requests
import time
import json
import subprocess
import re
from pathlib import Path

VLLM_URL = "http://localhost:8000"
MODEL = "Qwen/Qwen3-8B"
PROMPTS_FILE = Path(__file__).parent / "prompts.txt"
RESULTS_FILE = Path(__file__).parent / "results" / "vllm_results.json"


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmemused", "-d", "0"],
            capture_output=True, text=True, timeout=5
        )
        match = re.search(r'(\d+)\s*MB', result.stdout)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None


def load_prompts():
    """Load prompts from file"""
    prompts = {}
    current_category = None
    with open(PROMPTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_category = line[1:-1]
                prompts[current_category] = []
            elif line and current_category:
                prompts[current_category].append(line)
    return prompts


def generate_with_vllm(prompt, max_tokens=512):
    """Generate response using vLLM OpenAI-compatible API"""
    url = f"{VLLM_URL}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    # Measure time to first token
    start_time = time.perf_counter()
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    end_time = time.perf_counter()

    result = response.json()
    total_time = end_time - start_time

    if "error" in result:
        return {"error": result["error"]}

    choices = result.get("choices", [])
    if not choices:
        return {"error": "No choices in response"}

    response_text = choices[0].get("text", "")
    usage = result.get("usage", {})
    tokens = usage.get("completion_tokens", len(response_text.split()))
    prompt_tokens = usage.get("prompt_tokens", 0)

    return {
        "response": response_text,
        "total_time_sec": total_time,
        "tokens_generated": tokens,
        "prompt_tokens": prompt_tokens,
        "tokens_per_sec": tokens / total_time if total_time > 0 else 0,
        "gpu_memory_mb": get_gpu_memory()
    }


def run_benchmarks():
    """Run all benchmarks"""
    prompts = load_prompts()

    print(f"vLLM Benchmark ({MODEL})")
    print("=" * 50)

    # Check if vLLM is running
    try:
        requests.get(f"{VLLM_URL}/v1/models", timeout=5)
    except Exception as e:
        print(f"ERROR: vLLM not running at {VLLM_URL}")
        print(f"Start with: python -m vllm.entrypoints.openai.api_server ...")
        return None

    results = {}
    for category, category_prompts in prompts.items():
        print(f"\n[{category}]")
        category_results = []
        for i, prompt in enumerate(category_prompts):
            print(f"  Prompt {i+1}...", end=" ")
            result = generate_with_vllm(prompt)
            if "error" in result:
                print(f"ERROR: {result['error']}")
                continue
            print(f"{result['tokens_per_sec']:.1f} tokens/sec")
            category_results.append({
                "prompt_preview": prompt[:50] + "...",
                **result
            })
        results[category] = category_results

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    total_tokens = sum(sum(r["tokens_generated"] for r in results[c]) for c in results)
    total_time = sum(sum(r["total_time_sec"] for r in results[c]) for c in results)
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average throughput: {avg_tps:.1f} tokens/sec")

    return results


if __name__ == "__main__":
    run_benchmarks()
