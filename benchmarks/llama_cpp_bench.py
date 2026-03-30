#!/usr/bin/env python3
"""
Benchmark script for llama.cpp CLI (llama-cli)
Supports both Vulkan and ROCm backends
Measures: throughput, latency, memory usage, token generation speed
"""

import subprocess
import time
import json
import re
import os
import signal
from pathlib import Path

# Use downloaded GGUF model
MODEL_PATH = str(Path.home() / "llama.cpp/models/Qwen3-8B-Q4_K_M.gguf")
PROMPTS_FILE = Path(__file__).parent / "prompts.txt"
RESULTS_DIR = Path(__file__).parent / "results"


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


def find_llama_cli():
    """Find llama-cli binary"""
    paths = [
        str(Path.home() / "llama.cpp/build/bin/llama-cli"),
        "./llama.cpp/build/bin/llama-cli",
        "/usr/local/bin/llama-cli",
        "/usr/bin/llama-cli"
    ]
    for p in paths:
        if Path(p).exists():
            return p
    return None


def generate_with_llama_cli(prompt, backend="vulkan", n_ctx=4096, max_tokens=128):
    """Generate response using llama-cli directly"""
    llama_cli = find_llama_cli()
    if not llama_cli:
        return {"error": "llama-cli not found. Build with: cmake .. -DGGML_VULKAN=ON -DGGML_HIP=ON && make -j$(nproc)"}

    if not Path(MODEL_PATH).exists():
        return {"error": f"Model not found at {MODEL_PATH}"}

    # Set GPU backend env
    env = os.environ.copy()
    if backend == "vulkan":
        env["GGML_VULKAN"] = "1"
    elif backend == "rocm":
        env["GGML_HIPBLAS"] = "1"
        env["HIP_VISIBLE_DEVICES"] = "0"
    else:
        # CPU fallback
        pass

    # Use pipe input for non-interactive mode
    cmd = [
        llama_cli,
        "-m", MODEL_PATH,
        "-p", prompt[:500],  # Truncate long prompts
        "-c", str(n_ctx),
        "-n", str(max_tokens),
        "-ngl", "99",  # GPU layers
        "-tb", "4",
        "-t", "8",
        "--log-disable",
        "--simple-io"
    ]

    start_time = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        stdout, stderr = proc.communicate(timeout=90)
        end_time = time.perf_counter()
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        end_time = time.perf_counter()
        return {"error": "Timeout", "total_time_sec": end_time - start_time}

    total_time = end_time - start_time
    output = stdout.decode('utf-8', errors='replace')
    err_output = stderr.decode('utf-8', errors='replace')

    if proc.returncode != 0 and not output:
        return {"error": err_output[:200] or "Command failed", "returncode": proc.returncode}

    # Parse tokens per second from output
    # Look for patterns like "Generation: 104.9 t/s" or "tokens per second: 104.9"
    tokens_per_sec = 0
    for line in output.split('\n'):
        match = re.search(r'[Gg]eneration:\s*([\d.]+)\s*t/s', line)
        if match:
            tokens_per_sec = float(match.group(1))
            break
        match = re.search(r'tokens per second:\s*([\d.]+)', line)
        if match:
            tokens_per_sec = float(match.group(1))
            break

    # Count tokens generated
    tokens = len(output.split())

    return {
        "response": output[:500],
        "total_time_sec": total_time,
        "tokens_generated": tokens,
        "tokens_per_sec": tokens_per_sec,
        "gpu_memory_mb": get_gpu_memory(),
        "backend": backend,
        "stderr": err_output[:200]
    }


def run_benchmarks(backend="vulkan"):
    """Run all benchmarks for a specific backend"""
    prompts = load_prompts()

    backend_name = backend.upper()
    print(f"llama.cpp Benchmark ({backend_name})")
    print("=" * 50)

    llama_cli = find_llama_cli()
    if not llama_cli:
        print(f"ERROR: llama-cli not found")
        print("Build with: cd ~/llama.cpp && mkdir build && cd build && cmake .. -DGGML_VULKAN=ON -DGGML_HIP=ON && make -j$(nproc)")
        return None

    results_file = RESULTS_DIR / f"llamaccp_{backend}_results.json"
    results = {}

    for category, category_prompts in prompts.items():
        print(f"\n[{category}]")
        category_results = []
        for i, prompt in enumerate(category_prompts):
            print(f"  Prompt {i+1}...", end=" ", flush=True)
            result = generate_with_llama_cli(prompt, backend=backend)
            if "error" in result:
                print(f"ERROR: {result['error'][:60]}")
                continue
            print(f"{result['tokens_per_sec']:.1f} tokens/sec")
            category_results.append({
                "prompt_preview": prompt[:50] + "...",
                **result
            })
        results[category] = category_results

    # Save results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Summary
    print("\n" + "=" * 50)
    print(f"SUMMARY ({backend_name})")
    print("=" * 50)
    total_tokens = sum(sum(r["tokens_generated"] for r in results[c]) for c in results)
    total_time = sum(sum(r["total_time_sec"] for r in results[c]) for c in results)
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average throughput: {avg_tps:.1f} tokens/sec")

    return results


if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "vulkan"
    run_benchmarks(backend)
