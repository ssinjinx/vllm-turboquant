#!/usr/bin/env python3
"""
Benchmark script for llama.cpp server (llama-server)
Supports both Vulkan and ROCm backends via environment variables
Measures: throughput, latency, memory usage, token generation speed
"""

import requests
import time
import json
import subprocess
import re
import os
import signal
from pathlib import Path

LLAMA_SERVER_PORT = 8080
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


def find_llama_server():
    """Find llama-server binary"""
    paths = [
        str(Path.home() / "llama.cpp/build/bin/llama-server"),
        "./llama.cpp/build/bin/llama-server",
        "/usr/local/bin/llama-server",
    ]
    for p in paths:
        if Path(p).exists():
            return p
    return None


def start_llama_server(backend="vulkan", port=8080):
    """Start llama-server in background"""
    llama_server = find_llama_server()
    if not llama_server:
        return None, "llama-server not found"

    # Set GPU backend env
    env = os.environ.copy()
    if backend == "vulkan":
        env["GGML_VULKAN"] = "1"
    elif backend == "rocm":
        env["GGML_HIPBLAS"] = "1"
        env["HIP_VISIBLE_DEVICES"] = "0"

    cmd = [
        llama_server,
        "-m", MODEL_PATH,
        "-ngl", "99",
        "--port", str(port),
        "--host", "127.0.0.1",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        # Wait for server to start
        time.sleep(8)
        if proc.poll() is not None:
            return None, "Server failed to start"
        return proc, None
    except Exception as e:
        return None, str(e)


def stop_llama_server(proc):
    """Stop llama-server"""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def generate_with_server(prompt, base_url, max_tokens=128):
    """Generate response using llama-server API"""
    url = f"{base_url}/v1/completions"
    payload = {
        "prompt": prompt[:500],  # Truncate
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.7
    }

    headers = {"Content-Type": "application/json"}

    start_time = time.perf_counter()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
    except Exception as e:
        return {"error": str(e)}
    end_time = time.perf_counter()

    total_time = end_time - start_time

    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}

    result = response.json()

    if "error" in result:
        return {"error": result["error"]}

    choices = result.get("choices", [])
    if not choices:
        return {"error": "No choices in response"}

    response_text = choices[0].get("text", "")
    usage = result.get("usage", {})
    tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)

    # Get timing info from server
    timings = result.get("timings", {})
    prompt_per_sec = timings.get("prompt_per_second", 0)
    predicted_per_sec = timings.get("predicted_per_second", 0)

    return {
        "response": response_text[:500],
        "total_time_sec": total_time,
        "tokens_generated": tokens,
        "prompt_tokens": prompt_tokens,
        "tokens_per_sec": predicted_per_sec or (tokens / total_time if total_time > 0 else 0),
        "prompt_per_sec": prompt_per_sec,
        "gpu_memory_mb": get_gpu_memory(),
        "server_timings": timings
    }


def run_benchmarks(backend="vulkan"):
    """Run all benchmarks for a specific backend"""
    prompts = load_prompts()

    backend_name = backend.upper()
    print(f"llama.cpp Server Benchmark ({backend_name})")
    print("=" * 50)

    llama_server = find_llama_server()
    if not llama_server:
        print(f"ERROR: llama-server not found")
        print("Build with: cd ~/llama.cpp && mkdir build && cd build && cmake .. -DGGML_VULKAN=ON -DGGML_HIP=ON -DLLAMA_BUILD_SERVER=ON && make -j$(nproc)")
        return None

    # Start server
    port = 8080 if backend == "vulkan" else 8081
    base_url = f"http://127.0.0.1:{port}"

    print(f"Starting llama-server ({backend_name}) on port {port}...")
    proc, err = start_llama_server(backend, port)
    if err:
        print(f"ERROR: {err}")
        return None

    try:
        # Check if server is ready
        try:
            requests.get(f"{base_url}/v1/models", timeout=5)
        except Exception:
            print("ERROR: Server not responding")
            stop_llama_server(proc)
            return None

        results_file = RESULTS_DIR / f"llamaccp_{backend}_results.json"
        results = {}

        for category, category_prompts in prompts.items():
            print(f"\n[{category}]")
            category_results = []
            for i, prompt in enumerate(category_prompts):
                print(f"  Prompt {i+1}...", end=" ", flush=True)
                result = generate_with_server(prompt, base_url)
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

    finally:
        stop_llama_server(proc)


if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "vulkan"
    run_benchmarks(backend)
