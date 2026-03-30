# ROCm Benchmark Results

Benchmark scripts and results for vLLM TurboQuant on AMD ROCm.

## Test System

- **GPU**: AMD RX 7900 XTX (24GB VRAM, RDNA3, gfx1100)
- **OS**: Linux 6.17.0
- **ROCm**: 7.2
- **Model**: Qwen3-8B (Q4_K_M via llama.cpp, native via vLLM)

## Benchmark Results

### llama.cpp (GGUF format)

| Backend | Prompt Type | Tokens/sec |
|---------|-------------|------------|
| Vulkan | short | 100.7 |
| Vulkan | medium | 100.6 |
| Vulkan | long | 100.0 |
| Vulkan | reasoning | 99.8 |
| Vulkan | code | 98.9 |
| Vulkan | math | 99.6 |
| **ROCm** | short | 100.5 |
| **ROCm** | medium | 100.3 |
| **ROCm** | long | 99.9 |
| **ROCm** | reasoning | 99.6 |
| **ROCm** | code | 99.3 |
| **ROCm** | math | 99.2 |

### vLLM TurboQuant (ROCm)

| Throughput | Notes |
|------------|-------|
| ~54 tok/s | Native model, KV cache compression enabled |

## Analysis

- llama.cpp (Vulkan/ROCm): ~100 tok/s
- vLLM TurboQuant: ~54 tok/s

vLLM TurboQuant appears slower for small models, but its advantage is **memory efficiency**. With 3-5x KV cache compression, it can run models 3-5x larger that wouldn't fit in VRAM otherwise.

## Running Benchmarks

```bash
# Benchmark vLLM
cd benchmarks
python3 vllm_bench.py

# Benchmark llama.cpp (Vulkan)
python3 llama_cpp_bench.py vulkan

# Benchmark llama.cpp (ROCm)
python3 llama_cpp_bench.py rocm
```

## Prompt Categories

- **short**: 10-15 token prompts
- **medium**: 20-30 token prompts
- **long**: 40+ token prompts
- **reasoning**: Step-by-step reasoning tasks
- **code**: Code generation tasks
- **math**: Mathematical problem solving
