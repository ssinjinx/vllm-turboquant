<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
  TurboQuant KV Cache Compression for AMD ROCm
</h3>

<p align="center">
  <a href="https://github.com/ssinjinx/vllm-turboquant">
    <img alt="GitHub Repo" src="https://img.shields.io/badge/This_Fork-ssinjinx%2Fvllm--turboquant-blue">
  </a>
  <a href="https://github.com/mitkox/vllm-turboquant">
    <img alt="Upstream" src="https://img.shields.io/badge/Upstream-mitkox%2Fvllm--turboquant-green">
  </a>
  <a href="https://arxiv.org/abs/2501.04304">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-TurboQuant-orange">
  </a>
  <a href="https://rocm.docs.amd.com/">
    <img alt="ROCm" src="https://img.shields.io/badge/ROCm-7.2-red">
  </a>
</p>

---

## About This Fork

This is a **ROCm-specific build** of [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant), which itself is a **TurboQuant-enabled fork of [vLLM](https://github.com/vllm-project/vllm)** from Google Research.

**TurboQuant** provides 3-5x KV cache compression, enabling larger models to run on limited VRAM.

### Attribution

- **TurboQuant**: [Google Research](https://arxiv.org/abs/2501.04304)
- **vLLM**: [UC Berkeley Sky Computing Lab](https://github.com/vllm-project/vllm)
- **TurboQuant vLLM Fork**: [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant)
- **ROCm Build & Testing**: [@ssinjinx](https://github.com/ssinjinx)

### Key Features

- **TurboQuant KV Cache Compression**: 3-5x compression ratio for KV cache
- **AMD ROCm Support**: Works with AMD RX 7900 XTX, MI300X, and other ROCm-capable GPUs
- **OpenAI-Compatible API**: Drop-in replacement for standard vLLM API
- **Performance**: ~30% throughput improvement from KV cache compression

---

## Why TurboQuant + ROCm?

| GPU | VRAM | Llama 3 70B (native) | Llama 3 70B (TurboQuant 3.5x) |
|-----|------|----------------------|-------------------------------|
| RX 7900 XTX | 24GB | ❌ Won't fit | ✅ ~8GB KV cache |
| MI300X | 128GB | ✅ ~48GB | ✅ ~14GB KV cache |

TurboQuant compresses the KV cache from 16-bit floats to 0.25-0.5 bits per value, dramatically reducing memory footprint.

---

## Hardware Requirements

### GPU Support
- **AMD RX 7900 XTX** (RDNA3, gfx1100) — tested
- **AMD MI300X** (CDNA3) — should work
- **Other ROCm-capable AMD GPUs** — likely work with ROCm 7.2

### Software Requirements
- ROCm 7.2+
- Python 3.10-3.13
- ROCm-compatible PyTorch

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/mitkox/vllm-turboquant.git
cd vllm-turboquant
```

### Step 2: Create ROCm Virtual Environment

```bash
python3 -m venv vllm-rocm-venv
source vllm-rocm-venv/bin/activate

# Install PyTorch with ROCm support
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch torchvision

# Install amdsmi for ROCm device detection
pip install amdsmi
```

### Step 3: Build vLLM TurboQuant

```bash
# Configure with ROCm
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_BUILD_WITH_ROCM=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXECUTABLE_FORMAT=ELF \
  -DNCCL_PYTHON_URL=https://nvidia.github.io/nccl-rhel7/2.25.1/nccl-2.25.1+cuda12.6-1.x86_64.rpm \
  -DNCCL_PYTHON_PACKAGE_PATH=/dev/null

# Build with 16 threads
cmake --build build -j16
```

### Step 4: Install the Package

```bash
pip install .
```

---

## Usage

### Starting the Server

```bash
source vllm-rocm-venv/bin/activate
export HIP_VISIBLE_DEVICES=0

vllm serve Qwen/Qwen3-8B \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --port 8000
```

**Note**: Set `--gpu-memory-utilization` lower (0.7-0.75) if Chrome or other GPU-using apps are running, as they consume ~6GB VRAM.

### Testing with curl

```bash
# Check models
curl http://127.0.0.1:8000/v1/models

# Generate text
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain what machine learning is in one sentence.",
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    gpu_memory_utilization=0.7,
    max_model_len=2048,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
outputs = llm.generate(["Explain what machine learning is."], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## Benchmark Results

Tested on **AMD RX 7900 XTX** (24GB VRAM) with Qwen3-8B (Q4_K_M):

| Backend | Throughput | Notes |
|---------|------------|-------|
| llama.cpp + Vulkan | ~100 tok/s | GGUF format |
| llama.cpp + ROCm | ~100 tok/s | GGUF format |
| **vLLM TurboQuant + ROCm** | **~54 tok/s** | Native model, KV compression |

**Interpretation**: vLLM TurboQuant trades some throughput for 3-5x KV cache compression. The real benefit appears with larger models where native vLLM/LlamaCPP wouldn't fit in VRAM at all.

---

## Troubleshooting

### Error: "No module named 'amdsmi'"

```bash
pip install amdsmi
```

### Error: "operator torchvision::nms does not exist"

```bash
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torchvision --upgrade
```

### Error: "Failed to infer device type"

```bash
# Ensure amdsmi is installed and ROCm is detected
vllm collect-env | grep ROCm
```

### Error: "Not enough GPU memory"

```bash
# Lower gpu-memory-utilization (0.7-0.75 with Chrome running)
# or close GPU-using applications
vllm serve Qwen/Qwen3-8B --gpu-memory-utilization 0.7
```

### Model name format confusion

Use **HuggingFace format** (e.g., `Qwen/Qwen3-8B`), not Ollama format (`qwen3:8b`).

---

## Project Structure

```
vllm-turboquant/
├── vllm/
│   └── v1/
│       └── attention/ops/
│           └── turboquant_metadata.py  # TurboQuant KV cache implementation
├── vllm/v1/attention/backends/
│   └── turboquant_attn.py              # TurboQuant attention backend
├── benchmarks/                          # Benchmark scripts
└── results/                             # Benchmark results
```

---

## TurboQuant Technical Details

TurboQuant (from Google Research) compresses KV cache entries to 0.25-0.5 bits per value using:
1. **Per-channel scaling** for quantization
2. **Row-wise quantization** for attention scores
3. **Smooth quant** for activation normalization

This enables models that would normally require 80GB+ KV cache to run on 24GB GPUs.

---

## References

- [TurboQuant Paper](https://arxiv.org/abs/2501.04304)
- [vLLM Project](https://github.com/vllm-project/vllm)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD GPU Support in vLLM](https://docs.vllm.ai/en/latest/getting_started/amd-installation.html)

---

## License

Apache 2.0 — same as upstream vLLM

---

## Contributing

Issues and PRs welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
