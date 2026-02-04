# PyTorch Mastery — Interview Prep

Structured PyTorch practice: **data → models → training → debugging → pipelines → advanced topics**. Incremental and cumulative; each level builds on the previous.

**Start here:** [PEDAGOGY.md](PEDAGOGY.md) — learning path, why the order matters, and how the pieces connect.

---

## Learning path (summary)

| Level | Folder | You learn |
|-------|--------|-----------|
| **1** | `01_data_processing` | Device, tensor shapes, `cat`/`stack`, conditioning a feature map, Dataset, DataLoader |
| **1** | `02_models` | ConcatBlock, ConditionalMLP, Encoder/Decoder/FullModel (concat, conditioning, plugging modules) |
| **1** | `03_training` | One-epoch loop: zero_grad → forward → loss → backward → step; move data to device |
| **1** | `04_bug_finding` | 10 common bugs: labels dtype, `cat` dim, device, eval(), shape, in-place, target shape, BN eval, scheduler |
| **2** | `05_level2_intermediate` | Train+val loop, backbone+head, checkpoint save/load |
| **3** | `06_level3_advanced` | Multi-input, gradient clipping, gradient accumulation, mixed precision (FP16), gradient checkpointing, PathLabelDataset |
| **4** | `07_level4_industry` | Full pipeline, early stopping, LR scheduler |
| **Cumulative** | `08_cumulative` | End-to-end pipeline, two-stage training (freeze backbone, finetune head) |
| **Expert** | `09_expert` | Custom autograd.Function, distributed-ready structure |
| **Advanced** | `10_diffusion_flow` | Noise-prediction model, DDPM step, diffusion bug |
| **Advanced** | `11_transformers` | Scaled dot-product attention, multi-head, encoder block |
| **Advanced** | `12_popular_architectures` | ResidualBlock, SimpleCNN, VGGBlock, make_layer, Bottleneck, SmallResNet, InceptionModule |
| **Advanced** | `13_lora_adapters` | LoRALinear, BottleneckAdapter, AdapterBlock, inject_lora, build_lora_adapter_model (connect to base layers) |
| **Interview** | `14_computation_exercises` | Conv 1D/2D/3D (output shape, params, FLOPs), Transformer/ViT shapes & params, complexity, tricky Q&A |

Study guides: **PyTorch_Mastery_Guide** (core, **quizzes** on mixed precision, gradient accumulation, checkpointing, etc.) · **Advanced_Architectures_Guide** (diffusion, transformers, architectures) · **Popular_Architectures_Explained** (the “why” only, no code — use with `12_popular_architectures`) · **LoRA_and_Adapters_Guide** (LoRA, bottleneck adapters, injection, use with `13_lora_adapters`) · **Computation_Exercises_Guide** (conv/transformer/ViT dimensions, params, complexity — use with `14_computation_exercises`).

---

## How to use

1. **Challenges** — Implement what the docstring asks; run the file to check asserts. Don’t peek at solutions until you’ve tried.
2. **Solutions** — Reference implementations in `solution_*.py`; compare after you attempt the challenge.
3. **Bug-finding** — Run `buggy_code_*.py`, see the error, fix it, then open `solution_bug_*.py` to see the intended fix and reason.

**Requirements:** `pip install torch torchvision` (or `pip install -r requirements.txt`). Code uses `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and moves model and batches to `device` where needed.

---

## Running the test suite

A full pytest suite validates all **solutions** (shapes, dtypes, no NaN/Inf, gradient flow where relevant). Bad implementations in challenge files will fail their own `if __name__ == "__main__"` asserts when you run them.

```bash
cd practice/week4_pytorch
pip install -r requirements.txt   # torch, torchvision, pytest
pytest tests/ -v
```

- **101 tests** across 01–13 plus **test_edge_cases.py** (edge-case and cross-cutting tests).
- **test_edge_cases.py** adds: batch size 1, numerical sanity (finite outputs), device consistency, gradient flow (no NaN grads), parametrized batch sizes, expected errors (mismatched batch raises), and checks that challenge files raise `NotImplementedError` before implementation.
- To run a single file: `pytest tests/test_01_data_processing.py -v` or `pytest tests/test_edge_cases.py -v`
- Challenge files are not run by pytest (except edge-case tests that assert they raise `NotImplementedError`); run each `challenge_*.py` or `buggy_code_*.py` yourself and fix until asserts pass.
