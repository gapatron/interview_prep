# 13 — LoRA and Adapters

Implement **LoRA**, **bottleneck adapters**, **parallel adapter**, **learnable prefix**, **IA³**, **adapter stack**, and **injection** (by name + recursive).

**Study guide:** `practice/study_guides/LoRA_and_Adapters_Guide.md`

## Challenges (in order)

**Core (LoRA + bottleneck)**  
1. **LoRALinear** — Wrap `nn.Linear` with low-rank A, B; freeze base; forward = base(x) + scale * (x @ A.T @ B.T).  
2. **BottleneckAdapter** — down_proj → activation → up_proj; residual or delta-only (for AdapterBlock).  
3. **AdapterBlock** — sublayer + adapter; forward = x + sublayer(x) + adapter_delta(x + sublayer(x)).  
4. **inject_lora** — Replace named `nn.Linear` with LoRALinear; return (model, lora_params).  
5. **build_lora_adapter_model** — Base block + LoRA on specified linears + optional AdapterBlock; freeze base; return (model, trainable_params).

**More architectures & injection**  
6. **ParallelAdapterBlock** — forward(x) = sublayer(x) + adapter_delta(x).  
7. **LearnablePrefix** — prepend learnable (prefix_len, d_model) to (B, L, d) → (B, prefix_len+L, d).  
8. **IA3Linear** — wrap Linear with elementwise scale (out_features,); base frozen.  
9. **inject_ia3** — Replace named `nn.Linear` with IA3Linear; return (model, ia3_params).  
10. **inject_lora_recursive** — Replace **all** nn.Linear in model with LoRALinear; return (model, lora_params).  
11. **AdapterStack** — multiple BottleneckAdapters; forward(x, adapter_id=k).

## Base model

`base_model.py` provides **BaseMLP** (named linears: fc1, fc2, fc3) and **BaseTransformerBlock** (named linears: proj_in, proj_out, ff1, ff2) for injection. No dependency on 11_transformers.

## Run tests

From `practice/week4_pytorch`: `pytest tests/test_13_lora_adapters.py -v`
