# LoRA and Adapters Guide — Efficient Fine-Tuning in PyTorch

**Purpose:** Implement LoRA (Low-Rank Adaptation) and bottleneck adapters that connect to base model layers. **Interview-style:** parameter efficiency, where to inject, how gradients flow, and how to wire adapters into existing modules.

**Code:** `practice/week4_pytorch/13_lora_adapters/`

---

# Table of Contents

1. [LoRA (Low-Rank Adaptation)](#1-lora-low-rank-adaptation)
2. [Bottleneck Adapters](#2-bottleneck-adapters)
3. [Adapter Blocks and Where to Insert](#3-adapter-blocks-and-where-to-insert)
4. [Parallel Adapter](#4-parallel-adapter)
5. [Learnable Prefix (Prefix Tuning)](#5-learnable-prefix-prefix-tuning)
6. [IA³ (Infused Adapter)](#6-ia³-infused-adapter)
7. [Adapter Stack (multi-task)](#7-adapter-stack-multi-task)
8. [Injecting into Base Models](#8-injecting-into-base-models)
9. [Design Choices & Interview Questions](#9-design-choices--interview-questions)
10. [Common Bugs](#10-common-bugs)

---

## 1. LoRA (Low-Rank Adaptation)

### Idea

- **Full fine-tuning** updates all weights \(W\); expensive and prone to overfitting on small data.
- **LoRA** assumes the update is low-rank: \(\Delta W = B A\) where \(A \in \mathbb{R}^{r \times d_{\text{in}}}\), \(B \in \mathbb{R}^{d_{\text{out}} \times r}\), \(r \ll \min(d_{\text{in}}, d_{\text{out}})\).
- **Forward:** \(y = W x + \alpha \cdot (B A) x = W x + \alpha \cdot B(A x)\). So we keep \(W\) frozen and only train \(A\) and \(B\).
- **Parameters:** \(r \cdot d_{\text{in}} + d_{\text{out}} \cdot r\) instead of \(d_{\text{in}} \cdot d_{\text{out}}\).

### Implementation Notes

- **Freeze** the base `nn.Linear` (e.g. `base.requires_grad_(False)`).
- **Store** `lora_A` (shape `(r, in_features)`) and `lora_B` (shape `(out_features, r)`).
- **Forward:** `base(x) + scale * F.linear(x, B @ A)` or `base(x) + scale * (x @ A.T @ B.T)`.
- **Init:** \(A\) with small random (e.g. Kaiming), \(B\) with zeros so initial \(\Delta W = 0\) and behavior equals the base model at start.
- **Scaling:** Often `scale = alpha / r` for a hyperparameter `alpha` (e.g. `alpha = r` → `scale = 1`).

### Where to Apply LoRA

- **Transformers:** Typically the query/value (and sometimes key) projections in attention, and/or the two linear layers in the FFN. Leave LayerNorm and (optionally) the output projection as-is.
- **MLPs:** Any subset of linear layers; usually the “wide” layers to get most parameter savings.

---

## 2. Bottleneck Adapters

### Idea

- **Adapter:** A small module inserted after a sublayer: down-project (e.g. \(d \to r\)), non-linearity, up-project (\(r \to d\)), then **residual**: output = input + adapter(input).
- **Parameters:** \(d \cdot r + r \cdot d = 2 d r\) (plus biases). \(r\) is the bottleneck dimension (e.g. 4–64).

### Two Uses

1. **With residual (standalone):** `output = x + up(activation(down(x)))`. Used when the adapter is the only addition.
2. **Delta only (inside a block):** `output = up(activation(down(x)))`. Used when you add this on top of a sublayer output: `h = x + sublayer(x); output = h + adapter_delta(h)`.

### Implementation Notes

- **down_proj:** `nn.Linear(d_model, bottleneck_dim)`.
- **up_proj:** `nn.Linear(bottleneck_dim, d_model)`.
- **Init:** Often initialize `up_proj` to zero so the initial adapter output is zero and the residual dominates at start.
- **Activation:** GELU or ReLU.

---

## 3. Adapter Blocks and Where to Insert

### AdapterBlock Pattern

- **Sublayer** (e.g. attention or FFN): `h = x + sublayer(x)`.
- **Adapter delta:** `delta = up(activation(down(h)))`.
- **Output:** `h + delta`.

So the adapter adds a learned delta on top of the sublayer output; the adapter sees the “hidden state” after the first residual.

### Where to Insert Adapters (Houlsby-style)

- **After attention:** \(h = x + \text{Attn}(x)\); then \(h = h + \text{Adapter}(h)\).
- **After FFN:** \(h = h + \text{FFN}(h)\); then \(h = h + \text{Adapter}(h)\).

So two adapters per transformer block (post-attention, post-FFN) is common. Our code uses a single AdapterBlock that wraps one sublayer (e.g. the whole block) and adds one adapter; you can stack or duplicate for multiple insertion points.

---

## 4. Parallel Adapter

### Idea

- **Serial (Houlsby):** adapter sees the **output** of the sublayer: \(h = x + \text{sublayer}(x)\); then \(h = h + \text{adapter}(h)\).
- **Parallel:** adapter sees the **input** \(x\); output = \(\text{sublayer}(x) + \text{adapter}(x)\). Both branches see the same input.

### When to Use

- Parallel adds capacity without changing the sublayer’s gradient path; serial lets the adapter modify the post-sublayer representation. Parallel is simpler and often used in “adapter in parallel” variants.

### Implementation

- **ParallelAdapterBlock(sublayer, adapter):** `forward(x) = sublayer(x) + adapter.forward_delta(x)`.

---

## 5. Learnable Prefix (Prefix Tuning)

### Idea

- **Prefix tuning:** Prepend a fixed number of **learnable** vectors (prefix) to the sequence. Downstream attention sees (prefix tokens + content tokens).
- Input \((B, L, d)\) → output \((B, \text{prefix\_len} + L, d)\). The prefix acts as “virtual” context (e.g. for K, V in attention).

### Implementation

- **LearnablePrefix(prefix_len, d_model):** Parameter `(prefix_len, d_model)`. Forward: `cat(prefix.expand(B,-1,-1), x, dim=1)`.

### Where to Use

- Before the first attention layer (or before each block) so that key/value include the prefix. Often used with frozen backbone + trainable prefix only.

---

## 6. IA³ (Infused Adapter)

### Idea

- **IA³:** Learnable **elementwise scaling** on the output of a linear: \(\text{output} = W x \odot \ell\), where \(\ell \in \mathbb{R}^{d_{\text{out}}}\) is a trainable vector. Base \(W\) is frozen.
- Parameters: only \(d_{\text{out}}\) (one scalar per output dimension). Used in T-Few, very parameter-efficient.

### Implementation

- **IA3Linear(base):** Store base (frozen), scale `nn.Parameter(out_features,)` init ones. Forward: `base(x) * scale` (scale broadcasts).

### inject_ia3

- Replace named `nn.Linear` with `IA3Linear`; return (model, list of scale parameters) for the optimizer.

---

## 7. Adapter Stack (multi-task)

### Idea

- **AdapterStack:** Hold **multiple** bottleneck adapters (e.g. one per task or per language). At forward time, choose one by index: `forward(x, adapter_id=k)`.
- Same base, different adapter per task; switch by passing `adapter_id`.

### Implementation

- **AdapterStack(d_model, bottleneck_dim, num_adapters):** `nn.ModuleList` of `BottleneckAdapter`. Forward: `x + adapters[adapter_id].forward_delta(x)`.

---

## 8. Injecting into Base Models

### Replacing Linears with LoRALinear

- **By name:** `inject_lora(model, linear_names, rank, scale)` — iterate over `linear_names`, `getattr(model, name)`; if it’s `nn.Linear`, replace with `LoRALinear(layer, rank, scale)` and `setattr(model, name, lora_layer)`.
- **Recursive:** `inject_lora_recursive(model, rank, scale)` — walk `model.named_modules()`; for each `nn.Linear` get parent and child name (split path); `setattr(parent, child_name, LoRALinear(linear, rank, scale))`. Skip root (empty name). Return (model, lora_params).
- **Return trainable params:** Collect all `lora_A` and `lora_B` parameters and return them for the optimizer so you only pass `lora_params` to the optimizer (base remains frozen).
- **IA³ by name:** `inject_ia3(model, linear_names)` — replace named linears with `IA3Linear`; return (model, list of scale parameters).

### Connecting to “Real” Base Models

- **Shape compatibility:** LoRALinear must have the same `in_features` and `out_features` as the original Linear; then input/output shapes are unchanged.
- **Device:** Move the base model to device first; when you replace with LoRALinear, the new module gets the same device as the base layer’s parameters.
- **Checkpointing:** When saving, save base weights (or load from pretrained) and separately save LoRA state dict (only `lora_A`, `lora_B`); at load time, load base then inject LoRA and load LoRA state.

---

## 9. Design Choices & Interview Questions

| Topic | What to say |
|--------|-------------|
| **LoRA rank** | Larger \(r\) = more capacity, more parameters. Typical 4–64 for LLMs. Trade-off: expressiveness vs overfitting and memory. |
| **LoRA vs full fine-tune** | LoRA: few trainable params, less overfitting, easy to swap adapters. Full: more capacity, needs more data and compute. |
| **Adapter bottleneck size** | Like LoRA rank: 4–64 common. Smaller = more parameter-efficient, less capacity. |
| **Where to put LoRA** | Attention Q/V (and sometimes K) and FFN linears. Skip LayerNorm and embeddings if you want minimal change. |
| **Scale (alpha/r)** | `scale = alpha / r` with `alpha \approx r` often used; larger alpha amplifies the LoRA update. |
| **Parallel adapter** | Same bottleneck (down–up) but output = sublayer(x) + adapter(x); adapter sees input, not sublayer output. |
| **Prefix tuning** | Learnable prefix prepended to sequence; attention sees prefix + content; very few params (prefix_len × d_model). |
| **IA³** | Elementwise scale on linear output; one scalar per output dim; used in T-Few. |
| **Serial vs parallel** | Serial: adapter on sublayer output. Parallel: adapter on input; output = sublayer(x) + adapter(x). |
| **Adapter stack** | Multiple adapters (e.g. per task); forward(x, adapter_id=k); same base, switch adapter by index. |

---

## 10. Common Bugs

| Bug | Symptom / cause | Fix |
|-----|------------------|-----|
| **Base layer not frozen** | Base weights change; LoRA is not the only trainable part. | Call `base.requires_grad_(False)` after wrapping. |
| **Wrong LoRA init** | Training unstable or no effect. | Init \(B\) to zeros so \(\Delta W = 0\) at start; \(A\) small random (Kaiming). |
| **Shape mismatch after inject** | Forward fails. | LoRALinear must use same in/out features as the original Linear. |
| **Optimizer has base params** | Base gets updated. | Optimizer should only receive `lora_params` (and adapter params), not `model.parameters()`. |
| **Adapter residual twice** | Double residual (2h + delta). | In AdapterBlock use “delta only” (up(down(h))) and add once: h + delta. |
| **Device mismatch** | Tensors on CPU/GPU mixed. | Move model (and injected LoRA) to device before forward; move inputs to same device. |

---

## Code Challenges Roadmap (13_lora_adapters)

**Core (LoRA + bottleneck)**  
1. **LoRALinear** — Wrap `nn.Linear` with low-rank A, B; freeze base; forward = base(x) + scale * (x @ A.T @ B.T).  
2. **BottleneckAdapter** — down_proj → activation → up_proj; optional residual; support `forward_delta` for blocks.  
3. **AdapterBlock** — sublayer + adapter; forward = x + sublayer(x) + adapter_delta(x + sublayer(x)).  
4. **inject_lora** — Replace named `nn.Linear` with LoRALinear; return (model, lora_params).  
5. **build_lora_adapter_model** — Base block + LoRA on specified linears + optional AdapterBlock; freeze base; return (model, trainable_params).

**More architectures & injection**  
6. **ParallelAdapterBlock** — forward(x) = sublayer(x) + adapter_delta(x).  
7. **LearnablePrefix** — prepend learnable (prefix_len, d_model) to (B, L, d) → (B, prefix_len+L, d).  
8. **IA3Linear** — wrap Linear with elementwise scale (out_features,); forward = base(x) * scale; base frozen.  
9. **inject_ia3** — Replace named `nn.Linear` with IA3Linear; return (model, ia3_params).  
10. **inject_lora_recursive** — Replace **all** nn.Linear in model (via named_modules) with LoRALinear; return (model, lora_params).  
11. **AdapterStack** — multiple BottleneckAdapters; forward(x, adapter_id=k) uses adapter k.

**Interview:** LoRALinear and inject_lora by name; serial vs parallel adapter; prefix tuning; IA³ (elementwise scale); recursive injection; adapter stack for multi-task.
