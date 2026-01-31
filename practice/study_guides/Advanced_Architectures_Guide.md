# Advanced Architectures Guide — Diffusion, Flow Matching, Transformers, Popular Nets

**Purpose:** Chapters on diffusion models & flow matching (design, bugs, alternatives), building the Transformer from scratch, and implementing popular architectures (forward → layers, stacking). **Interview-style: solving problems, finding bugs, considering alternatives.**

**Estimated study time:** 8–14 hours (with code challenges)

---

# Table of Contents

1. [Chapter A: Diffusion Models & Flow Matching](#chapter-a-diffusion-models--flow-matching)
2. [Chapter B: Transformers](#chapter-b-transformers)
3. [Chapter C: Building Popular Architectures](#chapter-c-building-popular-architectures)
4. [Code Challenges Roadmap](#code-challenges-roadmap)

---

# Chapter A: Diffusion Models & Flow Matching

## A.1 Core Ideas

- **Diffusion (DDPM-style):** Data \(x_0\) is gradually noised to \(x_T\) (forward process); a model learns to **predict noise** (or \(x_0\)) so we can reverse the process and sample \(x_0\) from \(x_T\).
- **Forward process:** \(x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon\), where \(\varepsilon \sim \mathcal{N}(0,I)\) and \(\bar\alpha_t\) comes from a **noise schedule** (e.g. linear betas).
- **Training:** Sample \(t\), sample noise \(\varepsilon\), form \(x_t\), train the model to predict \(\varepsilon\) (or \(x_0\)). Loss = MSE between predicted and true noise (or \(x_0\)).
- **Sampling:** Start from \(x_T \sim \mathcal{N}(0,I)\), then iteratively denoise (e.g. DDPM update or DDIM) to get \(x_0\).

**Flow matching** (alternative): Learn a vector field that transports samples from a simple distribution to data; often simpler training (single-step objectives) and faster sampling. **Design alternative to diffusion:** when to use flow matching vs diffusion (speed, stability, conditioning).

## A.2 Design Choices & Interview Questions

| Topic | What to say |
|--------|--------------|
| **Noise schedule** | Linear vs cosine; cosine often better for images. \(\beta_t\) small early → less distortion; larger later → more noise. |
| **Predict \(\varepsilon\) vs \(x_0\)** | Predicting noise \(\varepsilon\) is common (Ho et al.); equivalent to predicting \(x_0\) with a reparameterization. Implementation: model gets \(x_t, t\) and outputs \(\hat\varepsilon\); loss = \(\|\hat\varepsilon - \varepsilon\|^2\). |
| **Timestep conditioning** | \(t\) must be fed to the model (embedding or sinusoidal). Shape: broadcast to match spatial dims or add as a global embedding. |
| **Conditioning (class, text)** | Concatenate condition with \(t\) embedding; or cross-attention in U-Net; or adapter layers. |
| **Flow matching vs diffusion** | Flow: learn ODE flow; often straight paths, fewer steps. Diffusion: learn score / noise; many steps unless distilled. Choice: quality vs speed, ease of conditioning. |

## A.3 Common Bugs in Diffusion Code

| Bug | Symptom / cause | Fix |
|-----|------------------|-----|
| **Wrong shape for \(x_t\)** | \(x_t\) must be same shape as \(x_0\). If you flatten too early, spatial structure is lost. | Keep (N, C, H, W) through noise and model. |
| **Timestep \(t\) not used** | Model ignores \(t\) → same prediction for all \(t\). | Pass \(t\) into the model (embed and add or concat). |
| **\(t\) as float in loss** | Model expects integer indices for embedding. | Use integer \(t \in \{0,\ldots,T-1\}\) for embedding lookup (or sinusoidal with float). |
| **Device mismatch** | \(x_t\) on CPU, model on GPU. | Move \(x_t, t, \varepsilon\) to same device as model. |
| **Wrong loss reduction** | Per-pixel loss not reduced → gradient scale wrong. | Use `reduction='mean'` (default) so loss is scalar. |
| **Sampling: wrong step direction** | Using \(+\) instead of \(-\) when subtracting noise. | Denoising: \(x_{t-1}\) = mean - noise term; check sign from derivation. |
| **Alpha / sigma shape** | \(\bar\alpha_t\) must broadcast to (N, C, H, W) when computing \(x_t\). | Reshape: `alpha_bar = alpha_bar.view(-1, 1, 1, 1)` or equivalent. |

## A.4 Alternatives & Extensions

- **DDIM:** Deterministic sampler; fewer steps, same model.
- **Score-based (VE/VP SDE):** Different parameterization; model predicts score \(\nabla \log p(x_t)\).
- **Flow matching / Rectified Flow:** \(dx/dt = v_t(x)\); train to match velocity; sample with ODE solver.
- **Latent diffusion:** Diffusion in VAE latent space (e.g. Stable Diffusion); faster and less memory.
- **Classifier-free guidance:** Train with random dropout of condition; at test time use \( \hat\varepsilon = \hat\varepsilon_u + w\,(\hat\varepsilon_c - \hat\varepsilon_u) \).

## A.5 Minimal Training Step (Pseudocode)

```python
# Training one step (DDPM-style)
def train_step(model, x0, t, eps, alpha_bar_t):
    # x0: (N, C, H, W), t: (N,) long, eps: (N, C, H, W), alpha_bar_t: (N, 1, 1, 1)
    xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    pred_eps = model(xt, t)
    loss = F.mse_loss(pred_eps, eps)
    return loss
```

**Interview:** Be able to implement a noise-prediction forward pass (model inputs: \(x_t\), \(t\); output: \(\hat\varepsilon\)), and the loss. Know how to build \(x_t\) from \(x_0\) and \(\varepsilon\) with the right shapes.

---

# Chapter B: Transformers

## B.1 Building Blocks

- **Scaled dot-product attention:**  
  \(\text{Attention}(Q,K,V) = \text{softmax}\big(\frac{QK^\top}{\sqrt{d_k}}\big)V\).  
  \(Q,K,V\) from linear projections of the same input (self-attention). Shapes: (batch, heads, seq, head_dim).

- **Multi-head attention:**  
  Several attention heads in parallel; concat outputs then project:  
  \(\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\ldots)\;W^O\).  
  Implement: Linear for Q, K, V (or one Linear and split), then reshape to (B, H, L, d_k), attention per head, concat, final Linear.

- **Position encoding:**  
  Add positional information so order matters. **Sinusoidal:** \(PE_{pos,2i} = \sin(pos/10000^{2i/d})\), \(PE_{pos,2i+1} = \cos(...)\). **Learned:** nn.Embedding(max_len, d_model).

- **Encoder block:**  
  Self-attention → Add & Norm → FFN (two linear with GELU/ReLU in between) → Add & Norm.  
  FFN: \(x \to \text{Linear}(d \to 4d) \to \text{GELU} \to \text{Linear}(4d \to d)\).

- **Decoder block (autoregressive):**  
  Masked self-attention (causal mask) → Add & Norm → Cross-attention (Q from decoder, K,V from encoder) → Add & Norm → FFN → Add & Norm.

## B.2 Forward Flow (Encoder-Only, e.g. BERT-style)

1. Input tokens → embedding + positional encoding.
2. For each encoder block:  
   - \(x = x + \text{MultiHeadAttention}(x)\) then LayerNorm.  
   - \(x = x + \text{FFN}(x)\) then LayerNorm.  
3. Output: (batch, seq_len, d_model).

Shapes to keep straight:  
- Embedding out: (B, L, d_model).  
- Q,K,V after projection: (B, L, d_model); reshape to (B, num_heads, L, head_dim).  
- Attention scores: (B, H, L, L); mask if causal.  
- After attention: (B, L, d_model).

## B.3 Common Bugs in Transformer Code

| Bug | Cause | Fix |
|-----|--------|-----|
| **Wrong mask for causal attention** | Upper triangle should be -inf before softmax. | `mask.tril() == 1` for valid; set invalid to -1e9. |
| **Dimension mismatch in attention** | Q,K,V head_dim or num_heads don’t match. | Ensure d_model % num_heads == 0; head_dim = d_model // num_heads. |
| **Forgetting LayerNorm** | Pre-LN vs Post-LN: most modern use Pre-LN (norm before sublayer). | Apply norm before attention/FFN, then add residual. |
| **Position encoding not added** | Model is permutation-invariant. | Add PE to embeddings before first block. |
| **Float vs long for positions** | Embedding expects long indices. | Use torch.arange(seq_len, device=x.device).long(). |
| **Batch dimension in mask** | Mask shape (L, L) vs (B, 1, L, L) for broadcasting. | Use (1, 1, L, L) or (B, 1, L, L) so it broadcasts. |

## B.4 Design Choices

- **Pre-LN vs Post-LN:** Pre-LN (norm first) is more stable for deep transformers.
- **Absolute vs relative position:** Relative (e.g. T5, relative bias in attention) can generalize better to longer sequences.
- **Encoder-only vs encoder–decoder:** Encoder-only for classification/embedding; encoder–decoder for generation with cross-attention.

## B.5 Minimal Code Sketch (Encoder Block)

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        x = x + self._sa_block(self.norm1(x), src_mask)
        x = x + self.ff(self.norm2(x))
        return x

    def _sa_block(self, x, mask):
        return self.self_attn(x, x, x, attn_mask=mask)[0]
```

**Interview:** Implement scaled dot-product attention from scratch (Q, K, V inputs; return output and optional attention weights). Then multi-head by splitting and concatenating.

---

# Chapter C: Building Popular Architectures

**Important:** The challenge files in `12_popular_architectures/` do **not** give you the code — they describe what to implement and point to **Popular_Architectures_Explained.md** for the *why* (residual connections, BN order, shortcuts, shapes, common bugs). Use the Explained guide to derive your implementation; solutions are in separate `solution_*.py` files for reference only.

## C.1 Philosophy: Forward First, Then Layers

1. **Write the forward pass in words:** “Input goes through conv blocks, then we add skip, then pool, then linear.”
2. **Define submodules:** ResidualBlock, ConvBlock, etc., with clear input/output shapes.
3. **Stack in forward:** Call blocks in order; ensure shapes match (especially for skip connections).
4. **Register correctly:** All learnable blocks as `nn.Module` children (either `nn.Sequential` or explicit `self.block = ...`).

## C.2 ResNet

- **Idea:** Learn residual \(F(x) = H(x) - x\); block computes \(y = F(x) + x\) so gradients flow through the skip.
- **Basic block (ResNet-18/34):** Two 3×3 convs, BatchNorm, ReLU; add input to output. Shape: (N, C, H, W) → same if stride=1; if stride=2, downsample the residual (e.g. 1×1 conv with stride 2) so shapes match.
- **Bottleneck (ResNet-50+):** 1×1 (reduce C), 3×3, 1×1 (expand); fewer params and same receptive field.
- **Forward:** stem (conv + BN + ReLU + pool) → layer1, layer2, layer3, layer4 (each a few blocks) → global pool → linear.

**Common bugs:**  
- Skip and main path shapes differ after stride-2 → downsample shortcut (1×1 conv or pad).  
- Forgetting to add residual: `return self.conv2(relu(self.conv1(x))) + x` (and handle shortcut when dimensions change).

## C.3 VGG-Style

- **Idea:** Stack 3×3 convs (same padding so spatial size preserved until pool); periodic max pool to halve size.
- **Forward:** conv blocks (each: conv→BN→ReLU, repeated) → maxpool → … → flatten → FC → logits.
- **Setting up correctly:** Use `nn.Sequential` for each conv block; build list of channel sizes (e.g. [64, 128, 256, 512]) and loop to create layers.

## C.4 Simple CNN from Scratch (Template)

- Conv → BN → ReLU → Pool (repeat); then AdaptiveAvgPool2d(1) → Flatten → Linear.
- Ensure: `out_channels` of block \(i\) = `in_channels` of block \(i+1\); after pool, spatial size halves; final linear input size = last channel count.

## C.5 Checklist for “Building Correctly”

| Step | Check |
|------|--------|
| **Shapes** | Log or assert shapes after each block; especially after stride-2 and skip add. |
| **Device** | All parameters and buffers on same device (model.to(device); data.to(device)). |
| **Modes** | model.train() / model.eval() for BN and Dropout. |
| **Registration** | Every nn.Module used in forward is stored as attribute or inside a registered Sequential. |
| **Output** | Final layer: logits (no softmax if using CrossEntropyLoss). |

## C.6 Minimal Residual Block (Pseudocode)

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm2d(out_c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)
```

**Interview:** Implement ResidualBlock with optional shortcut for stride/change of channels; then sketch full ResNet forward (stem → layers → pool → fc).

---

# Code Challenges Roadmap

| Folder | Content | Time |
|--------|---------|------|
| **10_diffusion_flow/** | Noise prediction model (x_t, t → ε), DDPM training step, buggy diffusion, design Q&A | 2–3 h |
| **11_transformers/** | Scaled dot-product attention, multi-head block, encoder block, small transformer | 2–3 h |
| **12_popular_architectures/** | ResidualBlock, SimpleCNN, VGGBlock, make_layer, Bottleneck, SmallResNet, InceptionModule. **Why only:** Popular_Architectures_Explained.md (no code answers). | 3–5 h |

Run challenges from `practice/week4_pytorch/` (e.g. `python week4_pytorch/10_diffusion_flow/challenge_noise_prediction.py`).

---

**Good luck with diffusion, flow matching, transformers, and architecture design in interviews.**
