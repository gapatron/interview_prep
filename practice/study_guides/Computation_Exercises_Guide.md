# Computation Exercises Guide — Parameters, Dimensions, Complexity

**Purpose:** One place for all formulas and dimension conventions used in `week4_pytorch/14_computation_exercises/`. Use this when implementing the challenges and when rehearsing for interview questions on convolutions, transformers, and vision transformers (including 3D/video).

---

## Table of Contents

1. [Convolutions (1D, 2D, 3D)](#1-convolutions-1d-2d-3d)
2. [Transformers (1D sequence)](#2-transformers-1d-sequence)
3. [Vision Transformers (2D image, 3D video)](#3-vision-transformers-2d-image-3d-video)
4. [Complexity and FLOPs](#4-complexity-and-flops)
5. [Interview tips and gotchas](#5-interview-tips-and-gotchas)

---

## 1. Convolutions (1D, 2D, 3D)

### 1.1 Output shape (single dimension)

For one spatial/temporal dimension with input length \(L_{\text{in}}\), kernel size \(K\), stride \(S\), padding \(P\):

\[
L_{\text{out}} = \left\lfloor \frac{L_{\text{in}} + 2P - K}{S} \right\rfloor + 1
\]

- **Conv1d:** input `(B, C_in, L)` → output `(B, C_out, L_out)`.
- **Conv2d:** apply the formula to H and W separately (with their own K, S, P if using tuples). Input `(B, C_in, H, W)` → output `(B, C_out, H_out, W_out)`.
- **Conv3d:** apply to T, H, W. Input `(B, C_in, T, H, W)` → output `(B, C_out, T_out, H_out, W_out)`.

**Example (2D):** \(H = W = 32\), \(K = 3\), \(S = 1\), \(P = 1\) → \(H_{\text{out}} = (32 + 2 - 3)/1 + 1 = 32\).

**Example (3D video):** Input `(B, 3, 8, 224, 224)`, kernel `(3,3,3)`, stride 1, padding 1 → output `(B, C_out, 8, 224, 224)`.

### 1.2 Parameter count

- **Conv1d:** \(C_{\text{in}} \times C_{\text{out}} \times K + C_{\text{out}}\) (if bias).
- **Conv2d:** \(C_{\text{in}} \times C_{\text{out}} \times K_H \times K_W + C_{\text{out}}\).
- **Conv3d:** \(C_{\text{in}} \times C_{\text{out}} \times K_T \times K_H \times K_W + C_{\text{out}}\).

Parameters do **not** depend on input length or spatial size—only on channels and kernel size.

### 1.3 FLOPs (mult-adds, forward pass)

Each output element requires \(C_{\text{in}} \times (K_1 \cdots K_d)\) multiplies and the same number of adds (roughly), then \(C_{\text{out}}\) outputs per position. So:

- **Conv1d:** \((C_{\text{in}} \cdot K \cdot C_{\text{out}}) \times L_{\text{out}}\).
- **Conv2d:** \((C_{\text{in}} \cdot K_H \cdot K_W \cdot C_{\text{out}}) \times H_{\text{out}} \times W_{\text{out}}\).
- **Conv3d:** \((C_{\text{in}} \cdot K_T \cdot K_H \cdot K_W \cdot C_{\text{out}}) \times T_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}\).

### 1.4 Dilation (optional)

If dilation \(D\) is used, effective kernel size per dimension is \(K + (K-1)(D-1)\). Then the output-length formula uses this effective size. (Exercises use no dilation by default.)

---

## 2. Transformers (1D sequence)

### 2.1 Dimension conventions

- Input: token indices `(B, L)` → after embedding `(B, L, d_model)`.
- **B** = batch size, **L** = sequence length, **d_model** = embedding dimension.
- **num_heads** = H, **head_dim** = d_model / H (must divide evenly).

### 2.2 Shapes through multi-head attention

| Step | Shape |
|------|--------|
| Input | `(B, L, d_model)` |
| Q, K, V after linear | `(B, L, d_model)` each |
| Reshape to heads | `(B, H, L, head_dim)` |
| Scores Q @ K^T | `(B, H, L, L)` |
| Softmax + @ V | `(B, H, L, head_dim)` |
| Concat heads | `(B, L, d_model)` |
| Output projection W_out | `(B, L, d_model)` |

### 2.3 Parameter count (one encoder block)

- **Multi-Head Attention:** 4 linear layers (Q, K, V, out), each \(d_{\text{model}} \times d_{\text{model}}\).  
  Total: \(4 \cdot d_{\text{model}}^2 + 4 \cdot d_{\text{model}}\) (with bias).  
  No extra parameters for “heads”—they are a reshape of the same weights.

- **Feed-Forward:** Linear(\(d_{\text{model}} \to d_{\text{ff}}\)), then Linear(\(d_{\text{ff}} \to d_{\text{model}}\)).  
  \(d_{\text{model}} \cdot d_{\text{ff}} + d_{\text{ff}} + d_{\text{ff}} \cdot d_{\text{model}} + d_{\text{model}}\) (with bias).  
  Often \(d_{\text{ff}} = 4 \cdot d_{\text{model}}\).

- **Two LayerNorms:** each has weight and bias of size \(d_{\text{model}}\).  
  \(2 \times 2 \cdot d_{\text{model}} = 4 \cdot d_{\text{model}}\).

### 2.4 FLOPs (attention)

- Q @ K^T: `(B, L, d_model)` @ `(B, d_model, L)` → \(B \cdot L^2 \cdot d_{\text{model}}\) mult-adds.
- attn @ V: `(B, L, L)` @ `(B, L, d_model)` → \(B \cdot L^2 \cdot d_{\text{model}}\).
- Total: \(\approx 2 \cdot B \cdot L^2 \cdot d_{\text{model}}\) (dominant term).

---

## 3. Vision Transformers (2D image, 3D video)

### 3.1 ViT 2D (image)

- Input: `(B, C, H, W)` (e.g. B, 3, 224, 224).
- Patch size: \(P \times P\) (e.g. 16×16).
- Number of patches: \(N = (H/P) \times (W/P)\) (e.g. 14×14 = 196).
- Patch embedding: Conv2d(C, d_model, kernel_size=P, stride=P) → `(B, d_model, H/P, W/P)` → flatten to `(B, d_model, N)` → transpose to `(B, N, d_model)`.
- Prepend [CLS] token: `(B, 1, d_model)` → sequence length **L = N + 1** (e.g. 197).
- Rest of the model: standard transformer encoder on `(B, L, d_model)`.

**Parameter count (patch embed):** \(C \cdot d_{\text{model}} \cdot P \cdot P + d_{\text{model}}\).

### 3.2 ViT 3D (video)

- Input: `(B, C, T, H, W)` (e.g. B, 3, 16, 224, 224).
- Patch size: \((P_t, P_h, P_w)\) (e.g. 2×16×16).
- Number of patches: \(N = (T/P_t) \times (H/P_h) \times (W/P_w)\) (e.g. 8×14×14 = 1568).
- Patch embedding: Conv3d(C, d_model, kernel_size=(P_t,P_h,P_w), stride=(P_t,P_h,P_w)) → flatten → add [CLS] → **L = N + 1**.
- Parameters: \(C \cdot d_{\text{model}} \cdot P_t \cdot P_h \cdot P_w + d_{\text{model}}\).

### 3.3 Full ViT encoder params

- Patch embedding params (above).
- Plus **num_layers** × (one encoder block: MHA + FFN + 2×LayerNorm).
- Plus final LayerNorm: \(2 \cdot d_{\text{model}}\).

---

## 4. Complexity and FLOPs

| Component | Time (FLOPs) | Space (activation memory) |
|-----------|----------------|----------------------------|
| Self-attention | \(O(L^2 \cdot d)\) | \(O(B \cdot H \cdot L^2)\) for scores |
| Conv2d layer | \(O(C_{\text{in}} \cdot C_{\text{out}} \cdot K^2 \cdot H \cdot W)\) | \(O(B \cdot C_{\text{out}} \cdot H \cdot W)\) |
| Conv3d layer | \(O(C_{\text{in}} \cdot C_{\text{out}} \cdot K_T \cdot K_H \cdot K_W \cdot T \cdot H \cdot W)\) | Same idea in 3D |

- **Why attention is expensive:** Pairwise comparisons over L positions → L×L matrix → quadratic in L.
- **Why conv is linear in spatial size:** Each output position only looks at a local window; total work scales as number of output positions (linear in length/area/volume when stride and kernel are fixed).

---

## 5. Interview tips and gotchas

1. **Output shape:** Always state the formula; give one concrete example (e.g. 32, K=3, S=1, P=1 → 32).
2. **Parameters:** Emphasize “no dependence on H, W” for conv—only channels and kernel. For MHA, “4 × d_model²” and “heads are a reshape.”
3. **ViT sequence length:** “Num patches + 1 for [CLS].” For video: “(T/P_t)×(H/P_h)×(W/P_w) + 1.”
4. **Conv3d for video:** Input layout (B, C, T, H, W); kernel (kT, kH, kW). Same formula per dimension.
5. **LayerNorm params:** 2·d_model (one scale, one shift).
6. **Tricky:** “Does multi-head attention have more parameters than single-head?” No—same 4 linear layers; heads just split the output dimension.
7. **Tricky:** “What happens to FLOPs if you double the sequence length in a transformer?” Attention FLOPs go up by 4× (L²).
8. **Tricky:** “What happens to conv FLOPs if you double the image size?” Roughly 4× for 2D (double H and W).

Use the exercises in `14_computation_exercises/` to implement these formulas and run the asserts; use this guide to rehearse short, precise answers.
