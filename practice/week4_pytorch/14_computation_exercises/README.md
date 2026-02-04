# Computation Exercises — Parameters, Dimensions, Complexity

**Purpose:** Master the *mechanics* of convolutions, transformers, and vision transformers so you can answer tricky interview questions about parameters, output shapes, steps, and complexity. Covers **1D, 2D, and 3D** (including video) with explicit dimensions.

---

## What You'll Master

| Topic | What you practice |
|-------|-------------------|
| **Convolutions** | Output shape formula (1D/2D/3D), parameter count, FLOPs, padding/stride/dilation |
| **Transformers** | Embedding → Q/K/V shapes, attention matrix size, FFN params, per-layer and total params |
| **Vision Transformers (ViT)** | Patch embedding (2D image, 3D video), sequence length, positional encoding, param breakdown |
| **Complexity** | Time/space Big-O for attention vs conv, FLOPs, memory; subtle interview gotchas |

---

## Files

| File | Focus |
|------|--------|
| `01_convolution_computation.py` | Conv1d/2d/3d: output dimensions, num_params, optional FLOPs (explicit B, C, L/H/W/D) |
| `02_transformer_computation.py` | Transformer: shapes at each step, param count for MHA + FFN, 1D sequence (B, L, d_model) |
| `03_vision_transformer_computation.py` | ViT 2D (image) and 3D (video): patch grid, seq length, embedding dim, param counts |
| `04_complexity_and_interview_questions.py` | O(L²) attention, conv FLOPs, memory; interview Q&A with detailed answers |

**Study guide:** `practice/study_guides/Computation_Exercises_Guide.md` — formulas, dimension tables, and interview tips.

---

## How to Use

1. **Read** the docstrings and the study guide for formulas.
2. **Implement** the requested functions (output shape, parameter count, etc.) in each challenge file.
3. **Run** each file: `python 01_convolution_computation.py` (etc.) — asserts verify your answers.
4. **Compare** with `solution_*.py` if stuck.
5. **Practice** explaining aloud: "For a 3D conv with input (B, C_in, D, H, W), kernel (kD, kH, kW), the output length in the time dimension is ..."

---

## Conventions (Dimensions)

- **1D (e.g. audio, sequences):** `(B, C_in, L)` → conv → `(B, C_out, L_out)`.
- **2D (e.g. images):** `(B, C_in, H, W)` → conv → `(B, C_out, H_out, W_out)`.
- **3D (e.g. video):** `(B, C_in, T, H, W)` — T = time/frames, H, W = height, width. Conv → `(B, C_out, T_out, H_out, W_out)`.
- **Transformer:** `(B, L, d_model)` — B = batch, L = sequence length, d_model = embedding dimension.
- **ViT:** Image `(B, C, H, W)` → patch embed → `(B, num_patches + 1, d_model)`. Video: `(B, C, T, H, W)` → `(B, num_patches + 1, d_model)` with patches in space-time.

Run from `practice/week4_pytorch`:  
`python 14_computation_exercises/01_convolution_computation.py`  
etc.
