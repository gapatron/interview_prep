# PyTorch Track — Pedagogical Roadmap

This document describes **how the challenges build** and **why they are ordered this way**. Use it to plan your session and to see how each piece connects.

---

## Design principles

1. **Incremental** — Each folder assumes you have done the previous ones. Concepts (device, Dataset, training step) are introduced once and reused.
2. **Cumulative** — Later challenges *plug in* earlier ones: the full pipeline uses your DataLoader, your training loop, your checkpoint logic.
3. **No spoilers in challenges** — Challenge text explains *what* to implement and *why it matters*; it does not paste the code. Solutions are in separate `solution_*.py` files for reference.
4. **Bug-finding as review** — The bug series reinforces device, dtypes, shapes, eval mode, and scheduler use by having you *find* the mistake.

---

## Level 1: Foundations (01 → 02 → 03 → 04)

**Order:** `01_data_processing` → `02_models` → `03_training` → `04_bug_finding`

| Folder | What you learn | Why it comes here |
|--------|----------------|--------------------|
| **01_data_processing** | Device, tensor shapes, `cat` vs `stack`, conditioning a feature map (broadcast + concat). Custom `Dataset` and `DataLoader`. | You need correct shapes and device placement in every script. Dataset/DataLoader are the standard way to feed batches. |
| **02_models** | Concatenating features (ConcatBlock), conditioning an MLP (ConditionalMLP), connecting modules (Encoder → Decoder → FullModel). | These are the building blocks for multi-input and backbone+head designs you’ll use in later levels. |
| **03_training** | One-epoch loop: `zero_grad`, forward, loss, `backward`, `step`, and moving batches to device. | This is the core training step; every subsequent level reuses or extends it. |
| **04_bug_finding** | Common mistakes: labels dtype, `cat` dim, device mismatch, forgetting `eval()`, wrong shape in forward, in-place ops, target shape, scheduler step. | Fixing bugs reinforces the same concepts you used in 01–03; doing it in isolation sharpens debugging. |

**After Level 1** you can: create tensors and Datasets, build small modular models (concat, conditioning, encoder–decoder), run one epoch of training, and spot frequent PyTorch bugs.

---

## Level 2: Loops and checkpoints (05)

**Order:** After 01–04.

| Folder | What you learn | Why it comes here |
|--------|----------------|--------------------|
| **05_level2_intermediate** | Full train+validation loop (`train_one_epoch`, `validate`, `train_for_epochs`). Backbone+head model. Saving and loading checkpoints. | Real training always has validation and checkpointing; backbone+head is the standard pattern for classifiers and transfer learning. |

**After Level 2** you can: train for multiple epochs with validation, log metrics, and save/load state dicts. You can also structure a model as backbone (features) + head (logits).

---

## Level 3: Advanced patterns (06)

**Order:** After 05.

| Folder | What you learn | Why it comes here |
|--------|----------------|--------------------|
| **06_level3_advanced** | Multi-input model, gradient clipping, gradient accumulation, **mixed precision (FP16)**, **gradient checkpointing**, custom Dataset from path→tensor mapping. | Standard industry patterns. Mixed precision saves memory and speed on GPU; gradient checkpointing trades compute for memory. |

**After Level 3** you can: combine multiple inputs in one model, stabilize training with gradient clipping, simulate larger batches with accumulation, train with mixed precision (FP16) on CUDA, use gradient checkpointing to save memory, and write path-based Datasets.

---

## Level 4: Full pipeline (07)

**Order:** After 06.

| Folder | What you learn | Why it comes here |
|--------|----------------|--------------------|
| **07_level4_industry** | End-to-end pipeline: seed, device, data split, model, train+val loop, save best by validation metric. Early stopping and LR scheduler. | This ties everything together into one runnable script and introduces common training “knobs.” |

**After Level 4** you can: run a full training job with best-model saving, early stopping, and a learning-rate schedule.

---

## Cumulative and expert (08, 09)

| Folder | What you learn | Why it comes here |
|--------|----------------|--------------------|
| **08_cumulative** | End-to-end pipeline in one script; two-stage training (train full model, then freeze backbone and finetune head). | No new concepts—you wire Level 1–4 pieces together. Two-stage is a standard transfer-learning pattern. |
| **09_expert** | Custom `autograd.Function` (e.g. LeakyReLU); code structure for distributed training (rank, world size, where DDP/sampler would go). | For interviews that touch custom backward or multi-GPU; the rest of the track does not depend on this. |

---

## Advanced architectures (10, 11, 12)

These can be done **after Level 2** (or in parallel with 06–08) if you care about diffusion, transformers, or CNNs.

| Folder | What you learn | Dependencies |
|--------|----------------|--------------|
| **10_diffusion_flow** | Noise-prediction model (x_t, t → ε), DDPM training step, alpha_bar shape bug. | Device, tensors, simple nn.Module. |
| **11_transformers** | Scaled dot-product attention, multi-head attention, encoder block (Pre-LN, self-attn, FFN). | Attention formula and shapes; see Advanced_Architectures_Guide. |
| **12_popular_architectures** | ResidualBlock, SimpleCNN, VGGBlock, make_layer, Bottleneck, SmallResNet, InceptionModule. | Use *Popular_Architectures_Explained.md* for the “why”; challenges give structure and asserts only. |

---

## Suggested study order

- **Core track (interview essentials):** 01 → 02 → 03 → 04 → 05 → 07 → 08. Add 06 if you have time.
- **With bugs as review:** Do 04 (bugs 1–6) after 03; do bugs 7–10 after 05 or 07.
- **With advanced topics:** After 05, add 10, 11, or 12 as needed; 09 when you want custom autograd or distributed structure.

---

## How to use the materials

1. **Challenges** — Read the docstring (learning goal, what to implement). Implement without peeking at solutions. Run the file to check shape asserts or behavior.
2. **Solutions** — Use after you’ve attempted the challenge, to compare or to unstick. They are reference implementations, not the only correct answer.
3. **Study guides** — *PyTorch_Mastery_Guide* for core concepts and quick reference; *Advanced_Architectures_Guide* for diffusion, transformers, and architectures; *Popular_Architectures_Explained* for the “why” behind ResNet/VGG/Inception (no code).
4. **Bug-finding** — Run the buggy script, see the failure, fix it, then compare with the solution to see the intended fix and reason.

Good luck — work through in order when possible, and use this roadmap to see how each piece fits.
