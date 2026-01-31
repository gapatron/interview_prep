# PyTorch Mastery Guide — Adobe Interview Prep

**Purpose:** Complete review for PyTorch challenges: data processing, model training, connecting/concatenating/conditioning, and debugging. **Incremental, accumulating, industry-style.**

**Estimated study time:** 12–20 hours (full loading and mastery)

---

# Practice Roadmap (Cumulative)

| Level | Folder | Focus | Time |
|-------|--------|--------|------|
| **1** | `01_data_processing/`, `02_models/`, `03_training/`, `04_bug_finding/` | Tensors, Dataset, DataLoader, concat, conditioning, training step, find bugs 1–6 | 2–3 h |
| **2** | `05_level2_intermediate/` | Full train+val loop, backbone+head, checkpoint save/load | 2–3 h |
| **3** | `06_level3_advanced/` | Multi-input, gradient clipping, gradient accumulation, mixed precision, gradient checkpointing, PathLabelDataset | 2–3 h |
| **4** | `07_level4_industry/` | Full pipeline, early stopping, LR scheduler | 2–3 h |
| **Cumulative** | `08_cumulative/` | End-to-end pipeline, two-stage (freeze backbone, finetune head) | 1–2 h |
| **More bugs** | `04_bug_finding/` (bugs 7–10) | In-place autograd, loss/target shape, eval(), scheduler step | 1 h |
| **Expert** | `09_expert/` | Custom autograd.Function (LeakyReLU), distributed-ready pattern | 1–2 h |

**Total:** 12–20 hours of hands-on code. Do Level 1 first; each level builds on the previous.

---

# Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [Tensors & Data Processing](#2-tensors--data-processing)
3. [Datasets & DataLoaders](#3-datasets--dataloaders)
4. [Model Building: Layers, Concatenation, Conditioning](#4-model-building-layers-concatenation-conditioning)
5. [Training Loop & Optimization](#5-training-loop--optimization)
6. [Common Bugs & How to Fix Them](#6-common-bugs--how-to-fix-them)
7. [Challenges & Self-Assessment](#7-challenges--self-assessment)
8. [Answer Key](#8-answer-key)
9. [Quizzes](#9-quizzes)
10. [Quiz Answer Key](#10-quiz-answer-key)

---

# 1. Core Concepts

## 1.1 Why PyTorch?

- **Dynamic computation graph** — define-by-run; easier to debug.
- **Tensors** — like NumPy arrays but on CPU/GPU, with autograd.
- **Device** — `.to(device)` or `.cuda()` / `.cpu()`.

## 1.2 Key Imports (Memorize)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```

## 1.3 Device Handling

**Fill in the blanks:**

```python
device = torch.device("______" if torch.cuda.is_available() else "______")
model = MyModel().______(device)
x = x.______(device)
```

**Answers:** `"cuda"`, `"cpu"`, `.to(device)`, `.to(device)`.

## 1.4 Tensor Basics

| Operation        | Code                    | Notes                    |
|-----------------|-------------------------|--------------------------|
| Create from list| `torch.tensor([1,2,3])` | dtype inferred           |
| Zeros           | `torch.zeros(2, 3)`    | Shape (2, 3)             |
| Ones            | `torch.ones(2, 3)`      |                          |
| Random          | `torch.randn(2, 3)`     | Standard normal          |
| From NumPy      | `torch.from_numpy(arr)` | Shares memory by default |
| To NumPy        | `tensor.numpy()`        | CPU only                 |
| Shape           | `tensor.shape` or `.size()` |                    |
| Reshape         | `tensor.view(-1, 4)` or `.reshape(-1, 4)` | `-1` infers |

---

# 2. Tensors & Data Processing

## 2.1 Slicing, Indexing, Masking

```python
# Slicing: same as NumPy
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
t[0, :]      # first row
t[:, 1]      # second column
t[1, 1:3]    # row 1, cols 1–2

# Boolean mask
mask = t > 3
t[mask]      # 1D tensor of elements where mask is True
```

## 2.2 Concatenation & Stacking

| Goal              | Code                          | Output shape (example)     |
|-------------------|-------------------------------|----------------------------|
| Concatenate dim 0 | `torch.cat([a, b], dim=0)`    | (n1+n2, C)                 |
| Concatenate dim 1 | `torch.cat([a, b], dim=1)`    | (N, C1+C2)                 |
| Stack new dim     | `torch.stack([a, b], dim=0)` | (2, N, C)                  |

**Rule:** `cat` preserves existing dims; `stack` adds a new dim.

## 2.3 Squeeze & Unsqueeze

```python
x = torch.randn(1, 4, 1, 8)
x.squeeze()    # removes all size-1 dims → (4, 8)
x.unsqueeze(0) # add dim at 0 → (1, 1, 4, 1, 8)
x.unsqueeze(1) # add dim at 1 → (1, 1, 4, 1, 8)
```

**Interview tip:** Wrong `dim` in `cat`/`stack`/`squeeze`/`unsqueeze` is a common bug.

## 2.4 Permute & Transpose

```python
x = torch.randn(2, 3, 4)  # (N, C, L)
x.permute(0, 2, 1)        # (N, L, C)
x.transpose(1, 2)         # same as permute(0, 2, 1) for 3D
```

## 2.5 Fill-in: Data Processing

```python
# Given: a (N, 3, 32, 32) and b (N, 5). Produce (N, 8, 32, 32).
# b must be broadcast to (N, 5, 1, 1), then expanded to (N, 5, 32, 32).

b_expanded = b.______(______).______(______)
out = torch.______([a, b_expanded], dim=______)
```

**Answer:** `b.unsqueeze(2).unsqueeze(3).expand(-1, -1, 32, 32)` then `torch.cat([a, b_expanded], dim=1)`.

---

# 3. Datasets & DataLoaders

## 3.1 Custom Dataset Template

```python
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.______)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.______(x), torch.______(y)
```

**Answers:** `data` (or `labels`), `torch.tensor(x, dtype=...)`, `torch.tensor(y, dtype=...)`.

## 3.2 DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,  # 0 for Windows/debug
    pin_memory=True  # faster GPU transfer
)
for batch_x, batch_y in loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    ...
```

## 3.3 Common Bug: Wrong dtype

- Labels for classification: `torch.long` (integer).
- Inputs: `torch.float32`.
- Using `float` labels with `nn.CrossEntropyLoss` → **error**.

---

# 4. Model Building: Layers, Concatenation, Conditioning

## 4.1 Sequential & Module

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2)
)
# Or with nn.Module (flexible)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

## 4.2 Concatenating Features (Skip Connections / Multi-Input)

```python
class ConcatBlock(nn.Module):
    def __init__(self, in_a, in_b, out):
        super().__init__()
        self.fc = nn.Linear(in_a + in_b, out)  # cat along feature dim

    def forward(self, x_a, x_b):
        # x_a: (N, in_a), x_b: (N, in_b)
        x = torch.cat([x_a, x_b], dim=1)
        return self.fc(x)
```

## 4.3 Conditioning (e.g. FiLM, Embedding Conditioning)

```python
# Condition linear layer by a vector (e.g. class embedding)
class ConditionalLinear(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gamma = nn.Linear(cond_dim, out_dim)  # scale
        self.beta = nn.Linear(cond_dim, out_dim)   # shift

    def forward(self, x, cond):
        out = self.fc(x)
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return gamma * out + beta
```

## 4.4 Connecting Two Modules (Plugin Style)

```python
class Encoder(nn.Module):
    def forward(self, x):
        return self.backbone(x)

class Decoder(nn.Module):
    def forward(self, z):
        return self.head(z)

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

## 4.5 Shape Checklist

Before `cat`/`stack`/`matmul`:

- Batch dim usually `0`.
- Linear: `(N, in_features)` → `(N, out_features)`.
- Conv2d: `(N, C, H, W)`; after conv, check `(N, C_out, H_out, W_out)`.
- Always print `tensor.shape` when debugging.

---

# 5. Training Loop & Optimization

## 5.1 Standard Supervised Loop (Classification)

```python
model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.______()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.______()
        optimizer.______()
```

**Answers:** `zero_grad()`, `backward()`, `step()`.

## 5.2 Evaluation Mode

```python
model.eval()
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        logits = model(batch_x.to(device))
        ...
```

## 5.3 Gradient Flow Checklist

- `loss.backward()` before `optimizer.step()`.
- `optimizer.zero_grad()` each step (or accumulate intentionally).
- Parameters that should train: `requires_grad=True` (default for `nn.Parameter`).

---

# 6. Common Bugs & How to Fix Them

| Bug | Symptom | Fix |
|-----|---------|-----|
| Shape mismatch in `cat` | RuntimeError: size mismatch | Align dims with `unsqueeze`/`expand`/`view`; check `dim`. |
| Loss not decreasing | Loss constant or NaN | Check lr, `zero_grad`, labels dtype (long for CE), data scale. |
| CUDA out of memory | OOM | Smaller batch_size, `torch.no_grad()` for val, clear cache. |
| Single-element tensor as scalar | TypeError in condition | Use `item()`: `if loss.item() < 0.5`. |
| Wrong device | Tensor not on same device | `.to(device)` for model, inputs, labels. |
| Labels float for CrossEntropyLoss | RuntimeError | Convert to `torch.long`. |
| Forgetting `model.train()`/`eval()` | Dropout/BatchNorm behave wrong | Call `train()` in training, `eval()` in validation. |

---

# 7. Challenges & Self-Assessment

## Challenge 1: Data Processing

**Task:** Implement a function that takes `x` (N, C, H, W) and `cond` (N, D). Reshape `cond` to (N, D, 1, 1), expand to (N, D, H, W), concatenate with `x` along channels, return (N, C+D, H, W).

**Your code:**

```python
def concat_condition_to_feature_map(x, cond):
    # x: (N, C, H, W), cond: (N, D)
    # TODO
    pass
```

## Challenge 2: Custom Dataset

**Task:** Implement `TensorDataset` that takes two tensors `X` and `y`, and in `__getitem__` returns `(X[idx], y[idx])`.

**Your code:**

```python
class TensorDataset(Dataset):
    # TODO
    pass
```

## Challenge 3: Conditional MLP

**Task:** Build an MLP that takes input `x` (N, in_dim) and condition `c` (N, c_dim). First layer: `Linear(in_dim, h)`. Then concatenate hidden state with `c`, then `Linear(h + c_dim, num_classes)`.

**Your code:**

```python
class ConditionalMLP(nn.Module):
    # TODO
    pass
```

## Challenge 4: Find the Bug

**Code:**

```python
class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)

# Training:
loss = nn.CrossEntropyLoss()(output, labels)  # labels are float in [0, 1]
```

**What’s wrong?** (Answer in Answer Key.)

## Challenge 5: Training Loop

**Task:** Write a minimal training loop for 1 epoch: DataLoader, model, Adam, CrossEntropyLoss. Include `zero_grad`, `backward`, `step`, and moving tensors to device.

---

# 8. Answer Key

## 8.1 Challenge 1: concat_condition_to_feature_map

```python
def concat_condition_to_feature_map(x, cond):
    N, C, H, W = x.shape
    D = cond.shape[1]
    cond = cond.unsqueeze(2).unsqueeze(3).expand(N, D, H, W)
    return torch.cat([x, cond], dim=1)
```

## 8.2 Challenge 2: TensorDataset

```python
class TensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

## 8.3 Challenge 3: ConditionalMLP

```python
class ConditionalMLP(nn.Module):
    def __init__(self, in_dim, c_dim, h, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h)
        self.fc2 = nn.Linear(h + c_dim, num_classes)

    def forward(self, x, c):
        h = torch.relu(self.fc1(x))
        h_c = torch.cat([h, c], dim=1)
        return self.fc2(h_c)
```

## 8.4 Challenge 4: Bug

- **Bug:** `CrossEntropyLoss` expects **integer** class indices (long), not float in [0, 1].
- **Fix:** `labels = labels.long()` or use `torch.randint`/integer labels from the dataset.

## 8.5 Challenge 5: Training Loop (minimal)

```python
model.train()
for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()
    optimizer.step()
```

---

---

# Code Challenges (Practice Folder)

All code lives in **`practice/week4_pytorch/`**:

| Folder | Content |
|--------|---------|
| `01_data_processing/` | Tensors, concat, condition_to_feature_map, Dataset, DataLoader |
| `02_models/` | ConcatBlock, ConditionalMLP, Encoder/Decoder/FullModel |
| `03_training/` | Training loop (zero_grad, forward, loss, backward, step) |
| `04_bug_finding/` | Bugs 1–10: labels dtype, cat dim, device, eval(), shape, in-place, target shape, BN eval, scheduler |
| `05_level2_intermediate/` | Train+val loop, backbone+head, checkpoint save/load |
| `06_level3_advanced/` | Multi-input, gradient clipping, gradient accumulation, mixed precision, gradient checkpointing, PathLabelDataset |
| `07_level4_industry/` | Full pipeline, early stopping, LR scheduler |
| `08_cumulative/` | End-to-end pipeline, two-stage (freeze backbone, finetune head) |
| `09_expert/` | Custom autograd.Function (LeakyReLU), distributed-ready (rank, DDP stub) |

- **Challenge files:** `challenge_*.py` / `buggy_code_*.py` / `full_pipeline.py` — fill in TODOs or fix bugs.
- **Solution files:** `solution_*.py` — reference implementations.

Run from `practice/`:

```bash
cd practice
python week4_pytorch/01_data_processing/challenge_1_tensors.py   # after filling TODOs
python week4_pytorch/04_bug_finding/buggy_code_1.py              # will error until fixed
python week4_pytorch/05_level2_intermediate/solution_train_val_loop.py
python week4_pytorch/07_level4_industry/solution_full_pipeline.py
python week4_pytorch/08_cumulative/solution_end_to_end.py
```

---

# Industry Patterns (Quick Reference)

| Pattern | What to do |
|--------|------------|
| **Device** | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`; move model and every batch: `model.to(device)`, `x.to(device)`. |
| **Seed** | `torch.manual_seed(42)`; `torch.cuda.manual_seed_all(42)` for reproducibility. |
| **Train/Val** | `model.train()` in training loop; `model.eval()` + `torch.no_grad()` in validation. |
| **Labels** | Classification: integer indices `torch.long`; CrossEntropyLoss expects class indices, not one-hot. |
| **Checkpoint** | `torch.save({"model": model.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}, path)`; load with `model.load_state_dict(torch.load(path, map_location="cpu")["model"])`. |
| **Save best** | Track `best_val_acc`; when `val_acc > best_val_acc`, save checkpoint and update `best_val_acc`. |
| **Backbone + head** | `forward(x) = self.head(self.backbone(x))`; backbone outputs features, head outputs logits. |
| **Multi-input** | Two branches (e.g. image branch, meta branch); `torch.cat([feat_a, feat_b], dim=1)` then head. |
| **Gradient clipping** | After `loss.backward()`, before `optimizer.step()`: `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. |
| **Gradient accumulation** | Run several mini-batches with backward (no step); zero_grad only at start of group; loss /= accum_steps; step every accum_steps. Simulates larger batch. |
| **Mixed precision (FP16)** | `torch.amp.GradScaler("cuda")`; `with torch.amp.autocast("cuda"):` forward + loss; `scaler.scale(loss).backward()`; `scaler.step(optimizer)`; `scaler.update()`. Saves memory and speed on GPU. |
| **Gradient checkpointing** | `checkpoint(fn, *args)` runs fn in forward but recomputes it in backward; saves activation memory, increases compute. Use for heavy submodules. |

---

# Mixed Precision and Gradient Checkpointing (Concepts)

## Mixed precision (FP16)

- **What:** Run forward (and sometimes backward) in FP16 where safe; keep master weights in FP32. Reduces memory and can speed up on Tensor Core GPUs.
- **GradScaler:** Scales the loss before backward so small FP16 gradients don’t underflow; unscales before optimizer.step(); updates scale factor.
- **Order:** `with torch.amp.autocast("cuda"):` forward + loss → `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()`. Use only when `device.type == "cuda"`.

## Gradient checkpointing

- **What:** Don’t store intermediate activations for a submodule; recompute them during backward. **Trade-off:** less memory, more compute.
- **When:** Use for large layers, long sequences, or when you hit OOM. Wrap the heavy part in `torch.utils.checkpoint.checkpoint(fn, *args)`.
- **Rule:** The callable `fn` must take the saved tensors as args and return the output; it will be called again in backward.

---

# More Challenges (Fill-in / Find Bug)

## Level 2: Validation loop

Implement `validate(model, loader, criterion, device)` returning `(avg_loss, accuracy)`. Use `model.eval()` and `torch.no_grad()`; count correct with `(logits.argmax(1) == y).sum().item()`.

## Level 2: Backbone + head

Backbone: `x (N, C, H, W) -> view(N, -1) -> Linear -> ReLU -> (N, hidden)`. Head: `Linear(hidden, num_classes)`. Full: `head(backbone(x))`.

## Level 3: Multi-input

Image branch: flatten + Linear -> ReLU. Meta branch: Linear -> ReLU. Concat features along dim=1, then Linear to num_classes.

## Level 4: Full pipeline

1. set_seed; 2. build_dataloaders (TensorDataset, random_split, DataLoader); 3. model, criterion, optimizer; 4. for epoch: train_one_epoch, validate; if val_acc > best_acc: save checkpoint; 5. print best_val_acc.

## Bug 4: Device mismatch

Move batch to device: `x, y = x.to(device), y.to(device)` before `model(x)`.

## Bug 5: Forgot model.eval()

In validation, call `model.eval()` so Dropout/BatchNorm use eval behavior.

## Bug 6: Shape in forward

Pass backbone output to head: `return self.head(self.backbone(x))`, not `return self.head(x)`.

## Bug 7: In-place op breaking autograd

Use out-of-place `torch.relu(x)` instead of `x.relu_()` on tensors that need grad.

## Bug 8: Loss / target shape

CrossEntropyLoss expects targets shape `(N,)` and dtype `long`. Use `targets.squeeze()` or create `(N,)` from the start.

## Bug 9: Forgetting model.eval()

In validation, call `model.eval()` so BatchNorm uses running stats and Dropout is disabled.

## Bug 10: LR scheduler step

Call `scheduler.step()` once per epoch (after the training loop), not every batch.

---

# Adobe-Style Interview (What to Expect)

- **Data:** Tensor creation, `cat`/`stack`/`view`, custom `Dataset`/`DataLoader`, device placement.
- **Models:** Concatenation, conditioning (concat cond with features), backbone + head, multi-input (image + metadata).
- **Training:** Full loop (zero_grad, forward, loss, backward, step), train vs val (eval, no_grad), checkpointing, save best by val metric.
- **Debugging:** Wrong `dim` in `cat`, device mismatch, labels dtype/shape, forgetting `eval()`, in-place ops, scheduler step placement.
- **Industry:** Gradient clipping, early stopping, LR scheduler, two-stage (freeze then finetune), optional: custom backward, DDP-ready structure.

**Harder challenges (plugging things onto other things):** Two-stage training, multi-input classifier, gradient accumulation + clipping in one loop, full pipeline with early stopping and scheduler.

---

# 9. Quizzes

Test yourself before or after the code challenges. Answers are in section 10.

**Q1.** Why do we scale the loss by `1 / accum_steps` in gradient accumulation?

**Q2.** In mixed precision training, what is the role of `GradScaler`?

**Q3.** What is the trade-off of gradient checkpointing? When would you use it?

**Q4.** What is the correct order in a training step: zero_grad, forward, loss, backward, step? Where does gradient clipping go?

**Q5.** Why do we call `model.eval()` before validation? What happens to BatchNorm and Dropout?

**Q6.** CrossEntropyLoss expects logits of shape (N, C). What shape and dtype must the targets be?

**Q7.** You concatenate two feature tensors of shape (N, A) and (N, B) to feed a Linear layer. What must the Linear layer’s `in_features` be? Along which dim do you concat?

**Q8.** You want to simulate a batch size of 32 but only have memory for 8. How many accumulation steps do you need? When do you call optimizer.zero_grad() and optimizer.step()?

---

# 10. Quiz Answer Key

**A1.** So that the effective gradient is the same as if we had computed one backward on the full accumulated batch. Each mini-batch contributes `grad / accum_steps`; after accum_steps steps the sum equals the full-batch gradient.

**A2.** GradScaler scales the loss before backward so that FP16 gradients don’t underflow. It unscales gradients before optimizer.step() and updates its scale factor (e.g. when no inf/nan).

**A3.** Trade-off: less activation memory, more compute (recompute in backward). Use when you hit OOM or have large layers/sequences and can afford extra compute.

**A4.** Order: zero_grad → forward → loss → backward → (optional: clip_grad_norm_) → step. Clipping goes after backward, before step.

**A5.** model.eval() sets BatchNorm to use running stats (not batch stats) and Dropout to disable (no dropout). So validation metrics are consistent and not stochastic.

**A6.** Targets shape (N,) and dtype long (integer class indices). Not (N, 1) and not float.

**A7.** in_features = A + B. Concat along dim=1 so the tensor is (N, A+B).

**A8.** 32 / 8 = 4 accumulation steps. zero_grad() at the start of each group (e.g. every 4 batches); step() after every 4 batches (and optionally once more at the end if the last group is incomplete).

---

**See also:** `Advanced_Architectures_Guide.md` — chapters on **diffusion models & flow matching** (design, bugs, alternatives), **building the Transformer** from scratch (attention, multi-head, encoder block), and **popular architectures** (ResNet-style residual block, simple CNN from scratch). Code: `week4_pytorch/10_diffusion_flow/`, `11_transformers/`, `12_popular_architectures/`.

**Good luck with your Adobe interview.**
