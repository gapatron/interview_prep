# Popular Architectures Explained — The “Why,” No Code Answers

**Purpose:** Understand *why* each design choice exists. Use this when implementing the challenges in `week4_pytorch/12_popular_architectures/`. **No implementation code here** — only concepts, shapes, and common mistakes so you can derive the code yourself.

---

## 1. Why Residual Connections? (ResNet)

**Problem:** Very deep plain conv nets (e.g. VGG-style) get *worse* as you add layers — training loss goes up, gradients vanish. The network could in principle learn identity layers (output = input) to preserve depth, but in practice it doesn’t.

**Idea:** Don’t make the block predict \(H(x)\). Make it predict the *residual* \(F(x) = H(x) - x\). Then the block output is \(y = F(x) + x\). So:
- The block only has to learn *changes*; if nothing needs to change, it can learn \(F(x) \approx 0\).
- The **skip connection** \(+\, x\) gives a direct path for gradients: \(\frac{\partial \mathcal{L}}{\partial x}\) gets at least one term that doesn’t go through the conv layers, so gradients don’t vanish.
- Adding \(x\) and \(F(x)\) only works if they have the **same shape**. So when you change spatial size (stride &gt; 1) or channel count (in_c ≠ out_c), you must **project** \(x\) to match (e.g. 1×1 conv with the same stride so spatial and channel dimensions match).

**Takeaway:** Residual block = main path (convs) + shortcut. Shortcut is identity when shapes already match; otherwise it’s a small projection (1×1 conv + BN) so you can add.

---

## 2. Why BatchNorm *After* Conv, *Before* ReLU?

**Order:** `Conv → BatchNorm → ReLU` (and again for a second conv in the block).

- **BatchNorm** normalizes the pre-activation (conv output) to zero mean / unit variance (per channel). That stabilizes training and lets you use higher learning rates.
- Putting BN *before* ReLU means you’re normalizing the “raw” activations; ReLU then thresholds. If you put ReLU first, you’d be normalizing many zeros (sparse), which changes the role of BN.
- Convention in ResNet/VGG-style: **Conv → BN → ReLU**. The “main path” in a residual block is: conv1 → BN → ReLU → conv2 → BN; then you add the shortcut and apply **one more ReLU** on the sum (so the residual path output is non-negative after the add).

**Takeaway:** Conv outputs are normalized (BN), then non-linearity (ReLU). Final ReLU after (main + shortcut) so the block output is non-negative.

---

## 3. Why Stacks of 3×3 Convolutions?

**Receptive field:** Two 3×3 convs (with padding 1) have the same effective receptive field as one 5×5, but **fewer parameters** (2×9 vs 25 per channel). Three 3×3s match one 7×7 with even more parameter savings.

**VGG design:** Use only 3×3 convs (and 2×2 pool). Same padding (pad=1) keeps spatial size unchanged until you explicitly add a pooling layer. So you get depth (many layers) and a large receptive field without one huge kernel.

**Takeaway:** 3×3 with padding=1 preserves spatial size; stacking them grows receptive field and keeps params low. Pool when you want to downsample.

---

## 4. Why a Shortcut When stride ≠ 1 or in_c ≠ out_c?

**Residual:** \(y = F(x) + x\). For `+` to be valid, `F(x)` and `x` must have the same shape `(N, C, H, W)`.

- **stride=2** in the main path: spatial size is halved. \(x\) is still (N, in_c, H, W). So you need a shortcut that maps \(x\) to (N, out_c, H/2, W/2). A **1×1 conv with stride=2** (and out_c channels) does that; then BN so scale matches the main path.
- **in_c ≠ out_c:** channel counts differ. A 1×1 conv (in_c → out_c, stride=1) projects \(x\) to the right number of channels so you can add.

If stride=1 *and* in_c=out_c, the identity is already the right shape, so shortcut can be “do nothing” (empty Sequential or identity).

**Takeaway:** Before adding, both branches must have identical shape. Shortcut = identity when possible; otherwise 1×1 conv (with same stride as main path if you downsample) + BN.

---

## 5. VGG-Style: Why This Structure?

**Design:** Several “blocks.” Each block = multiple (Conv 3×3 → BN → ReLU), then one MaxPool(2). So spatial size is constant inside the block, then you halve it at the end of the block. Channel count usually doubles per block (64 → 128 → 256 → 512).

**Why it works:** Simple, repetitive structure. Same padding keeps spatial size until pool; doubling channels when you halve resolution keeps compute roughly balanced. No skip connections — so depth is limited by gradient flow, but the pattern (conv stack → pool) is the same you’ll reuse in ResNet as “stem” or “blocks.”

**Common mistake:** Wrong channel progression (e.g. block2 in_channels ≠ block1 out_channels), or forgetting that after each pool the spatial size halves, so the *next* block’s conv operates on a smaller grid.

---

## 6. ResNet “Layer”: Why Group Blocks Into Layers?

**ResNet-18 structure:** stem (one conv+BN+ReLU+pool) → **layer1, layer2, layer3, layer4**. Each “layer” is a sequence of residual blocks. Often:
- layer1: same resolution (stride 1), e.g. 64 channels, 2 blocks.
- layer2: first block uses stride=2 (and 64→128), so resolution halves; then more blocks with stride 1.
- layer3, layer4: same idea — first block downsamples (stride 2, double channels), rest stay at same resolution.

**Why “make_layer”:** So you can say “build N blocks; the first one has stride=s and in_c→out_c, the rest have stride=1 and out_c→out_c.” That avoids repeating the same logic and keeps shapes consistent (first block may need a shortcut for stride/channels; later blocks often identity shortcut).

**Takeaway:** One function/module “make_layer(in_c, out_c, num_blocks, first_stride)” that builds num_blocks ResidualBlocks, with the first one possibly downsampling and changing channels.

---

## 7. Why Bottleneck Blocks? (ResNet-50+)

**Basic block (ResNet-18/34):** two 3×3 convs. **Bottleneck:** 1×1 (reduce channels) → 3×3 → 1×1 (expand channels).

**Why:** The 3×3 conv is expensive (many params and FLOPs). If you do it on *fewer* channels (after a 1×1 “bottleneck”), you save compute. So: 1×1 reduces C to C/4 (or similar), 3×3 does the spatial work on C/4, then 1×1 expands back to C. Same receptive field as a basic block, fewer parameters.

**Shortcut:** Same rule — if stride≠1 or in_c≠out_c, shortcut is 1×1 conv (+ BN) so shapes match for addition.

---

## 8. Inception-Style: Why Parallel Branches Then Concat?

**Idea:** Instead of choosing one kernel size (3×3 or 5×5), run **several branches in parallel** (e.g. 1×1, 3×3, 5×5, 3×3 maxpool) and **concatenate** their outputs along the channel dimension. The network can learn how much to use each scale.

**Shapes:** Each branch must output the *same spatial size* (use padding so H, W match). Channel counts can differ; concat gives sum of channels. So you need to know the total out_channels when you define the next layer.

**Takeaway:** Multiple conv/pool branches in parallel → concat on channel dim → one tensor (N, C1+C2+C3+C4, H, W). Good interview topic for “design alternatives.”

---

## 9. Common Bugs (So You Can Avoid Them)

| Bug | Why it happens | What to check |
|-----|----------------|----------------|
| **Shape mismatch when adding residual** | Shortcut and main path have different (C, H, W). | When stride≠1 or in_c≠out_c, shortcut must project. Assert shapes before `+`. |
| **Forgetting the final ReLU** | Residual block should output ReLU(main + shortcut). | After `out = main + shortcut(x)`, return ReLU(out). |
| **Wrong channel progression** | Block i output channels ≠ block i+1 input channels. | Write down (in_c, out_c) for each block and for the linear. |
| **Linear input size wrong** | After global pool you have (N, C); linear must have in_features=C. | If you use AdaptiveAvgPool2d(1), flatten gives (N, C); fc.in_features must be C. |
| **Stride in the wrong place** | Downsampling in the *second* conv of a block but not in shortcut. | Put stride in the *first* conv of the block (and in shortcut 1×1) so both paths see the same resolution change. |
| **BN in eval mode** | During validation, BN must use running stats. | Call `model.eval()` before validation. |

---

## 10. Checklist Before You Submit

- **Shapes:** After each block/layer, do the shapes (N, C, H, W) match what the next layer expects? After global pool, is the flattened size equal to the linear’s in_features?
- **Shortcut:** For every residual add, are both tensors the same shape?
- **Order:** Conv → BN → ReLU in the main path; residual = main + shortcut; then ReLU on the sum.
- **No answers in this doc:** Use this for *reasoning*. Implement the details in the challenge files yourself.

Good luck — implement from scratch using only these explanations and the assert shapes in the tests.
