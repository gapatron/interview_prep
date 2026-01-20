# Quick Reference: ML Interview Topics

## ML Algorithm Implementations

### Naive Bayes
```python
# Key components to implement:
# 1. Calculate prior probabilities P(class)
# 2. Calculate likelihood P(feature|class)
# 3. Apply Bayes' theorem: P(class|features) ∝ P(class) * ∏P(feature|class)
# 4. Handle smoothing (Laplace smoothing) for unseen features
```

**Key Points:**
- Assumes feature independence (naive assumption)
- Works well with high-dimensional data
- Fast training and prediction
- Variants: Multinomial, Gaussian, Bernoulli

### Gradient Descent
```python
# Basic structure:
# 1. Initialize weights
# 2. For each iteration:
#    - Calculate gradient of loss function
#    - Update weights: w = w - learning_rate * gradient
#    - Check convergence
```

**Key Points:**
- Learning rate is critical (too high: overshoot, too low: slow)
- Variants: Batch (all data), Mini-batch (subset), Stochastic (one sample)
- Convergence: When gradient ≈ 0 or loss stops decreasing

---

## ML Theory Quick Facts

### PCA vs LDA

| Aspect | PCA | LDA |
|--------|-----|-----|
| **Type** | Unsupervised | Supervised |
| **Goal** | Maximize variance | Maximize class separation |
| **Uses labels?** | No | Yes |
| **Best for** | Dimensionality reduction | Classification with dimensionality reduction |
| **Assumptions** | Linear relationships | Normal distribution, equal covariance |

**When to use:**
- **PCA**: When you want to reduce dimensions without class information
- **LDA**: When you have labeled data and want to improve classification

### Sigmoid Function

**Definition**: σ(x) = 1 / (1 + e^(-x))

**Properties:**
- Range: (0, 1)
- S-shaped curve
- Derivative: σ'(x) = σ(x)(1 - σ(x))
- Used in: Logistic regression, neural networks (output layer for binary classification)

**Vanishing Gradient Problem:**
- Derivative is small when |x| is large
- Can cause slow learning in deep networks

### Regularization Techniques

**L1 Regularization (Lasso):**
- Adds λ * |w| to loss function
- Can set weights to exactly 0 (feature selection)
- Produces sparse models
- Good for feature selection

**L2 Regularization (Ridge):**
- Adds λ * w² to loss function
- Shrinks weights but doesn't eliminate
- Produces smooth models
- Good for preventing overfitting

**Elastic Net:**
- Combines L1 and L2: λ₁|w| + λ₂w²
- Gets benefits of both

**When to use:**
- **L1**: When you suspect many irrelevant features
- **L2**: When you want to prevent overfitting without feature selection
- **Elastic Net**: When you want both benefits

### Evaluation Metrics

**AUROC (Area Under ROC Curve):**
- Range: 0 to 1 (1 = perfect, 0.5 = random)
- Measures ability to distinguish between classes
- **Good for**: Balanced datasets
- **Interpretation**: Probability that model ranks random positive higher than random negative

**AUPRC (Area Under Precision-Recall Curve):**
- Range: 0 to 1 (1 = perfect)
- Focuses on positive class performance
- **Good for**: Imbalanced datasets
- **Better than AUROC when**: Classes are highly imbalanced

**When to use:**
- **AUROC**: Balanced classes, overall performance important
- **AUPRC**: Imbalanced classes, positive class is critical

**Other Metrics:**
- **Precision**: TP / (TP + FP) - Of predicted positives, how many are correct?
- **Recall**: TP / (TP + FN) - Of actual positives, how many did we find?
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean

---

## Coding Patterns

### String Manipulation (Count and Say Pattern)
```python
# Pattern: Build result iteratively
def countAndSay(n):
    result = "1"
    for _ in range(n - 1):
        result = next_sequence(result)
    return result

def next_sequence(s):
    # Count consecutive characters
    # Build new string: "count" + "digit"
```

### Graph Problems (Reconstruct Itinerary Pattern)
```python
# Pattern: DFS with backtracking
# Key: Use sorted order, visit edges once
def findItinerary(tickets):
    graph = build_graph(tickets)
    result = []
    dfs("JFK", graph, result)
    return result[::-1]  # Reverse because we build backwards
```

---

## Interview Tips

### Before Coding
1. ✅ **Ask clarifying questions**
   - Input format?
   - Edge cases?
   - Constraints?
   - Expected output format?

2. ✅ **Restate the problem**
   - Confirm your understanding
   - Show you're thinking

3. ✅ **Think aloud**
   - Explain your approach
   - Discuss trade-offs
   - Show your reasoning

### While Coding
1. ✅ **Use descriptive names**
   - `num_islands` not `n`
   - `visited_nodes` not `v`

2. ✅ **Write clean code**
   - Proper indentation
   - Comments for complex logic
   - Modular functions

3. ✅ **Test as you go**
   - Test with examples
   - Handle edge cases
   - Check boundary conditions

### After Coding
1. ✅ **Walk through example**
   - Trace through your code
   - Verify correctness

2. ✅ **Discuss complexity**
   - Time complexity
   - Space complexity
   - Optimizations possible

3. ✅ **Accept feedback**
   - Listen to suggestions
   - Improve your code
   - Show collaboration

---

## Common Pitfalls

❌ **Jumping to code too quickly**
- Always clarify first

❌ **Not testing code**
- Always test with examples

❌ **Poor variable names**
- Use descriptive names

❌ **Not explaining thought process**
- Think aloud throughout

❌ **Ignoring edge cases**
- Consider empty inputs, single elements, etc.

❌ **Not accepting feedback**
- Show you can collaborate

---

## Time Management

### CodeSignal (70 minutes)
- **ML Theory (7 questions)**: ~25-30 minutes
- **Naive Bayes**: ~15 minutes
- **Gradient Descent**: ~15 minutes
- **LeetCode Medium**: ~15-20 minutes
- **Buffer**: ~5-10 minutes

### Technical Interview (60 minutes)
- **ML Discussion**: ~15-20 minutes
- **Coding Problem**: ~30-40 minutes
- **Questions/Clarifications**: ~5-10 minutes

---

## Key Formulas

### Naive Bayes
P(class|features) = P(class) * ∏P(feature|class) / P(features)

### Gradient Descent Update
w_new = w_old - learning_rate * ∇loss(w_old)

### Sigmoid
σ(x) = 1 / (1 + e^(-x))

### Precision
Precision = TP / (TP + FP)

### Recall
Recall = TP / (TP + FN)

### F1-Score
F1 = 2 * (Precision * Recall) / (Precision + Recall)

---

**Remember**: Understanding > Memorization. Be able to explain WHY, not just WHAT.
