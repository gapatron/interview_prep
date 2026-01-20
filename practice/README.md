# Pinterest ML Internship Interview - Practice Codebase

## Overview

This practice codebase is designed to help you prepare for Pinterest's Machine Learning internship interview. It covers all topics from the official interview prep document and interview experiences.

## Structure

```
practice/
├── week1_ml_fundamentals/          # Core ML algorithms from scratch
│   ├── 01_naive_bayes/             # Naive Bayes implementation
│   │   ├── level1_basics.py        # Easy: priors, likelihoods
│   │   ├── level2_gaussian.py      # Medium: Gaussian NB
│   │   └── level3_full_implementation.py  # Hard: Complete classifier
│   ├── 02_gradient_descent/        # Gradient descent variants
│   │   ├── level1_basics.py        # Easy: basic GD, linear regression
│   │   ├── level2_variants.py      # Medium: SGD, mini-batch, momentum
│   │   └── level3_advanced.py      # Hard: logistic regression, regularization
│   └── 03_evaluation_metrics/      # AUROC, AUPRC, and more
│       ├── level1_basics.py        # Easy: accuracy, precision, recall, F1
│       └── level2_auroc_auprc.py   # Medium-Hard: ROC/PR curves
│
├── week2_ml_theory/                # ML theory implementations
│   ├── 01_regularization/          # L1, L2, Elastic Net
│   ├── 02_pca_lda/                 # Dimensionality reduction
│   └── 03_activation_functions/    # Sigmoid, ReLU, etc.
│
├── week3_algorithms/               # LeetCode-style problems
│   ├── 01_string_problems/
│   │   └── count_and_say.py        # Pinterest confirmed problem
│   ├── 02_graph_problems/
│   │   ├── reconstruct_itinerary.py  # Pinterest confirmed problem
│   │   └── number_of_islands.py    # Pinterest retired problem
│   └── 03_pinterest_problems/
│       └── merge_k_sorted_lists.py # Pinterest retired problem
│
├── week4_integration/              # Mock interview problems
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest -v

# Run tests for specific module
pytest week1_ml_fundamentals/01_naive_bayes/ -v

# Run with coverage
pytest --cov=. --cov-report=html
```

## How to Use

### 1. Incremental Difficulty

Each topic has 2-3 levels:
- **Level 1 (Easy)**: Basic concepts, 15-20 minutes
- **Level 2 (Medium)**: Full implementation, 20-25 minutes  
- **Level 3 (Hard)**: Interview-ready, 25-30 minutes

Start with Level 1 and progress to Level 3.

### 2. Fill in the TODOs

Each file has:
- **Background**: Theory you need to know
- **Interview Tips**: What to say in interviews
- **TODO sections**: Code for you to implement
- **Tests**: Verify your implementation

Example workflow:
```python
# 1. Read the docstring and understand the problem
# 2. Implement the TODO section
# 3. Run pytest to verify
# 4. Check the "Interview Tips" for what to mention
```

### 3. Time Yourself

Interview-ready sections have time targets:
- **15 minutes**: Basic problems
- **20 minutes**: Medium problems
- **25-30 minutes**: Hard problems

Practice under time pressure!

### 4. Practice Communication

As you code:
1. **Think aloud**: Explain your approach
2. **Ask clarifying questions**: What are the constraints?
3. **Test your code**: Walk through examples
4. **Discuss complexity**: Time and space

## Topics Covered

### ML Fundamentals (Week 1-2)

| Topic | Importance | Files |
|-------|------------|-------|
| Naive Bayes | 🔴 Critical | `01_naive_bayes/` |
| Gradient Descent | 🔴 Critical | `02_gradient_descent/` |
| AUROC/AUPRC | 🔴 Critical | `03_evaluation_metrics/` |
| Regularization | 🟠 High | `01_regularization/` |
| PCA vs LDA | 🟠 High | `02_pca_lda/` |
| Sigmoid Function | 🟠 High | `03_activation_functions/` |

### Algorithm Problems (Week 3)

| Problem | Difficulty | Source | Files |
|---------|------------|--------|-------|
| Count and Say | Medium | LeetCode #38 | `01_string_problems/` |
| Reconstruct Itinerary | Hard | LeetCode #332 | `02_graph_problems/` |
| Number of Islands | Medium | LeetCode #200 | `02_graph_problems/` |
| Merge K Sorted Lists | Hard | LeetCode #23 | `03_pinterest_problems/` |

## Testing Your Solutions

```bash
# Test a specific file
pytest week1_ml_fundamentals/01_naive_bayes/level1_basics.py -v

# Test all naive bayes
pytest week1_ml_fundamentals/01_naive_bayes/ -v

# Test with print statements visible
pytest -v -s

# Test only functions that match a pattern
pytest -v -k "test_gaussian"
```

## Interview Checklist

Before your interview, ensure you can:

### ML Algorithms
- [ ] Implement Naive Bayes from scratch (15 min)
- [ ] Implement Gradient Descent from scratch (15 min)
- [ ] Explain and calculate AUROC/AUPRC
- [ ] Explain regularization (L1 vs L2)
- [ ] Explain PCA vs LDA trade-offs

### Algorithm Problems
- [ ] Solve Count and Say (15 min)
- [ ] Solve Number of Islands (15 min)
- [ ] Solve Merge K Sorted Lists (20 min)
- [ ] Solve Reconstruct Itinerary (25 min)

### Communication
- [ ] Think aloud while coding
- [ ] Ask clarifying questions
- [ ] Explain time/space complexity
- [ ] Test with examples

## Tips for Success

### 1. Understand Before Memorizing
Don't just memorize code. Understand **why** each step is needed.

### 2. Practice Communication
In interviews, HOW you solve matters as much as IF you solve.

### 3. Handle Edge Cases
Always consider: empty input, single element, duplicates, etc.

### 4. Know the Trade-offs
Be ready to discuss: "Why this approach vs another?"

### 5. Time Management
- 5 min: Understand problem, ask questions
- 20-25 min: Implement solution
- 5 min: Test and optimize

## Common Interview Questions

### Naive Bayes
- "Implement Naive Bayes for continuous features"
- "What is the naive assumption?"
- "How do you handle unseen features?"

### Gradient Descent
- "Implement gradient descent from scratch"
- "What are SGD, mini-batch, batch GD trade-offs?"
- "How do you choose learning rate?"

### Evaluation Metrics
- "When should you use AUROC vs AUPRC?"
- "Explain precision-recall trade-off"
- "What metric for imbalanced data?"

### Regularization
- "What's the difference between L1 and L2?"
- "When would you use each?"
- "How does regularization prevent overfitting?"

## Resources

- [Pinterest Engineering Blog](https://medium.com/pinterest-engineering)
- [Pinterest Interview FAQ](https://recruiting.pinteresttalent.com/portal/page/ml-faq)
- [CoderPad Practice](https://coderpad.io/)
- [LeetCode](https://leetcode.com/)

---

Good luck with your interview! 🍀
