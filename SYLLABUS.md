# Pinterest ML Internship Interview Preparation Syllabus

## Overview
This syllabus is designed to prepare you for Pinterest's Machine Learning Internship interview process, specifically focusing on:
1. **CodeSignal Assessment** (70 minutes)
2. **Technical Interview** (60 minutes via CoderPad)

---

## Interview Structure Analysis

### CodeSignal Assessment (70 minutes)
- **7 Questions**: Multiple-choice and short-answer on Machine Learning fundamentals
- **3 Coding Questions**:
  - Implement Naive Bayes from scratch
  - Implement Gradient Descent from scratch
  - LeetCode Medium difficulty problem

### Technical Interview (60 minutes)
- **ML Fundamentals Discussion** (~15-20 minutes)
- **Coding Question** (~30-40 minutes) - LeetCode Medium on CoderPad

---

## Core Themes Identified

### Theme 1: ML Algorithm Implementation
**Frequency**: Very High  
**Examples**: Naive Bayes, Gradient Descent  
**Why**: Tests understanding of ML fundamentals at implementation level

### Theme 2: ML Theory & Concepts
**Frequency**: Very High  
**Examples**: PCA vs LDA, Regularization, Evaluation Metrics  
**Why**: Assesses deep understanding of ML principles

### Theme 3: Data Structures & Algorithms
**Frequency**: High  
**Examples**: Count and Say, Reconstruct Itinerary, Graph problems  
**Why**: Core programming competency required for ML engineering

### Theme 4: ML-Specific Coding Patterns
**Frequency**: Medium-High  
**Examples**: String manipulation, data preprocessing, feature engineering  
**Why**: Practical ML engineering skills

### Theme 5: Evaluation & Metrics
**Frequency**: High  
**Examples**: AUROC, AUPRC, Precision/Recall  
**Why**: Critical for production ML systems

---

## Detailed Syllabus

## Week 1: ML Fundamentals & Theory

### Day 1-2: Core ML Algorithms
**Topics:**
- Naive Bayes (Multinomial, Gaussian, Bernoulli)
  - Mathematical foundation
  - Implementation from scratch
  - Use cases and assumptions
- Gradient Descent
  - Batch, Mini-batch, Stochastic variants
  - Learning rate selection
  - Implementation with and without frameworks
  - Convergence criteria

**Practice:**
- [ ] Implement Naive Bayes classifier from scratch
- [ ] Implement Gradient Descent optimizer from scratch
- [ ] Code both with and without NumPy (to show understanding)

**Resources:**
- Review mathematical foundations
- Practice explaining trade-offs

### Day 3-4: Dimensionality Reduction
**Topics:**
- PCA (Principal Component Analysis)
  - Mathematical derivation
  - When to use
  - Limitations
- LDA (Linear Discriminant Analysis)
  - Mathematical foundation
  - When to use vs PCA
  - Supervised vs Unsupervised distinction

**Key Comparison Points:**
- PCA: Unsupervised, maximizes variance
- LDA: Supervised, maximizes class separation
- When to use each

**Practice:**
- [ ] Explain PCA vs LDA with examples
- [ ] Implement PCA from scratch (optional but recommended)

### Day 5-7: Evaluation Metrics & Model Assessment
**Topics:**
- Classification Metrics:
  - Accuracy, Precision, Recall, F1-Score
  - **AUROC (Area Under ROC Curve)**
  - **AUPRC (Area Under Precision-Recall Curve)**
  - Confusion Matrix
  - When to use each metric
- Regression Metrics:
  - MSE, RMSE, MAE, R²
- Cross-validation strategies

**Practice:**
- [ ] Implement AUROC calculation from scratch
- [ ] Implement AUPRC calculation from scratch
- [ ] Explain when to use AUROC vs AUPRC (imbalanced datasets)
- [ ] Practice explaining trade-offs between metrics

---

## Week 2: Regularization & Model Optimization

### Day 1-3: Regularization Techniques
**Topics:**
- L1 Regularization (Lasso)
  - Mathematical formulation
  - Feature selection properties
- L2 Regularization (Ridge)
  - Mathematical formulation
  - Shrinking coefficients
- Elastic Net
- Dropout (for neural networks)
- Early Stopping
- When to use each technique

**Practice:**
- [ ] Implement L1 and L2 regularization in loss functions
- [ ] Explain how each affects model complexity
- [ ] Discuss bias-variance trade-off

### Day 4-5: Activation Functions & Neural Network Basics
**Topics:**
- **Sigmoid Function**
  - Mathematical definition
  - Properties (range, derivative, vanishing gradient)
  - Use cases
- Other activation functions:
  - ReLU, Tanh, Softmax
  - When to use each

**Practice:**
- [ ] Implement sigmoid and its derivative
- [ ] Explain vanishing gradient problem
- [ ] Compare activation functions

### Day 6-7: Ensemble Methods
**Topics:**
- Bagging (Random Forests)
- Boosting (AdaBoost, Gradient Boosting)
- Stacking
- When to use ensemble methods

**Practice:**
- [ ] Explain bagging vs boosting
- [ ] Discuss bias-variance trade-off in ensembles

---

## Week 3: Algorithm Implementation & Coding Practice

### Day 1-2: LeetCode Medium - String & Array Problems
**Focus Problems:**
- **Count and Say** (LeetCode)
  - Pattern recognition
  - String manipulation
  - Recursive/iterative approaches
- String manipulation patterns
- Array manipulation

**Practice:**
- [ ] Solve Count and Say (iterative and recursive)
- [ ] Practice similar string problems
- [ ] Time complexity analysis

### Day 3-4: Graph Algorithms
**Focus Problems:**
- **Reconstruct Itinerary** (LeetCode)
  - Graph traversal
  - DFS/BFS
  - Eulerian path concepts
- Graph representation
- Shortest path algorithms

**Practice:**
- [ ] Solve Reconstruct Itinerary
- [ ] Practice DFS/BFS implementations
- [ ] Understand graph data structures

### Day 5-7: Pinterest-Specific Problems
**Problems to Practice:**
- Number of Islands (mentioned in PDF)
- Task Scheduler (mentioned in PDF)
- Neardups (mentioned in PDF)
- Find the Celebrity (mentioned in PDF)
- Merge k Sorted Lists (mentioned in PDF)
- Shortest Path problems (from LeetCode discussions)
- Count Pins problems (from LeetCode discussions)

**Practice:**
- [ ] Solve all Pinterest retired problems
- [ ] Review LeetCode discussion posts for Pinterest
- [ ] Practice explaining solutions clearly

---

## Week 4: Integration & Mock Interviews

### Day 1-2: ML Coding Patterns
**Topics:**
- Feature engineering in code
- Data preprocessing pipelines
- Handling missing data
- Normalization/standardization
- Working with imbalanced datasets

**Practice:**
- [ ] Write clean preprocessing functions
- [ ] Implement feature engineering patterns
- [ ] Practice explaining design choices

### Day 3-4: CodeSignal Practice
**Focus:**
- Time management (70 minutes total)
- Multiple-choice strategy
- Coding under time pressure
- Testing your code

**Practice:**
- [ ] Take CodeSignal practice tests
- [ ] Time yourself on ML coding problems
- [ ] Practice implementing algorithms quickly but correctly

### Day 5-7: Mock Interviews & Final Review
**Practice:**
- [ ] Mock interview: ML fundamentals + coding
- [ ] Practice on CoderPad
- [ ] Time yourself: 30-40 min for coding
- [ ] Practice thinking aloud
- [ ] Review all topics

**Mock Interview Format:**
1. ML theory questions (15 min)
2. Coding problem (35 min)
   - Ask clarifying questions
   - Think aloud
   - Write clean code
   - Test edge cases

---

## Key Problem List

### Must-Solve LeetCode Problems
1. **Count and Say** - https://leetcode.com/problems/count-and-say/
2. **Reconstruct Itinerary** - https://leetcode.com/problems/reconstruct-itinerary/
3. **Number of Islands** - (Pinterest retired)
4. **Task Scheduler** - (Pinterest retired)
5. **Merge k Sorted Lists** - (Pinterest retired)
6. **Find the Celebrity** - (Pinterest retired)
7. **Neardups** - (Pinterest retired)

### ML Implementation Problems
1. **Naive Bayes** - Implement from scratch
2. **Gradient Descent** - Implement from scratch
3. **PCA** - Implement from scratch (optional but recommended)
4. **AUROC/AUPRC** - Calculate from scratch

---

## Study Resources

### Official Pinterest Resources
- [ ] Pinterest ML Interview FAQs
- [ ] CodeSignal practice tests
- [ ] CoderPad familiarization
- [ ] Pinterest Engineering Blog
- [ ] Pinterview Pro: Machine Learning article

### External Resources
- [ ] LeetCode Premium (for Pinterest-specific problems)
- [ ] ML interview preparation books
- [ ] Practice explaining ML concepts clearly

---

## Interview Day Checklist

### Before CodeSignal Assessment
- [ ] Review ML fundamentals (quick pass)
- [ ] Practice implementing Naive Bayes (5 min warm-up)
- [ ] Practice implementing Gradient Descent (5 min warm-up)
- [ ] Ensure good internet connection
- [ ] Quiet environment, no distractions

### Before Technical Interview
- [ ] Review ML theory topics
- [ ] Practice on CoderPad interface
- [ ] Warm up with a LeetCode Medium problem
- [ ] Prepare questions about Pinterest/role
- [ ] Test audio/video for Google Meet

### During Interview
- [ ] **Ask clarifying questions** before coding
- [ ] **Think aloud** - explain your approach
- [ ] **Restate the problem** to confirm understanding
- [ ] Use **descriptive variable names**
- [ ] **Test your code** with examples
- [ ] **Handle edge cases**
- [ ] **Accept feedback** and improve code

---

## Success Criteria

### CodeSignal Assessment
- ✅ Complete all 10 questions
- ✅ Implement Naive Bayes correctly
- ✅ Implement Gradient Descent correctly
- ✅ Solve LeetCode Medium problem
- ✅ Show understanding in ML theory questions

### Technical Interview
- ✅ Answer ML fundamentals confidently
- ✅ Solve coding problem in 30-40 minutes
- ✅ Write clean, efficient code
- ✅ Communicate thought process clearly
- ✅ Ask good clarifying questions
- ✅ Handle feedback gracefully

---

## Common Pitfalls to Avoid

1. **Not asking clarifying questions** - Always clarify requirements first
2. **Jumping to code too quickly** - Think through approach first
3. **Not testing code** - Always test with examples
4. **Poor variable naming** - Use descriptive names
5. **Not explaining thought process** - Think aloud
6. **Ignoring edge cases** - Consider boundary conditions
7. **Not accepting feedback** - Show collaborative spirit

---

## Notes

- Focus on **understanding** not memorization
- Practice **explaining** concepts clearly
- **Time management** is critical
- **Clean code** matters as much as correctness
- **Communication** is part of the assessment

---

## Tracking Progress

Use this checklist to track your preparation:

### Week 1 Progress
- [ ] Naive Bayes implementation mastered
- [ ] Gradient Descent implementation mastered
- [ ] PCA vs LDA understood
- [ ] AUROC/AUPRC calculations mastered

### Week 2 Progress
- [ ] Regularization techniques understood
- [ ] Sigmoid function and derivatives mastered
- [ ] Ensemble methods reviewed

### Week 3 Progress
- [ ] All Pinterest problems solved
- [ ] Graph algorithms comfortable
- [ ] String manipulation patterns mastered

### Week 4 Progress
- [ ] CodeSignal practice completed
- [ ] CoderPad familiarized
- [ ] Mock interviews completed
- [ ] Ready for interview!

---

**Last Updated**: Based on 2025 interview experiences and official Pinterest documentation
