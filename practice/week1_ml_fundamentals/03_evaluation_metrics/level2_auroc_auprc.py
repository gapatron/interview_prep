"""
EVALUATION METRICS - LEVEL 2: AUROC AND AUPRC
=============================================

Difficulty: Medium-Hard
Time to complete: 25-30 minutes
Topics: ROC curve, AUROC, Precision-Recall curve, AUPRC

BACKGROUND:
-----------
AUROC and AUPRC are CRITICAL metrics that Pinterest WILL ask about!

ROC CURVE:
- X-axis: False Positive Rate (FPR)
- Y-axis: True Positive Rate (TPR = Recall)
- Each point is a different threshold
- Area Under ROC = AUROC

PRECISION-RECALL CURVE:
- X-axis: Recall
- Y-axis: Precision
- Area Under PR Curve = AUPRC

WHEN TO USE WHICH:
- AUROC: Balanced classes, overall performance
- AUPRC: Imbalanced classes (when positive class is rare and important)

INTERVIEW TIP:
--------------
"When should you use AUROC vs AUPRC?"
ANSWER: AUPRC is better for imbalanced datasets because:
1. AUROC can be misleadingly high even with poor precision on rare class
2. AUPRC focuses on the positive class performance
3. In production systems (fraud, disease), the positive class matters most

INSTRUCTIONS:
-------------
1. Complete each TODO section
2. Run tests: pytest level2_auroc_auprc.py -v
3. Understand the geometric interpretation
"""

import numpy as np
from typing import List, Tuple


# =============================================================================
# EXERCISE 1: TPR and FPR at Different Thresholds
# =============================================================================
def calculate_tpr_fpr_at_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate TPR and FPR at each threshold.
    
    For each threshold t:
    - Classify as positive if score >= t
    - Calculate TPR = TP / (TP + FN)
    - Calculate FPR = FP / (FP + TN)
    
    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted probabilities/scores
        thresholds: Array of thresholds to evaluate
    
    Returns:
        Tuple of (tpr_array, fpr_array)
    
    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        >>> thresholds = np.array([0.3, 0.5, 0.7])
        >>> tpr, fpr = calculate_tpr_fpr_at_thresholds(y_true, y_scores, thresholds)
    
    INTERVIEW TIP: Explain that ROC curve is parametric:
    - Each point corresponds to one threshold
    - Moving right = lower threshold (more positive predictions)
    """
    # TODO: Implement this function (10-15 lines)
    # For each threshold:
    #   1. Generate predictions: y_pred = (y_scores >= threshold)
    #   2. Calculate TP, TN, FP, FN
    #   3. Calculate TPR = TP / (TP + FN), handle division by zero
    #   4. Calculate FPR = FP / (FP + TN), handle division by zero
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 2: ROC Curve
# =============================================================================
def roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
    
    Returns:
        Tuple of (fpr, tpr, thresholds)
        - fpr: False positive rates
        - tpr: True positive rates  
        - thresholds: Thresholds used
    
    Algorithm:
    1. Get unique sorted thresholds from scores (plus 0 and 1)
    2. For each threshold, calculate TPR and FPR
    3. Return sorted by FPR (for plotting)
    
    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        >>> fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    GEOMETRIC INTERPRETATION:
    - Perfect classifier: Goes up to (0, 1) then right to (1, 1)
    - Random classifier: Diagonal from (0, 0) to (1, 1)
    - Bad classifier: Below the diagonal
    """
    # TODO: Implement this function (10-15 lines)
    # Step 1: Get sorted unique thresholds (include min-1, max+1 for endpoints)
    # Step 2: Calculate TPR, FPR for each threshold
    # Step 3: Sort by FPR for proper curve ordering
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 3: AUROC Calculation
# =============================================================================
def auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Area Under the ROC Curve.
    
    Uses trapezoidal rule to approximate area.
    
    AUROC = Σ (FPR[i+1] - FPR[i]) * (TPR[i+1] + TPR[i]) / 2
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
    
    Returns:
        AUROC score between 0 and 1
    
    INTERPRETATION:
    - AUROC = 1.0: Perfect classifier
    - AUROC = 0.5: Random classifier (diagonal)
    - AUROC < 0.5: Worse than random (flip predictions!)
    
    PROBABILISTIC INTERPRETATION:
    AUROC = P(score(random positive) > score(random negative))
    
    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.2, 0.7, 0.8])  # Perfect separation
        >>> auroc(y_true, y_scores)
        1.0
    
    INTERVIEW TIP: Be ready to explain both geometric and probabilistic interpretations!
    """
    # TODO: Implement this function (8-12 lines)
    # Step 1: Get ROC curve points
    # Step 2: Sort by FPR
    # Step 3: Calculate area using trapezoidal rule
    
    pass  # Remove this and implement


def auroc_fast(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Fast AUROC calculation using the ranking method.
    
    Based on the probabilistic interpretation:
    AUROC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    
    This is O(n log n) due to sorting, vs O(n²) for the naive approach.
    
    INTERVIEW BONUS: This shows algorithmic thinking!
    """
    # TODO: Implement this function (8-10 lines)
    # Step 1: Get ranks of scores (handle ties with average rank)
    # Step 2: Sum ranks of positive samples
    # Step 3: Apply formula
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 4: Precision-Recall Curve
# =============================================================================
def precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
    
    Returns:
        Tuple of (precision, recall, thresholds)
    
    Note: Length of precision/recall is len(thresholds) + 1
    because we include the endpoint where recall=0, precision=1
    
    GEOMETRIC INTERPRETATION:
    - Perfect classifier: Horizontal line at precision=1
    - Good classifier: Stays high in precision as recall increases
    - Bad classifier: Precision drops quickly as recall increases
    
    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        >>> precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    """
    # TODO: Implement this function (15-20 lines)
    # Step 1: Sort scores descending (with corresponding true labels)
    # Step 2: For each threshold (sorted scores), calculate precision and recall
    # Step 3: Handle edge cases (no predictions, no positives)
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 5: AUPRC Calculation
# =============================================================================
def auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Area Under the Precision-Recall Curve.
    
    Uses trapezoidal approximation (or step function).
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
    
    Returns:
        AUPRC score between 0 and 1
    
    INTERPRETATION:
    - AUPRC = 1.0: Perfect classifier
    - Random baseline: AUPRC ≈ proportion of positives (NOT 0.5!)
    
    WHY AUPRC FOR IMBALANCED DATA:
    If 1% of data is positive:
    - Random AUROC = 0.5 (looks decent)
    - Random AUPRC = 0.01 (shows the problem!)
    
    Example:
        >>> y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 20% positive
        >>> # Random scores would give AUPRC ≈ 0.2, not 0.5!
    
    INTERVIEW TIP: This is THE metric for fraud detection, disease diagnosis, etc.
    """
    # TODO: Implement this function (8-12 lines)
    # Step 1: Get PR curve points
    # Step 2: Calculate area using appropriate method
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 6: Average Precision (Alternative to AUPRC)
# =============================================================================
def average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Average Precision.
    
    AP = Σ (R_n - R_{n-1}) * P_n
    
    where P_n is precision at the n-th threshold
    and R_n is recall at the n-th threshold.
    
    This is equivalent to:
    - Taking all thresholds where a positive sample was ranked
    - Averaging the precision at those points
    
    Relationship to AUPRC:
    - AP is a different way to approximate the area under PR curve
    - Uses step function instead of trapezoidal rule
    - Often preferred in practice (e.g., object detection)
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
    
    Returns:
        Average precision score
    """
    # TODO: Implement this function (10-15 lines)
    # Step 1: Sort by scores descending
    # Step 2: Calculate cumulative precision at each positive sample
    # Step 3: Average the precisions
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 7: Understanding AUROC vs AUPRC
# =============================================================================
def compare_auroc_auprc_demo() -> None:
    """
    Demonstrate when AUROC vs AUPRC differ.
    
    Key insight: With imbalanced data, AUROC can be misleadingly high
    while AUPRC reveals the true performance on the minority class.
    
    RUN THIS TO UNDERSTAND THE CONCEPT!
    """
    np.random.seed(42)
    
    # Scenario 1: Balanced classes (50% positive)
    n_balanced = 200
    y_true_balanced = np.array([0] * 100 + [1] * 100)
    y_scores_balanced = np.concatenate([
        np.random.beta(2, 5, 100),  # Negatives: scores skewed low
        np.random.beta(5, 2, 100)   # Positives: scores skewed high
    ])
    
    # Scenario 2: Imbalanced classes (5% positive)
    n_imbalanced = 200
    y_true_imbalanced = np.array([0] * 190 + [1] * 10)
    y_scores_imbalanced = np.concatenate([
        np.random.beta(2, 5, 190),  # Negatives
        np.random.beta(5, 2, 10)    # Positives (rare)
    ])
    
    print("AUROC vs AUPRC Comparison")
    print("=" * 60)
    
    # Calculate metrics
    auroc_bal = auroc(y_true_balanced, y_scores_balanced)
    auroc_imb = auroc(y_true_imbalanced, y_scores_imbalanced)
    
    auprc_bal = auprc(y_true_balanced, y_scores_balanced)
    auprc_imb = auprc(y_true_imbalanced, y_scores_imbalanced)
    
    print("\nBalanced Dataset (50% positive):")
    print(f"  AUROC: {auroc_bal:.3f}" if auroc_bal else "  AUROC: N/A")
    print(f"  AUPRC: {auprc_bal:.3f}" if auprc_bal else "  AUPRC: N/A")
    print(f"  Random baseline AUPRC: 0.500")
    
    print("\nImbalanced Dataset (5% positive):")
    print(f"  AUROC: {auroc_imb:.3f}" if auroc_imb else "  AUROC: N/A")
    print(f"  AUPRC: {auprc_imb:.3f}" if auprc_imb else "  AUPRC: N/A")
    print(f"  Random baseline AUPRC: 0.050")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("In the imbalanced case, AUROC might look similar to balanced,")
    print("but AUPRC reveals the true difficulty of the problem!")
    print("\nRECOMMENDATION:")
    print("- Use AUROC for balanced problems or overall ranking quality")
    print("- Use AUPRC for imbalanced problems where positives matter")


# =============================================================================
# TESTS
# =============================================================================
class TestAUROCAUPRC:
    """Tests for AUROC and AUPRC calculations."""
    
    def test_perfect_separation_auroc(self):
        """Test AUROC with perfect separation."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        
        result = auroc(y_true, y_scores)
        if result is not None:
            assert result == 1.0, f"Expected 1.0, got {result}"
    
    def test_random_auroc(self):
        """Test AUROC with random predictions (approximately 0.5)."""
        np.random.seed(42)
        y_true = np.array([0] * 500 + [1] * 500)
        y_scores = np.random.rand(1000)
        
        result = auroc(y_true, y_scores)
        if result is not None:
            assert 0.4 < result < 0.6, f"Expected ~0.5, got {result}"
    
    def test_inverse_auroc(self):
        """Test AUROC with inverse predictions (should be 0)."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])  # Higher scores for negatives
        
        result = auroc(y_true, y_scores)
        if result is not None:
            assert result == 0.0, f"Expected 0.0, got {result}"
    
    def test_auroc_fast_matches_regular(self):
        """Test that fast AUROC matches regular AUROC."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)
        
        regular = auroc(y_true, y_scores)
        fast = auroc_fast(y_true, y_scores)
        
        if regular is not None and fast is not None:
            assert abs(regular - fast) < 0.01
    
    def test_perfect_auprc(self):
        """Test AUPRC with perfect separation."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        
        result = auprc(y_true, y_scores)
        if result is not None:
            assert result == 1.0, f"Expected 1.0, got {result}"
    
    def test_auprc_baseline_imbalanced(self):
        """Test that AUPRC baseline is proportion of positives."""
        np.random.seed(42)
        # 10% positive
        y_true = np.array([0] * 900 + [1] * 100)
        y_scores = np.random.rand(1000)  # Random scores
        
        result = auprc(y_true, y_scores)
        # Random baseline should be close to 0.1 (proportion of positives)
        if result is not None:
            assert 0.05 < result < 0.2, f"Expected ~0.1, got {result}"
    
    def test_roc_curve_endpoints(self):
        """Test that ROC curve has correct endpoints."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        if fpr is not None:
            # Should include (0, 0) and (1, 1)
            assert 0 in fpr
            assert 1 in fpr
    
    def test_pr_curve_shape(self):
        """Test PR curve shape."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        if precision is not None:
            # Precision should be between 0 and 1
            assert all(0 <= p <= 1 for p in precision)
            # Recall should be between 0 and 1
            assert all(0 <= r <= 1 for r in recall)
    
    def test_average_precision_perfect(self):
        """Test average precision with perfect classifier."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        
        result = average_precision(y_true, y_scores)
        if result is not None:
            assert result == 1.0


if __name__ == "__main__":
    print("Testing AUROC and AUPRC...")
    print("Run 'pytest level2_auroc_auprc.py -v' for full tests")
    
    # Run comparison demo
    print("\n")
    compare_auroc_auprc_demo()
