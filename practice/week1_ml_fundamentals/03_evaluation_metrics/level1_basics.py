"""
EVALUATION METRICS - LEVEL 1: BASICS
====================================

Difficulty: Easy
Time to complete: 15-20 minutes
Topics: Confusion matrix, Accuracy, Precision, Recall, F1

BACKGROUND:
-----------
Evaluation metrics tell us HOW WELL our model performs.
Different metrics matter for different problems!

CONFUSION MATRIX (for binary classification):
                    Predicted
                    Pos    Neg
        Actual Pos   TP     FN
               Neg   FP     TN

- TP (True Positive): Correctly predicted positive
- TN (True Negative): Correctly predicted negative
- FP (False Positive): Incorrectly predicted positive (Type I error)
- FN (False Negative): Incorrectly predicted negative (Type II error)

INTERVIEW TIP:
--------------
Pinterest WILL ask about metrics. Be ready to explain:
1. When to use which metric
2. Trade-offs between precision and recall
3. Why accuracy can be misleading

INSTRUCTIONS:
-------------
1. Complete each TODO section
2. Run tests: pytest level1_basics.py -v
3. Think about real-world examples for each metric
"""

import numpy as np
from typing import Tuple, Dict


# =============================================================================
# EXERCISE 1: Confusion Matrix
# =============================================================================
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix for binary classification.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
    
    Returns:
        2x2 confusion matrix:
        [[TN, FP],
         [FN, TP]]
    
    Example:
        >>> y_true = np.array([1, 1, 0, 0, 1])
        >>> y_pred = np.array([1, 0, 0, 1, 1])
        >>> confusion_matrix(y_true, y_pred)
        array([[1, 1],
               [1, 2]])
        # TN=1, FP=1, FN=1, TP=2
    
    INTERVIEW TIP: Draw this matrix and walk through examples!
    """
    # TODO: Implement this function (6-10 lines)
    # Step 1: Initialize 2x2 matrix of zeros
    # Step 2: Count TP: sum where y_true=1 AND y_pred=1
    # Step 3: Count TN: sum where y_true=0 AND y_pred=0
    # Step 4: Count FP: sum where y_true=0 AND y_pred=1
    # Step 5: Count FN: sum where y_true=1 AND y_pred=0
    # Step 6: Return matrix [[TN, FP], [FN, TP]]
    
    pass  # Remove this and implement


def get_tp_tn_fp_fn(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get TP, TN, FP, FN counts directly.
    
    Returns:
        Tuple of (TP, TN, FP, FN)
    """
    # TODO: Implement this function (4-6 lines)
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 2: Basic Metrics
# =============================================================================
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy: proportion of correct predictions.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy score between 0 and 1
    
    Example:
        >>> y_true = np.array([1, 1, 0, 0])
        >>> y_pred = np.array([1, 0, 0, 0])
        >>> accuracy(y_true, y_pred)
        0.75
    
    WHEN TO USE:
    - When classes are balanced
    - When all errors are equally bad
    
    WHEN NOT TO USE:
    - Imbalanced classes (99% negative -> always predicting negative gives 99% accuracy!)
    """
    # TODO: Implement this function (1-3 lines)
    
    pass  # Remove this and implement


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision: proportion of positive predictions that are correct.
    
    Precision = TP / (TP + FP)
    
    "Of all the items we predicted as positive, how many are actually positive?"
    
    Example:
        >>> y_true = np.array([1, 1, 0, 0])
        >>> y_pred = np.array([1, 1, 1, 0])  # 2 correct pos, 1 false pos
        >>> precision(y_true, y_pred)
        0.666...  # 2 / 3
    
    WHEN TO USE:
    - When false positives are costly
    - Example: Spam filter (don't want to mark good email as spam)
    
    HIGH PRECISION means: Few false positives (when we predict positive, we're usually right)
    """
    # TODO: Implement this function (3-5 lines)
    # Handle edge case: what if TP + FP = 0?
    
    pass  # Remove this and implement


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall (sensitivity, TPR): proportion of actual positives found.
    
    Recall = TP / (TP + FN)
    
    "Of all actual positive items, how many did we find?"
    
    Example:
        >>> y_true = np.array([1, 1, 0, 0])
        >>> y_pred = np.array([1, 0, 0, 0])  # Found 1 of 2 positives
        >>> recall(y_true, y_pred)
        0.5
    
    WHEN TO USE:
    - When false negatives are costly
    - Example: Disease diagnosis (don't want to miss sick patients)
    
    HIGH RECALL means: Few false negatives (we find most of the positives)
    """
    # TODO: Implement this function (3-5 lines)
    # Handle edge case: what if TP + FN = 0?
    
    pass  # Remove this and implement


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 score: harmonic mean of precision and recall.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    The harmonic mean penalizes extreme values. If either precision or
    recall is low, F1 will be low.
    
    Example:
        >>> y_true = np.array([1, 1, 0, 0])
        >>> y_pred = np.array([1, 1, 1, 0])
        >>> # Precision = 2/3, Recall = 2/2 = 1
        >>> # F1 = 2 * (2/3 * 1) / (2/3 + 1) = 0.8
        >>> f1_score(y_true, y_pred)
        0.8
    
    WHEN TO USE:
    - When you need a single metric balancing precision and recall
    - When classes are imbalanced
    
    INTERVIEW TIP: Why harmonic mean instead of arithmetic mean?
    - Harmonic mean penalizes extreme imbalances
    - If precision=1.0 and recall=0.1: arithmetic=0.55, harmonic=0.18
    """
    # TODO: Implement this function (4-6 lines)
    # Handle edge case: what if precision + recall = 0?
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 3: Specificity and False Positive Rate
# =============================================================================
def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity (TNR): proportion of actual negatives correctly identified.
    
    Specificity = TN / (TN + FP)
    
    "Of all actual negative items, how many did we correctly identify as negative?"
    
    This is like recall, but for the negative class!
    
    Example:
        >>> y_true = np.array([1, 1, 0, 0])
        >>> y_pred = np.array([1, 1, 1, 0])  # 1 TN, 1 FP
        >>> specificity(y_true, y_pred)
        0.5
    """
    # TODO: Implement this function (3-5 lines)
    
    pass  # Remove this and implement


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate False Positive Rate (FPR).
    
    FPR = FP / (FP + TN) = 1 - Specificity
    
    "Of all actual negative items, how many did we incorrectly mark as positive?"
    
    This is used in ROC curves!
    """
    # TODO: Implement this function (1-3 lines)
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 4: All Metrics Together
# =============================================================================
def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Generate a complete classification report.
    
    Returns:
        Dictionary with all metrics:
        {
            'accuracy': ...,
            'precision': ...,
            'recall': ...,
            'f1': ...,
            'specificity': ...,
            'fpr': ...
        }
    
    Example:
        >>> y_true = np.array([1, 1, 1, 0, 0, 0])
        >>> y_pred = np.array([1, 1, 0, 0, 0, 1])
        >>> report = classification_report(y_true, y_pred)
        >>> # Should contain all metrics
    """
    # TODO: Implement this function (use your other functions)
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 5: Understanding the Trade-offs
# =============================================================================
def precision_recall_tradeoff_demo() -> None:
    """
    Demonstrate precision-recall trade-off.
    
    When you adjust the classification threshold:
    - Higher threshold: Higher precision, lower recall
    - Lower threshold: Lower precision, higher recall
    
    This is important for Pinterest interviews!
    
    RUN THIS TO UNDERSTAND THE CONCEPT.
    """
    np.random.seed(42)
    
    # Simulate model probabilities
    # Class 0: probabilities around 0.3
    # Class 1: probabilities around 0.7
    probs_class0 = np.random.beta(3, 7, 100)  # Mean ~0.3
    probs_class1 = np.random.beta(7, 3, 100)  # Mean ~0.7
    
    y_true = np.array([0] * 100 + [1] * 100)
    y_probs = np.concatenate([probs_class0, probs_class1])
    
    print("Precision-Recall Trade-off Demo")
    print("=" * 50)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 50)
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred = (y_probs >= threshold).astype(int)
        
        prec = precision(y_true, y_pred) if precision(y_true, y_pred) else 0
        rec = recall(y_true, y_pred) if recall(y_true, y_pred) else 0
        f1 = f1_score(y_true, y_pred) if f1_score(y_true, y_pred) else 0
        
        print(f"{threshold:<12.1f} {prec:<12.3f} {rec:<12.3f} {f1:<12.3f}")
    
    print("\nObservations:")
    print("- As threshold increases, precision increases but recall decreases")
    print("- F1 score helps find a balance")
    print("- Choose threshold based on business requirements!")


# =============================================================================
# TESTS
# =============================================================================
class TestEvaluationMetricsBasics:
    """Tests for basic evaluation metrics."""
    
    def test_confusion_matrix_perfect(self):
        """Test confusion matrix with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        
        # [[TN, FP], [FN, TP]] = [[2, 0], [0, 2]]
        np.testing.assert_array_equal(cm, [[2, 0], [0, 2]])
    
    def test_confusion_matrix_all_wrong(self):
        """Test confusion matrix with all wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        cm = confusion_matrix(y_true, y_pred)
        
        # [[TN, FP], [FN, TP]] = [[0, 2], [2, 0]]
        np.testing.assert_array_equal(cm, [[0, 2], [2, 0]])
    
    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert accuracy(y_true, y_pred) == 1.0
    
    def test_accuracy_half(self):
        """Test accuracy at 50%."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert accuracy(y_true, y_pred) == 0.5
    
    def test_precision_perfect(self):
        """Test precision with no false positives."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert precision(y_true, y_pred) == 1.0
    
    def test_precision_with_fp(self):
        """Test precision with false positives."""
        y_true = np.array([1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0])  # 1 TP, 1 FP
        assert precision(y_true, y_pred) == 0.5
    
    def test_recall_perfect(self):
        """Test recall with no false negatives."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert recall(y_true, y_pred) == 1.0
    
    def test_recall_with_fn(self):
        """Test recall with false negatives."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])  # 1 TP, 1 FN
        assert recall(y_true, y_pred) == 0.5
    
    def test_f1_perfect(self):
        """Test F1 with perfect predictions."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert f1_score(y_true, y_pred) == 1.0
    
    def test_f1_balanced(self):
        """Test F1 calculation."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0])  # P=2/3, R=1
        f1 = f1_score(y_true, y_pred)
        expected = 2 * (2/3 * 1) / (2/3 + 1)
        assert abs(f1 - expected) < 1e-6
    
    def test_specificity(self):
        """Test specificity calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])  # TN=1, FP=1
        assert specificity(y_true, y_pred) == 0.5
    
    def test_fpr_complement_of_specificity(self):
        """Test that FPR = 1 - Specificity."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        
        spec = specificity(y_true, y_pred)
        fpr = false_positive_rate(y_true, y_pred)
        
        assert abs(fpr - (1 - spec)) < 1e-6
    
    def test_classification_report_keys(self):
        """Test that classification report has all keys."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        
        report = classification_report(y_true, y_pred)
        
        if report:
            expected_keys = {'accuracy', 'precision', 'recall', 'f1', 'specificity', 'fpr'}
            assert set(report.keys()) == expected_keys


if __name__ == "__main__":
    print("Testing Evaluation Metrics Basics...")
    print("Run 'pytest level1_basics.py -v' for full tests")
    
    # Run demo
    print("\n")
    precision_recall_tradeoff_demo()
