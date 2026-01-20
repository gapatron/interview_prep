"""
NAIVE BAYES - LEVEL 1: BASICS
=============================

Difficulty: Easy
Time to complete: 15-20 minutes
Topics: Probability basics, Bayes theorem fundamentals

BACKGROUND:
-----------
Naive Bayes is a probabilistic classifier based on Bayes' theorem with the
"naive" assumption that features are independent given the class.

Bayes' Theorem:
    P(class|features) = P(features|class) * P(class) / P(features)

Since P(features) is constant for all classes, we can simplify to:
    P(class|features) ∝ P(class) * P(features|class)

With the naive independence assumption:
    P(features|class) = P(f1|class) * P(f2|class) * ... * P(fn|class)

INTERVIEW TIP:
--------------
Start by explaining the math before coding. Pinterest interviewers want to
see that you understand WHY the algorithm works, not just HOW to code it.

INSTRUCTIONS:
-------------
1. Complete each TODO section
2. Run the tests: pytest level1_basics.py -v
3. Each function builds on the previous one
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


# =============================================================================
# EXERCISE 1: Calculate Prior Probabilities
# =============================================================================
def calculate_priors(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate the prior probability P(class) for each class.
    
    Prior probability = count of class / total samples
    
    Args:
        y: Array of class labels, e.g., [0, 1, 1, 0, 1]
    
    Returns:
        Dictionary mapping class label to its prior probability
        e.g., {0: 0.4, 1: 0.6}
    
    Example:
        >>> y = np.array([0, 0, 1, 1, 1])
        >>> calculate_priors(y)
        {0: 0.4, 1: 0.6}
    
    HINT: Use np.unique with return_counts=True, or Counter
    """
    # TODO: Implement this function (3-5 lines)
    # Step 1: Count occurrences of each class
    # Step 2: Divide by total number of samples
    # Step 3: Return as dictionary
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 2: Calculate Likelihood (for discrete features)
# =============================================================================
def calculate_likelihood_discrete(
    X: np.ndarray, 
    y: np.ndarray, 
    feature_idx: int, 
    feature_value: int, 
    class_label: int
) -> float:
    """
    Calculate P(feature=value | class) for a discrete feature.
    
    This is the likelihood: given a class, what's the probability of
    seeing this particular feature value?
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Class labels (n_samples,)
        feature_idx: Which feature column to look at
        feature_value: The value we're calculating probability for
        class_label: The class we're conditioning on
    
    Returns:
        P(feature=value | class)
    
    Example:
        >>> X = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [1, 1]])
        >>> y = np.array([1, 1, 0, 0, 1])
        >>> # P(feature_0=1 | class=1) = count(feature_0=1 AND class=1) / count(class=1)
        >>> calculate_likelihood_discrete(X, y, 0, 1, 1)
        1.0  # All class 1 samples have feature_0=1
    
    HINT: Filter X to only samples where y == class_label, then count
    """
    # TODO: Implement this function (4-6 lines)
    # Step 1: Get indices where y == class_label
    # Step 2: Get the feature values for those samples
    # Step 3: Count how many equal feature_value
    # Step 4: Divide by total count of class_label
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 3: Predict Single Sample (Discrete Naive Bayes)
# =============================================================================
def predict_single_discrete(
    sample: np.ndarray,
    priors: Dict[int, float],
    likelihoods: Dict[Tuple[int, int, int], float],  # (feature_idx, value, class) -> prob
    classes: List[int]
) -> int:
    """
    Predict the class for a single sample using Naive Bayes.
    
    For each class, calculate:
        P(class) * P(f1|class) * P(f2|class) * ... * P(fn|class)
    
    Return the class with the highest probability.
    
    Args:
        sample: Single sample feature vector, e.g., [1, 0, 1]
        priors: Dictionary of prior probabilities {class: P(class)}
        likelihoods: Pre-computed likelihoods {(feature_idx, value, class): P(value|class)}
        classes: List of possible class labels
    
    Returns:
        Predicted class label
    
    Example:
        >>> sample = np.array([1, 0])
        >>> priors = {0: 0.5, 1: 0.5}
        >>> likelihoods = {
        ...     (0, 1, 0): 0.2, (0, 1, 1): 0.8,  # P(f0=1|class)
        ...     (0, 0, 0): 0.8, (0, 0, 1): 0.2,  # P(f0=0|class)
        ...     (1, 1, 0): 0.3, (1, 1, 1): 0.7,  # P(f1=1|class)
        ...     (1, 0, 0): 0.7, (1, 0, 1): 0.3,  # P(f1=0|class)
        ... }
        >>> predict_single_discrete(sample, priors, likelihoods, [0, 1])
        1  # Because P(1)*P(f0=1|1)*P(f1=0|1) > P(0)*P(f0=1|0)*P(f1=0|0)
    
    INTERVIEW TIP: Explain that you're using log probabilities in production
    to avoid underflow, but for simplicity we use regular multiplication here.
    """
    # TODO: Implement this function (8-12 lines)
    # Step 1: Initialize best_class and best_prob
    # Step 2: For each class:
    #         - Start with prior probability
    #         - Multiply by likelihood for each feature
    #         - Track if this is the best so far
    # Step 3: Return best_class
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 4: Laplace Smoothing
# =============================================================================
def calculate_likelihood_with_smoothing(
    X: np.ndarray, 
    y: np.ndarray, 
    feature_idx: int, 
    feature_value: int, 
    class_label: int,
    alpha: float = 1.0,
    n_feature_values: int = 2  # For binary features
) -> float:
    """
    Calculate likelihood with Laplace (additive) smoothing.
    
    Without smoothing, if a feature value never appears with a class,
    P(feature|class) = 0, which zeros out the entire probability!
    
    Laplace smoothing adds a small count to avoid this:
        P(feature=v|class) = (count + alpha) / (total + alpha * n_values)
    
    Args:
        X, y, feature_idx, feature_value, class_label: Same as before
        alpha: Smoothing parameter (1.0 = add-one smoothing)
        n_feature_values: Number of possible values for this feature
    
    Returns:
        Smoothed probability P(feature=value | class)
    
    Example:
        >>> # Even if feature_value never appears with class, prob > 0
        >>> X = np.array([[1], [1], [1]])
        >>> y = np.array([0, 0, 0])
        >>> # Without smoothing, P(f=0|class=0) = 0
        >>> # With smoothing: (0 + 1) / (3 + 1*2) = 0.2
        >>> calculate_likelihood_with_smoothing(X, y, 0, 0, 0)
        0.2
    
    INTERVIEW TIP: Always mention Laplace smoothing when implementing Naive Bayes!
    """
    # TODO: Implement this function (5-7 lines)
    # Step 1: Get samples where y == class_label
    # Step 2: Count how many have feature == feature_value
    # Step 3: Apply smoothing formula: (count + alpha) / (total + alpha * n_values)
    
    pass  # Remove this and implement


# =============================================================================
# TESTS - Run with: pytest level1_basics.py -v
# =============================================================================
class TestNaiveBayesBasics:
    """Tests for Level 1 exercises."""
    
    def test_calculate_priors_balanced(self):
        """Test priors with balanced classes."""
        y = np.array([0, 0, 1, 1])
        priors = calculate_priors(y)
        assert priors[0] == 0.5
        assert priors[1] == 0.5
    
    def test_calculate_priors_imbalanced(self):
        """Test priors with imbalanced classes."""
        y = np.array([0, 1, 1, 1, 1])
        priors = calculate_priors(y)
        assert priors[0] == 0.2
        assert priors[1] == 0.8
    
    def test_calculate_priors_multiclass(self):
        """Test priors with 3 classes."""
        y = np.array([0, 0, 1, 1, 1, 2])
        priors = calculate_priors(y)
        assert abs(priors[0] - 2/6) < 1e-6
        assert abs(priors[1] - 3/6) < 1e-6
        assert abs(priors[2] - 1/6) < 1e-6
    
    def test_likelihood_discrete_basic(self):
        """Test basic likelihood calculation."""
        X = np.array([[1], [1], [0], [0]])
        y = np.array([1, 1, 0, 0])
        # P(feature=1 | class=1) = 2/2 = 1.0
        assert calculate_likelihood_discrete(X, y, 0, 1, 1) == 1.0
        # P(feature=0 | class=0) = 2/2 = 1.0
        assert calculate_likelihood_discrete(X, y, 0, 0, 0) == 1.0
    
    def test_likelihood_discrete_partial(self):
        """Test likelihood with partial matches."""
        X = np.array([[1], [0], [1], [0]])
        y = np.array([1, 1, 0, 0])
        # P(feature=1 | class=1) = 1/2 = 0.5
        assert calculate_likelihood_discrete(X, y, 0, 1, 1) == 0.5
    
    def test_predict_single_discrete(self):
        """Test single prediction."""
        priors = {0: 0.5, 1: 0.5}
        likelihoods = {
            (0, 0, 0): 0.9, (0, 0, 1): 0.1,
            (0, 1, 0): 0.1, (0, 1, 1): 0.9,
        }
        # Sample [1] should predict class 1 (0.5 * 0.9 > 0.5 * 0.1)
        assert predict_single_discrete(np.array([1]), priors, likelihoods, [0, 1]) == 1
        # Sample [0] should predict class 0
        assert predict_single_discrete(np.array([0]), priors, likelihoods, [0, 1]) == 0
    
    def test_smoothing_prevents_zero(self):
        """Test that smoothing prevents zero probabilities."""
        X = np.array([[1], [1], [1]])
        y = np.array([0, 0, 0])
        # Without smoothing, P(f=0|class=0) would be 0
        # With smoothing: (0 + 1) / (3 + 2) = 0.2
        prob = calculate_likelihood_with_smoothing(X, y, 0, 0, 0, alpha=1.0, n_feature_values=2)
        assert prob > 0
        assert abs(prob - 0.2) < 1e-6
    
    def test_smoothing_doesnt_break_normal(self):
        """Test smoothing with normal case."""
        X = np.array([[1], [0], [1], [0]])
        y = np.array([0, 0, 0, 0])
        # count(f=1|class=0) = 2, total = 4
        # With smoothing: (2 + 1) / (4 + 2) = 0.5
        prob = calculate_likelihood_with_smoothing(X, y, 0, 1, 0, alpha=1.0, n_feature_values=2)
        assert abs(prob - 0.5) < 1e-6


if __name__ == "__main__":
    # Quick manual test
    print("Testing Naive Bayes Basics...")
    
    # Test priors
    y = np.array([0, 0, 1, 1, 1])
    priors = calculate_priors(y)
    print(f"Priors: {priors}")  # Should be {0: 0.4, 1: 0.6}
    
    # Test likelihood
    X = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [1, 1]])
    y = np.array([1, 1, 0, 0, 1])
    lik = calculate_likelihood_discrete(X, y, 0, 1, 1)
    print(f"P(f0=1|class=1): {lik}")  # Should be 1.0
    
    print("\nRun 'pytest level1_basics.py -v' for full tests")
