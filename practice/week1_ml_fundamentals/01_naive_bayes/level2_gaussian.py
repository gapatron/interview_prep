"""
NAIVE BAYES - LEVEL 2: GAUSSIAN NAIVE BAYES
============================================

Difficulty: Medium
Time to complete: 20-25 minutes
Topics: Continuous features, Gaussian distribution, fitting parameters

BACKGROUND:
-----------
For continuous features, we can't just count occurrences. Instead, we assume
features follow a Gaussian (normal) distribution for each class.

P(feature=x | class) = (1 / sqrt(2π σ²)) * exp(-(x - μ)² / (2σ²))

Where:
- μ (mean) and σ² (variance) are estimated from training data for each class

This is what sklearn's GaussianNB implements!

INTERVIEW TIP:
--------------
When asked "implement Naive Bayes for continuous features", this is what
they want. Be ready to explain:
1. Why we use Gaussian assumption
2. How to estimate μ and σ from data
3. How to handle numerical stability (log probabilities)

INSTRUCTIONS:
-------------
1. Complete each TODO section
2. Run the tests: pytest level2_gaussian.py -v
3. Compare your implementation with sklearn at the end
"""

import numpy as np
from typing import Dict, Tuple
from collections import defaultdict


# =============================================================================
# EXERCISE 1: Fit Class Statistics
# =============================================================================
def fit_gaussian_params(
    X: np.ndarray, 
    y: np.ndarray
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, float]]:
    """
    Calculate mean and variance for each feature, for each class.
    
    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
    
    Returns:
        Tuple of (means, variances, priors):
        - means: {class: array of feature means}
        - variances: {class: array of feature variances}
        - priors: {class: prior probability}
    
    Example:
        >>> X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]])
        >>> y = np.array([0, 0, 1, 1])
        >>> means, vars, priors = fit_gaussian_params(X, y)
        >>> means[0]  # Mean of class 0 features
        array([1.25, 1.9])
        >>> priors[0]
        0.5
    
    HINT: Use np.mean and np.var with axis=0 for column-wise statistics
    """
    # TODO: Implement this function (10-15 lines)
    # Step 1: Get unique classes
    # Step 2: Calculate priors for each class
    # Step 3: For each class:
    #         - Filter X to samples of that class
    #         - Calculate mean of each feature
    #         - Calculate variance of each feature
    # Step 4: Return means, variances, priors dictionaries
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 2: Calculate Gaussian Probability
# =============================================================================
def gaussian_probability(x: float, mean: float, var: float) -> float:
    """
    Calculate probability density for a value under Gaussian distribution.
    
    P(x | μ, σ²) = (1 / sqrt(2π σ²)) * exp(-(x - μ)² / (2σ²))
    
    Args:
        x: The value to calculate probability for
        mean: Mean (μ) of the distribution
        var: Variance (σ²) of the distribution
    
    Returns:
        Probability density at x
    
    Example:
        >>> # Standard normal at x=0 should be ~0.3989
        >>> gaussian_probability(0, 0, 1)
        0.3989...
    
    NUMERICAL STABILITY TIP:
    Add a small epsilon to variance to prevent division by zero:
        var = var + 1e-9
    """
    # TODO: Implement this function (3-5 lines)
    # Step 1: Add small epsilon to var for stability
    # Step 2: Calculate the exponent: -(x - mean)² / (2 * var)
    # Step 3: Calculate coefficient: 1 / sqrt(2π * var)
    # Step 4: Return coefficient * exp(exponent)
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 3: Calculate Log Probability (Numerical Stability)
# =============================================================================
def log_gaussian_probability(x: float, mean: float, var: float) -> float:
    """
    Calculate LOG probability density for numerical stability.
    
    When multiplying many small probabilities, we get underflow (result = 0).
    Using log probabilities, we add instead of multiply:
        log(P(A) * P(B)) = log(P(A)) + log(P(B))
    
    log P(x | μ, σ²) = -0.5 * log(2π σ²) - (x - μ)² / (2σ²)
    
    Args:
        x: The value to calculate log probability for
        mean: Mean of the distribution
        var: Variance of the distribution
    
    Returns:
        Log probability density at x
    
    Example:
        >>> log_gaussian_probability(0, 0, 1)
        -0.9189...  # log(0.3989...)
    
    INTERVIEW TIP:
    ALWAYS mention log probabilities when implementing probabilistic models!
    This shows production-level thinking.
    """
    # TODO: Implement this function (3-5 lines)
    # Step 1: Add epsilon to var
    # Step 2: Calculate: -0.5 * log(2π * var) - (x - mean)² / (2 * var)
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 4: Predict with Gaussian Naive Bayes
# =============================================================================
def predict_gaussian(
    X: np.ndarray,
    means: Dict[int, np.ndarray],
    variances: Dict[int, np.ndarray],
    priors: Dict[int, float]
) -> np.ndarray:
    """
    Predict class labels using Gaussian Naive Bayes.
    
    For each sample and each class, calculate:
        log P(class | features) ∝ log P(class) + Σ log P(feature_i | class)
    
    Return the class with highest log probability.
    
    Args:
        X: Test features (n_samples, n_features)
        means: {class: feature means} from training
        variances: {class: feature variances} from training
        priors: {class: prior probability} from training
    
    Returns:
        Predicted class labels (n_samples,)
    
    Example:
        >>> # After training on clearly separated data
        >>> X_test = np.array([[1.0, 2.0], [5.0, 8.0]])
        >>> predictions = predict_gaussian(X_test, means, variances, priors)
        >>> predictions
        array([0, 1])  # First sample closer to class 0, second to class 1
    """
    # TODO: Implement this function (15-20 lines)
    # Step 1: Initialize predictions array
    # Step 2: Get list of classes
    # Step 3: For each sample:
    #         - For each class, calculate log posterior:
    #           log P(class) + sum of log P(feature|class) for each feature
    #         - Select class with highest log posterior
    # Step 4: Return predictions array
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 5: Complete Gaussian Naive Bayes Classifier
# =============================================================================
class GaussianNaiveBayes:
    """
    Complete Gaussian Naive Bayes implementation.
    
    This is what you'd implement in an interview when asked for
    "Naive Bayes from scratch for continuous features."
    
    Usage:
        clf = GaussianNaiveBayes()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
    """
    
    def __init__(self):
        self.means = {}
        self.variances = {}
        self.priors = {}
        self.classes = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        """
        Fit the classifier to training data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self (for method chaining)
        """
        # TODO: Implement fit (5-10 lines)
        # Use your fit_gaussian_params function!
        # Store means, variances, priors, and classes
        
        pass  # Remove this and implement
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for test data.
        
        Args:
            X: Test features (n_samples, n_features)
        
        Returns:
            Predicted class labels (n_samples,)
        """
        # TODO: Implement predict (3-5 lines)
        # Use your predict_gaussian function!
        
        pass  # Remove this and implement
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for test data.
        
        This is a BONUS exercise - implement if you have time!
        
        Args:
            X: Test features (n_samples, n_features)
        
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        # TODO (BONUS): Implement predict_proba
        # Calculate log probabilities, then use softmax to convert to probabilities
        # softmax(x) = exp(x) / sum(exp(x))
        
        pass  # Remove this and implement


# =============================================================================
# TESTS
# =============================================================================
class TestGaussianNaiveBayes:
    """Tests for Gaussian Naive Bayes."""
    
    def test_fit_gaussian_params_basic(self):
        """Test fitting parameters."""
        X = np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 11.0], [11.0, 12.0]])
        y = np.array([0, 0, 1, 1])
        means, variances, priors = fit_gaussian_params(X, y)
        
        assert priors[0] == 0.5
        assert priors[1] == 0.5
        np.testing.assert_array_almost_equal(means[0], [1.5, 2.5])
        np.testing.assert_array_almost_equal(means[1], [10.5, 11.5])
    
    def test_gaussian_probability_standard_normal(self):
        """Test Gaussian probability with standard normal."""
        # P(0 | mean=0, var=1) ≈ 0.3989
        prob = gaussian_probability(0, 0, 1)
        assert abs(prob - 0.3989422804014327) < 1e-6
    
    def test_gaussian_probability_shifted(self):
        """Test Gaussian probability with shifted mean."""
        # P(5 | mean=5, var=1) should equal P(0 | mean=0, var=1)
        prob1 = gaussian_probability(0, 0, 1)
        prob2 = gaussian_probability(5, 5, 1)
        assert abs(prob1 - prob2) < 1e-6
    
    def test_log_gaussian_probability(self):
        """Test log probability calculation."""
        prob = gaussian_probability(0, 0, 1)
        log_prob = log_gaussian_probability(0, 0, 1)
        assert abs(np.log(prob) - log_prob) < 1e-6
    
    def test_predict_gaussian_separable(self):
        """Test prediction with well-separated classes."""
        # Class 0: features around (0, 0)
        # Class 1: features around (10, 10)
        means = {0: np.array([0.0, 0.0]), 1: np.array([10.0, 10.0])}
        variances = {0: np.array([1.0, 1.0]), 1: np.array([1.0, 1.0])}
        priors = {0: 0.5, 1: 0.5}
        
        X_test = np.array([[0.5, 0.5], [9.5, 9.5]])
        predictions = predict_gaussian(X_test, means, variances, priors)
        
        np.testing.assert_array_equal(predictions, [0, 1])
    
    def test_classifier_fit_predict(self):
        """Test complete classifier."""
        np.random.seed(42)
        # Generate clearly separable data
        X_class0 = np.random.randn(50, 2) + np.array([0, 0])
        X_class1 = np.random.randn(50, 2) + np.array([5, 5])
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 50 + [1] * 50)
        
        clf = GaussianNaiveBayes()
        clf.fit(X, y)
        predictions = clf.predict(X)
        
        # Should get most predictions correct on training data
        accuracy = (predictions == y).mean()
        assert accuracy > 0.9
    
    def test_classifier_vs_sklearn(self):
        """Compare with sklearn implementation."""
        try:
            from sklearn.naive_bayes import GaussianNB
        except ImportError:
            return  # Skip if sklearn not installed
        
        np.random.seed(42)
        X_class0 = np.random.randn(100, 3) * 2 + np.array([0, 0, 0])
        X_class1 = np.random.randn(100, 3) * 2 + np.array([5, 5, 5])
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 100 + [1] * 100)
        
        # Split into train/test
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        clf = GaussianNaiveBayes()
        clf.fit(X_train, y_train)
        our_pred = clf.predict(X_test)
        
        sklearn_clf = GaussianNB()
        sklearn_clf.fit(X_train, y_train)
        sklearn_pred = sklearn_clf.predict(X_test)
        
        # Should match sklearn closely
        agreement = (our_pred == sklearn_pred).mean()
        assert agreement > 0.95, f"Only {agreement:.2%} agreement with sklearn"


if __name__ == "__main__":
    print("Testing Gaussian Naive Bayes...")
    print("Run 'pytest level2_gaussian.py -v' for full tests")
    
    # Quick demo
    np.random.seed(42)
    X_class0 = np.random.randn(50, 2) + np.array([0, 0])
    X_class1 = np.random.randn(50, 2) + np.array([5, 5])
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * 50 + [1] * 50)
    
    clf = GaussianNaiveBayes()
    clf.fit(X, y)
    
    test_samples = np.array([[0, 0], [5, 5], [2.5, 2.5]])
    predictions = clf.predict(test_samples)
    print(f"Test samples: {test_samples.tolist()}")
    print(f"Predictions: {predictions}")
