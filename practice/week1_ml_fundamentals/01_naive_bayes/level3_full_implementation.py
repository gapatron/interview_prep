"""
NAIVE BAYES - LEVEL 3: INTERVIEW-READY IMPLEMENTATION
======================================================

Difficulty: Hard (Interview Level)
Time to complete: 25-30 minutes
Topics: Complete implementation, multiple variants, edge cases

BACKGROUND:
-----------
This is the complete implementation you'd write in an interview.
It handles:
1. Multiple variants (Gaussian, Multinomial, Bernoulli)
2. Edge cases (zero variance, missing values)
3. Log probabilities for numerical stability
4. Prediction probabilities (not just labels)

INTERVIEW FORMAT:
-----------------
You have 15-20 minutes to implement Naive Bayes from scratch.
The interviewer will likely ask:
1. "Implement Naive Bayes for continuous features"
2. Follow-up: "How would you handle numerical stability?"
3. Follow-up: "What if a feature has zero variance?"

INSTRUCTIONS:
-------------
1. Time yourself - try to complete in under 20 minutes
2. Practice explaining your code as you write
3. Run tests to verify: pytest level3_full_implementation.py -v
"""

import numpy as np
from typing import Optional, List, Union
from abc import ABC, abstractmethod


class BaseNaiveBayes(ABC):
    """
    Base class for Naive Bayes implementations.
    
    Provides common functionality for all variants.
    """
    
    def __init__(self):
        self.classes_: Optional[np.ndarray] = None
        self.class_prior_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseNaiveBayes':
        """Fit the model to training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        n_samples = len(y)
        
        # Calculate class priors
        self.class_prior_ = np.array([
            np.sum(y == c) / n_samples for c in self.classes_
        ])
        
        self.n_features_ = X.shape[1]
        
        # Subclass-specific fitting
        self._fit(X, y)
        
        return self
    
    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Subclass-specific fitting logic."""
        pass
    
    @abstractmethod
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Calculate log P(class) + log P(features | class) for each class."""
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using softmax."""
        jll = self._joint_log_likelihood(X)
        # Softmax for numerical stability
        jll_max = jll.max(axis=1, keepdims=True)
        log_proba = jll - jll_max - np.log(
            np.exp(jll - jll_max).sum(axis=1, keepdims=True)
        )
        return np.exp(log_proba)
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict log class probabilities."""
        return np.log(self.predict_proba(X))


# =============================================================================
# EXERCISE 1: Complete Gaussian Naive Bayes
# =============================================================================
class GaussianNB(BaseNaiveBayes):
    """
    Gaussian Naive Bayes classifier.
    
    Assumes features follow a Gaussian distribution within each class.
    
    Attributes after fitting:
        theta_: Mean of each feature per class (n_classes, n_features)
        var_: Variance of each feature per class (n_classes, n_features)
    
    Example:
        >>> clf = GaussianNB()
        >>> clf.fit(X_train, y_train)
        >>> clf.predict(X_test)
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        """
        Args:
            var_smoothing: Added to variance for numerical stability
        """
        super().__init__()
        self.var_smoothing = var_smoothing
        self.theta_: Optional[np.ndarray] = None  # Means
        self.var_: Optional[np.ndarray] = None    # Variances
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculate mean and variance for each feature, for each class.
        
        TODO: Implement this method (10-15 lines)
        
        Steps:
        1. Initialize theta_ and var_ arrays with shape (n_classes, n_features)
        2. For each class:
           - Get samples belonging to that class
           - Calculate mean of each feature -> store in theta_
           - Calculate variance of each feature -> store in var_
        3. Add var_smoothing to all variances
        
        HINT: Use np.mean(X_class, axis=0) and np.var(X_class, axis=0)
        """
        # TODO: Implement
        pass
    
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate joint log likelihood for each class.
        
        For each sample and each class:
            log P(class) + Σ log P(feature_i | class)
        
        Where log P(feature | class) is log of Gaussian PDF:
            -0.5 * log(2π * var) - (x - mean)² / (2 * var)
        
        TODO: Implement this method (10-15 lines)
        
        Returns:
            Array of shape (n_samples, n_classes)
        
        HINT: You can vectorize this!
        - For each class, calculate all samples at once
        - log P(x|class) = -0.5 * (log(2π*var) + (x-mean)²/var)
        - Sum across features for each sample
        """
        # TODO: Implement
        pass


# =============================================================================
# EXERCISE 2: Multinomial Naive Bayes (for text/count data)
# =============================================================================
class MultinomialNB(BaseNaiveBayes):
    """
    Multinomial Naive Bayes classifier.
    
    Used for discrete features (like word counts in text classification).
    
    P(feature_i | class) = (count_i + alpha) / (total_count + alpha * n_features)
    
    Attributes after fitting:
        feature_count_: Count of each feature per class
        feature_log_prob_: Log probability of each feature per class
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Laplace smoothing parameter (1.0 = add-one smoothing)
        """
        super().__init__()
        self.alpha = alpha
        self.feature_count_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculate feature counts and log probabilities.
        
        TODO: Implement this method (10-15 lines)
        
        Steps:
        1. Initialize feature_count_ array (n_classes, n_features)
        2. For each class:
           - Sum feature values for samples of that class
           - Store in feature_count_
        3. Calculate feature_log_prob_ with Laplace smoothing:
           log P(feature | class) = log((count + alpha) / (total + alpha * n_features))
        """
        # TODO: Implement
        pass
    
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate joint log likelihood for multinomial model.
        
        For text classification with word counts, this is:
            log P(class) + Σ count_i * log P(word_i | class)
        
        TODO: Implement this method (5-8 lines)
        
        HINT: Use matrix multiplication: X @ feature_log_prob_.T
        """
        # TODO: Implement
        pass


# =============================================================================
# EXERCISE 3: Bernoulli Naive Bayes (for binary features)
# =============================================================================
class BernoulliNB(BaseNaiveBayes):
    """
    Bernoulli Naive Bayes classifier.
    
    Used for binary features (feature present/absent).
    
    Key difference from Multinomial: Bernoulli considers BOTH presence AND absence.
    P(x | class) = P(feature=1|class)^x * P(feature=0|class)^(1-x)
    
    This is important for document classification where absence of a word is informative!
    """
    
    def __init__(self, alpha: float = 1.0, binarize: Optional[float] = 0.0):
        """
        Args:
            alpha: Laplace smoothing parameter
            binarize: Threshold for binarizing features (None to skip)
        """
        super().__init__()
        self.alpha = alpha
        self.binarize = binarize
        self.feature_prob_: Optional[np.ndarray] = None  # P(feature=1 | class)
    
    def _binarize_features(self, X: np.ndarray) -> np.ndarray:
        """Convert features to binary based on threshold."""
        if self.binarize is not None:
            return (X > self.binarize).astype(np.float64)
        return X
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculate probability of each feature being 1 for each class.
        
        TODO: Implement this method (10-12 lines)
        
        Steps:
        1. Binarize features if needed
        2. For each class:
           - Count how many times each feature is 1
           - Calculate P(feature=1 | class) with smoothing
        """
        # TODO: Implement
        pass
    
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate joint log likelihood for Bernoulli model.
        
        log P(x | class) = Σ [x_i * log P(f_i=1|c) + (1-x_i) * log P(f_i=0|c)]
        
        TODO: Implement this method (8-12 lines)
        """
        # TODO: Implement
        pass


# =============================================================================
# BONUS: Complete Interview Implementation (timed practice)
# =============================================================================
def interview_gaussian_nb(X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray) -> np.ndarray:
    """
    THIS IS THE FUNCTION YOU'D WRITE IN AN INTERVIEW.
    
    Given training data and test data, return predictions.
    Time yourself: Can you implement this in 15 minutes?
    
    Requirements:
    1. Use Gaussian distribution for continuous features
    2. Handle numerical stability (log probabilities)
    3. Return predicted class labels
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        X_test: Test features (n_test, n_features)
    
    Returns:
        Predicted labels for X_test
    
    INTERVIEW TIP:
    - Start by explaining your approach
    - Write clean, readable code
    - Handle edge cases (zero variance)
    - Mention log probabilities
    """
    # TODO: Implement from scratch in 15 minutes
    # Don't use the classes above - write it inline!
    
    pass


# =============================================================================
# TESTS
# =============================================================================
class TestNaiveBayesImplementations:
    """Tests for all Naive Bayes variants."""
    
    def test_gaussian_nb_fit(self):
        """Test GaussianNB fitting."""
        X = np.array([[1.0, 2.0], [1.5, 2.5], [5.0, 6.0], [5.5, 6.5]])
        y = np.array([0, 0, 1, 1])
        
        clf = GaussianNB()
        clf.fit(X, y)
        
        assert clf.theta_ is not None
        assert clf.var_ is not None
        assert clf.theta_.shape == (2, 2)  # 2 classes, 2 features
        np.testing.assert_array_almost_equal(clf.theta_[0], [1.25, 2.25])
        np.testing.assert_array_almost_equal(clf.theta_[1], [5.25, 6.25])
    
    def test_gaussian_nb_predict(self):
        """Test GaussianNB prediction."""
        np.random.seed(42)
        X_c0 = np.random.randn(50, 2) + [0, 0]
        X_c1 = np.random.randn(50, 2) + [5, 5]
        X = np.vstack([X_c0, X_c1])
        y = np.array([0] * 50 + [1] * 50)
        
        clf = GaussianNB()
        clf.fit(X, y)
        
        accuracy = (clf.predict(X) == y).mean()
        assert accuracy > 0.95
    
    def test_gaussian_nb_predict_proba(self):
        """Test probability predictions sum to 1."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 3, 100)
        
        clf = GaussianNB()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(100))
    
    def test_gaussian_nb_vs_sklearn(self):
        """Compare with sklearn."""
        try:
            from sklearn.naive_bayes import GaussianNB as SklearnGNB
        except ImportError:
            return
        
        np.random.seed(42)
        X = np.random.randn(200, 4)
        y = np.random.randint(0, 2, 200)
        
        our_clf = GaussianNB()
        our_clf.fit(X[:150], y[:150])
        
        sk_clf = SklearnGNB()
        sk_clf.fit(X[:150], y[:150])
        
        agreement = (our_clf.predict(X[150:]) == sk_clf.predict(X[150:])).mean()
        assert agreement > 0.95
    
    def test_multinomial_nb_fit(self):
        """Test MultinomialNB fitting."""
        # Word count data
        X = np.array([[3, 0, 1], [2, 0, 2], [0, 3, 1], [0, 4, 0]])
        y = np.array([0, 0, 1, 1])
        
        clf = MultinomialNB()
        clf.fit(X, y)
        
        assert clf.feature_log_prob_ is not None
        # Log probs should be negative (probs < 1)
        assert (clf.feature_log_prob_ <= 0).all()
    
    def test_multinomial_nb_predict(self):
        """Test MultinomialNB prediction."""
        # Simulate document-word counts
        np.random.seed(42)
        X = np.random.poisson(lam=2, size=(100, 10))
        y = np.array([0] * 50 + [1] * 50)
        # Make classes somewhat distinguishable
        X[:50, :5] += 3
        X[50:, 5:] += 3
        
        clf = MultinomialNB()
        clf.fit(X, y)
        accuracy = (clf.predict(X) == y).mean()
        assert accuracy > 0.7
    
    def test_bernoulli_nb_fit(self):
        """Test BernoulliNB fitting."""
        X = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1]])
        y = np.array([0, 0, 1, 1])
        
        clf = BernoulliNB()
        clf.fit(X, y)
        
        assert clf.feature_prob_ is not None
        # Probabilities should be between 0 and 1
        assert (clf.feature_prob_ > 0).all()
        assert (clf.feature_prob_ < 1).all()
    
    def test_bernoulli_nb_binarization(self):
        """Test BernoulliNB with binarization."""
        X = np.array([[0.5, 1.5], [0.3, 0.7], [2.0, 0.1], [1.8, 0.2]])
        y = np.array([0, 0, 1, 1])
        
        clf = BernoulliNB(binarize=0.5)
        clf.fit(X, y)
        
        # After binarization with threshold 0.5:
        # [[0, 1], [0, 1], [1, 0], [1, 0]]
        predictions = clf.predict(X)
        assert len(predictions) == 4
    
    def test_interview_implementation(self):
        """Test the interview implementation."""
        np.random.seed(42)
        X_train = np.vstack([
            np.random.randn(50, 2) + [0, 0],
            np.random.randn(50, 2) + [5, 5]
        ])
        y_train = np.array([0] * 50 + [1] * 50)
        X_test = np.array([[0, 0], [5, 5], [2.5, 2.5]])
        
        predictions = interview_gaussian_nb(X_train, y_train, X_test)
        
        if predictions is not None:
            assert predictions[0] == 0  # Close to class 0 center
            assert predictions[1] == 1  # Close to class 1 center


if __name__ == "__main__":
    print("=" * 60)
    print("NAIVE BAYES - INTERVIEW PRACTICE")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Implement all TODO sections")
    print("2. Run: pytest level3_full_implementation.py -v")
    print("3. Time yourself on interview_gaussian_nb (target: 15 min)")
    print("\nGood luck!")
