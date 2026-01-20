"""
GRADIENT DESCENT - LEVEL 3: ADVANCED (Interview Ready)
======================================================

Difficulty: Hard
Time to complete: 25-30 minutes
Topics: Logistic Regression, Regularization, Adaptive Methods

BACKGROUND:
-----------
This level covers advanced topics you might see in Pinterest interviews:
1. Gradient descent for logistic regression (classification)
2. Regularization in gradient descent (L1, L2)
3. Understanding adaptive optimizers (Adam, RMSprop)

INTERVIEW TIP:
--------------
If asked to implement gradient descent, you might need to:
1. Derive gradients for different loss functions
2. Add regularization terms
3. Discuss trade-offs between optimizers
"""

import numpy as np
from typing import Tuple, List, Optional


# =============================================================================
# HELPER: Sigmoid Function
# =============================================================================
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid function with numerical stability.
    
    σ(z) = 1 / (1 + exp(-z))
    
    For numerical stability, use:
    - If z >= 0: σ(z) = 1 / (1 + exp(-z))
    - If z < 0: σ(z) = exp(z) / (1 + exp(z))
    
    This avoids overflow in exp() for large negative z.
    """
    # TODO: Implement numerically stable sigmoid (5-8 lines)
    # HINT: Use np.where to handle positive and negative z separately
    
    pass  # Remove this and implement


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Compute derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))
    """
    s = sigmoid(z)
    return s * (1 - s)


# =============================================================================
# EXERCISE 1: Logistic Regression Gradient
# =============================================================================
def logistic_regression_gradient(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Compute gradient of binary cross-entropy loss for logistic regression.
    
    Loss = -(1/n) * Σ [y*log(p) + (1-y)*log(1-p)]
    where p = sigmoid(X @ w)
    
    Gradient: ∂Loss/∂w = (1/n) * X.T @ (p - y)
    
    Note: The gradient has the same form as linear regression,
    but with sigmoid(X @ w) instead of X @ w!
    
    Args:
        X: Features (n_samples, n_features)
        y: Binary targets (n_samples,) with values 0 or 1
        weights: Current weights (n_features,)
    
    Returns:
        Gradient vector (n_features,)
    
    INTERVIEW TIP: Be ready to derive this!
    The key insight is that log-loss has a nice gradient form.
    """
    # TODO: Implement this function (4-6 lines)
    # Step 1: Calculate predictions: p = sigmoid(X @ weights)
    # Step 2: Calculate gradient: (1/n) * X.T @ (p - y)
    
    pass  # Remove this and implement


def logistic_regression_loss(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    epsilon: float = 1e-15
) -> float:
    """
    Compute binary cross-entropy loss.
    
    Loss = -(1/n) * Σ [y*log(p) + (1-y)*log(1-p)]
    
    Add epsilon to avoid log(0).
    
    Args:
        X: Features
        y: Binary targets
        weights: Current weights
        epsilon: Small value for numerical stability
    
    Returns:
        Scalar loss value
    """
    # TODO: Implement this function (5-7 lines)
    # Step 1: Calculate predictions p = sigmoid(X @ weights)
    # Step 2: Clip p to [epsilon, 1-epsilon] to avoid log(0)
    # Step 3: Calculate loss
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 2: Gradient Descent for Logistic Regression
# =============================================================================
def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    n_iterations: int = 1000,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    Fit logistic regression using gradient descent.
    
    Args:
        X: Features (n_samples, n_features)
        y: Binary targets (n_samples,)
        learning_rate: Step size
        n_iterations: Maximum iterations
        tolerance: Convergence threshold
    
    Returns:
        Tuple of (weights, loss_history)
    
    Example:
        >>> X = np.array([[1, -2], [1, -1], [1, 1], [1, 2]])
        >>> y = np.array([0, 0, 1, 1])
        >>> weights, losses = fit_logistic_regression(X, y)
        >>> # Should classify correctly based on second feature
    """
    # TODO: Implement this function (12-15 lines)
    # Similar to linear regression, but use logistic gradient and loss
    
    pass  # Remove this and implement


def predict_logistic(X: np.ndarray, weights: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Predict binary class labels."""
    probabilities = sigmoid(X @ weights)
    return (probabilities >= threshold).astype(int)


# =============================================================================
# EXERCISE 3: L2 Regularization (Ridge)
# =============================================================================
def gradient_with_l2_regularization(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lambda_reg: float,
    loss_type: str = "mse"
) -> np.ndarray:
    """
    Compute gradient with L2 regularization.
    
    L2 regularization adds λ * ||w||² to the loss:
        Total Loss = Original Loss + λ * Σ w_i²
    
    Gradient of regularization term: 2 * λ * w
    
    Total Gradient = Original Gradient + 2 * λ * w
    
    Note: Usually we DON'T regularize the bias term (first weight).
    
    Args:
        X: Features
        y: Targets
        weights: Current weights
        lambda_reg: Regularization strength
        loss_type: "mse" for linear regression, "logistic" for log loss
    
    Returns:
        Gradient with regularization
    
    INTERVIEW TIP: L2 regularization:
    - Shrinks weights towards zero (but doesn't eliminate them)
    - Equivalent to adding Gaussian prior on weights
    - Helps with overfitting
    """
    # TODO: Implement this function (8-12 lines)
    # Step 1: Calculate base gradient (depends on loss_type)
    # Step 2: Add regularization gradient: 2 * lambda * weights
    # Step 3: (Optional) Don't regularize bias term (weights[0])
    
    pass  # Remove this and implement


def fit_with_l2_regularization(
    X: np.ndarray,
    y: np.ndarray,
    lambda_reg: float = 0.1,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    loss_type: str = "mse"
) -> Tuple[np.ndarray, List[float]]:
    """
    Fit model with L2 regularization.
    
    Args:
        X: Features
        y: Targets
        lambda_reg: Regularization strength
        learning_rate: Step size
        n_iterations: Max iterations
        loss_type: "mse" or "logistic"
    
    Returns:
        Tuple of (weights, loss_history)
    """
    # TODO: Implement this function
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 4: L1 Regularization (Lasso) - Subgradient Method
# =============================================================================
def gradient_with_l1_regularization(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lambda_reg: float
) -> np.ndarray:
    """
    Compute gradient with L1 regularization.
    
    L1 regularization adds λ * ||w||₁ = λ * Σ|w_i| to the loss.
    
    The "gradient" of |w| is:
    - +1 if w > 0
    - -1 if w < 0
    - 0 if w = 0 (technically undefined, we use 0)
    
    This is called the SUBGRADIENT.
    
    Args:
        X: Features
        y: Targets
        weights: Current weights
        lambda_reg: Regularization strength
    
    Returns:
        Gradient with L1 regularization
    
    INTERVIEW TIP: L1 regularization:
    - Can set weights exactly to zero (sparse solutions)
    - Good for feature selection
    - Equivalent to Laplace prior on weights
    """
    # TODO: Implement this function (8-10 lines)
    # Step 1: Calculate MSE gradient
    # Step 2: Calculate subgradient of L1 term: lambda * sign(weights)
    # Step 3: Add them together (don't regularize bias)
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 5: Understanding Adam (Conceptual)
# =============================================================================
class AdamOptimizer:
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Adam combines:
    1. Momentum (first moment estimate)
    2. RMSprop (second moment estimate - adaptive learning rate)
    
    Key idea: Different learning rates for different parameters,
    based on historical gradient information.
    
    Update rules:
        m = β1 * m + (1 - β1) * gradient          # First moment (momentum)
        v = β2 * v + (1 - β2) * gradient²         # Second moment (RMSprop)
        m_hat = m / (1 - β1^t)                    # Bias correction
        v_hat = v / (1 - β2^t)                    # Bias correction
        weights = weights - lr * m_hat / (sqrt(v_hat) + ε)
    
    INTERVIEW TIP: Know that Adam:
    - Adapts learning rate per parameter
    - Works well with sparse gradients
    - Default choice for deep learning
    - Typical hyperparams: lr=0.001, β1=0.9, β2=0.999, ε=1e-8
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # State
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Timestep
    
    def step(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform one Adam update step.
        
        TODO: Implement this method (10-15 lines)
        """
        # Initialize moments if first step
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)
        
        # TODO: Implement Adam update
        # Step 1: Increment timestep
        # Step 2: Update biased first moment: m = β1 * m + (1 - β1) * g
        # Step 3: Update biased second moment: v = β2 * v + (1 - β2) * g²
        # Step 4: Compute bias-corrected moments
        # Step 5: Update weights
        
        pass  # Remove this and implement


def fit_with_adam(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 1000,
    learning_rate: float = 0.001
) -> Tuple[np.ndarray, List[float]]:
    """
    Fit linear regression using Adam optimizer.
    
    Args:
        X: Features
        y: Targets
        n_iterations: Number of iterations
        learning_rate: Learning rate for Adam
    
    Returns:
        Tuple of (weights, loss_history)
    """
    # TODO: Implement using AdamOptimizer class
    
    pass  # Remove this and implement


# =============================================================================
# INTERVIEW EXERCISE: Complete Implementation
# =============================================================================
def interview_gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task: str = "regression",
    regularization: Optional[str] = None,
    lambda_reg: float = 0.1
) -> np.ndarray:
    """
    INTERVIEW QUESTION: Implement gradient descent from scratch.
    
    Time yourself: 15-20 minutes
    
    Requirements:
    1. Support both regression (MSE) and classification (log loss)
    2. Support L2 regularization (optional)
    3. Return predictions on test set
    
    Args:
        X_train: Training features
        y_train: Training targets (continuous for regression, 0/1 for classification)
        X_test: Test features
        task: "regression" or "classification"
        regularization: None or "l2"
        lambda_reg: Regularization strength
    
    Returns:
        Predictions on X_test
    
    INTERVIEW TIPS:
    1. Start by explaining your approach
    2. Ask clarifying questions (learning rate? iterations?)
    3. Handle edge cases (zero variance features)
    4. Mention numerical stability
    """
    # TODO: Implement complete solution
    
    pass  # Remove this and implement


# =============================================================================
# TESTS
# =============================================================================
class TestAdvancedGradientDescent:
    """Tests for advanced GD topics."""
    
    def test_sigmoid_basic(self):
        """Test sigmoid function."""
        assert abs(sigmoid(np.array([0]))[0] - 0.5) < 1e-6
        assert sigmoid(np.array([100]))[0] > 0.99
        assert sigmoid(np.array([-100]))[0] < 0.01
    
    def test_sigmoid_stable(self):
        """Test numerical stability of sigmoid."""
        # Should not overflow
        result = sigmoid(np.array([1000, -1000]))
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_logistic_gradient_shape(self):
        """Test gradient shape."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        w = np.zeros(5)
        
        grad = logistic_regression_gradient(X, y, w)
        if grad is not None:
            assert grad.shape == (5,)
    
    def test_logistic_regression_separable(self):
        """Test on perfectly separable data."""
        np.random.seed(42)
        X = np.column_stack([np.ones(100), np.concatenate([
            np.random.randn(50) - 3,  # Class 0
            np.random.randn(50) + 3   # Class 1
        ])])
        y = np.array([0]*50 + [1]*50)
        
        weights, losses = fit_logistic_regression(X, y, learning_rate=0.1)
        
        if weights is not None:
            preds = predict_logistic(X, weights)
            accuracy = (preds == y).mean()
            assert accuracy > 0.9, f"Accuracy: {accuracy}"
    
    def test_l2_regularization_shrinks_weights(self):
        """Test that L2 regularization shrinks weights."""
        np.random.seed(42)
        X = np.column_stack([np.ones(100), np.random.randn(100, 5)])
        y = X[:, 1] + 0.5 * X[:, 2]  # True weights on features 1 and 2
        
        # Without regularization
        w_noreg, _ = fit_with_l2_regularization(X, y, lambda_reg=0)
        # With strong regularization
        w_reg, _ = fit_with_l2_regularization(X, y, lambda_reg=1.0)
        
        if w_noreg is not None and w_reg is not None:
            # Regularized weights should have smaller magnitude
            assert np.linalg.norm(w_reg) < np.linalg.norm(w_noreg)
    
    def test_l1_gradient_sparsity(self):
        """Test L1 gradient computation."""
        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        w = np.array([0.5, 0.0, -0.3])
        
        grad = gradient_with_l1_regularization(X, y, w, lambda_reg=0.1)
        
        if grad is not None:
            assert grad.shape == (3,)
    
    def test_adam_step(self):
        """Test Adam optimizer step."""
        optimizer = AdamOptimizer()
        w = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        
        new_w = optimizer.step(w, grad)
        
        if new_w is not None:
            assert new_w.shape == w.shape
            # Weights should change
            assert not np.allclose(new_w, w)
    
    def test_interview_implementation_regression(self):
        """Test interview implementation on regression."""
        np.random.seed(42)
        X_train = np.column_stack([np.ones(100), np.random.randn(100)])
        y_train = 2 + 3 * X_train[:, 1]
        X_test = np.column_stack([np.ones(10), np.random.randn(10)])
        
        preds = interview_gradient_descent(X_train, y_train, X_test, task="regression")
        
        if preds is not None:
            assert len(preds) == 10
    
    def test_interview_implementation_classification(self):
        """Test interview implementation on classification."""
        np.random.seed(42)
        X_train = np.column_stack([np.ones(100), np.concatenate([
            np.random.randn(50) - 2,
            np.random.randn(50) + 2
        ])])
        y_train = np.array([0]*50 + [1]*50)
        X_test = np.column_stack([np.ones(10), np.concatenate([
            np.random.randn(5) - 2,
            np.random.randn(5) + 2
        ])])
        
        preds = interview_gradient_descent(X_train, y_train, X_test, task="classification")
        
        if preds is not None:
            assert len(preds) == 10
            assert set(preds).issubset({0, 1})


if __name__ == "__main__":
    print("=" * 60)
    print("GRADIENT DESCENT - ADVANCED PRACTICE")
    print("=" * 60)
    print("\nTopics covered:")
    print("1. Sigmoid function (numerical stability)")
    print("2. Logistic regression with gradient descent")
    print("3. L2 regularization")
    print("4. L1 regularization")
    print("5. Adam optimizer")
    print("\nRun 'pytest level3_advanced.py -v' for tests")
