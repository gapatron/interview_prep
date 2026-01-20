"""
GRADIENT DESCENT - LEVEL 1: BASICS
==================================

Difficulty: Easy
Time to complete: 15-20 minutes
Topics: Gradient concept, basic optimization, learning rate

BACKGROUND:
-----------
Gradient Descent is THE fundamental optimization algorithm in ML.
It finds the minimum of a function by iteratively moving in the direction
of steepest descent (negative gradient).

Update rule:
    θ_new = θ_old - learning_rate * gradient(θ_old)

INTERVIEW TIP:
--------------
When implementing gradient descent, ALWAYS mention:
1. Learning rate selection (too high = overshoot, too low = slow)
2. Convergence criteria (when to stop)
3. Variants (batch, mini-batch, stochastic)

INSTRUCTIONS:
-------------
1. Complete each TODO section
2. Run tests: pytest level1_basics.py -v
3. Visualize the optimization path (bonus)
"""

import numpy as np
from typing import Callable, Tuple, List


# =============================================================================
# EXERCISE 1: Simple Quadratic Optimization
# =============================================================================
def gradient_descent_1d(
    f: Callable[[float], float],
    grad_f: Callable[[float], float],
    x0: float,
    learning_rate: float = 0.1,
    n_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[float, List[float]]:
    """
    Find the minimum of a 1D function using gradient descent.
    
    Args:
        f: The function to minimize
        grad_f: The gradient (derivative) of f
        x0: Starting point
        learning_rate: Step size (η)
        n_iterations: Maximum iterations
        tolerance: Stop if |gradient| < tolerance
    
    Returns:
        Tuple of (final_x, history) where history is list of x values
    
    Example:
        >>> # Minimize f(x) = x^2, gradient = 2x
        >>> f = lambda x: x**2
        >>> grad_f = lambda x: 2*x
        >>> x_min, history = gradient_descent_1d(f, grad_f, x0=5.0)
        >>> abs(x_min) < 0.01  # Should be close to 0
        True
    
    THINK ALOUD:
    "I'll start at x0 and repeatedly:
    1. Calculate the gradient at current position
    2. Move in the opposite direction of the gradient
    3. Stop when gradient is very small (we found the minimum)"
    """
    # TODO: Implement this function (8-12 lines)
    # Step 1: Initialize x = x0 and history list
    # Step 2: Loop for n_iterations:
    #         - Calculate gradient at current x
    #         - Check if |gradient| < tolerance -> break
    #         - Update x = x - learning_rate * gradient
    #         - Append x to history
    # Step 3: Return (x, history)
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 2: Multi-dimensional Gradient Descent
# =============================================================================
def gradient_descent_nd(
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    learning_rate: float = 0.1,
    n_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Gradient descent for multi-dimensional functions.
    
    This is the general form used in ML where we optimize multiple parameters.
    
    Args:
        grad_f: Function that returns gradient vector
        x0: Starting point (numpy array)
        learning_rate: Step size
        n_iterations: Maximum iterations
        tolerance: Stop if ||gradient|| < tolerance
    
    Returns:
        Tuple of (final_x, history)
    
    Example:
        >>> # Minimize f(x,y) = x^2 + y^2
        >>> # Gradient = [2x, 2y]
        >>> grad_f = lambda p: 2 * p
        >>> x0 = np.array([5.0, 3.0])
        >>> x_min, history = gradient_descent_nd(grad_f, x0)
        >>> np.linalg.norm(x_min) < 0.01
        True
    
    HINT: Use np.linalg.norm() to compute ||gradient||
    """
    # TODO: Implement this function (8-12 lines)
    # Same as 1D but:
    # - x is a numpy array
    # - gradient is a numpy array
    # - Use np.linalg.norm(gradient) for tolerance check
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 3: Linear Regression with Gradient Descent
# =============================================================================
def linear_regression_gradient(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Calculate gradient of MSE loss for linear regression.
    
    Loss = (1/n) * Σ(y_pred - y)²
         = (1/n) * Σ(X @ w - y)²
    
    Gradient w.r.t. weights:
        ∂Loss/∂w = (2/n) * X.T @ (X @ w - y)
    
    Args:
        X: Features matrix (n_samples, n_features)
        y: Target values (n_samples,)
        weights: Current weights (n_features,)
    
    Returns:
        Gradient vector (n_features,)
    
    Example:
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])
        >>> y = np.array([1, 2, 3])
        >>> w = np.array([0.0, 0.0])
        >>> grad = linear_regression_gradient(X, y, w)
        >>> # Gradient should point towards optimal weights
    
    DERIVATION (be ready to explain):
    1. y_pred = X @ w
    2. error = y_pred - y
    3. Loss = (1/n) * error.T @ error
    4. ∂Loss/∂w = (2/n) * X.T @ error
    """
    # TODO: Implement this function (3-5 lines)
    # Step 1: Calculate predictions: y_pred = X @ weights
    # Step 2: Calculate error: error = y_pred - y
    # Step 3: Calculate gradient: (2/n) * X.T @ error
    
    pass  # Remove this and implement


def fit_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    Fit linear regression using gradient descent.
    
    Args:
        X: Features matrix (n_samples, n_features)
        y: Target values (n_samples,)
        learning_rate: Step size
        n_iterations: Maximum iterations
        tolerance: Convergence threshold
    
    Returns:
        Tuple of (weights, loss_history)
    
    Example:
        >>> X = np.column_stack([np.ones(100), np.linspace(0, 10, 100)])
        >>> y = 2 + 3 * np.linspace(0, 10, 100) + np.random.randn(100) * 0.5
        >>> weights, losses = fit_linear_regression(X, y)
        >>> # weights should be close to [2, 3]
    
    INTERVIEW TIP: Add bias term by prepending column of 1s to X
    """
    # TODO: Implement this function (12-15 lines)
    # Step 1: Initialize weights to zeros (or small random)
    # Step 2: Initialize loss_history list
    # Step 3: Loop for n_iterations:
    #         - Calculate gradient using linear_regression_gradient
    #         - Update weights
    #         - Calculate loss: MSE = mean((X @ weights - y)^2)
    #         - Append loss to history
    #         - Check convergence (optional)
    # Step 4: Return (weights, loss_history)
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 4: Learning Rate Experiments
# =============================================================================
def compare_learning_rates(
    X: np.ndarray,
    y: np.ndarray,
    learning_rates: List[float],
    n_iterations: int = 100
) -> dict:
    """
    Compare different learning rates for linear regression.
    
    This helps understand the impact of learning rate selection!
    
    Args:
        X: Features matrix
        y: Target values
        learning_rates: List of learning rates to try
        n_iterations: Number of iterations for each
    
    Returns:
        Dictionary mapping learning_rate -> loss_history
    
    What to observe:
    - Too small: Slow convergence
    - Just right: Fast convergence to minimum
    - Too large: Oscillation or divergence
    
    INTERVIEW TIP: Discuss adaptive learning rates (Adam, RMSprop)
    when asked about learning rate selection.
    """
    # TODO: Implement this function (5-8 lines)
    # For each learning rate:
    # - Run fit_linear_regression
    # - Store loss history
    # Return dictionary
    
    pass  # Remove this and implement


# =============================================================================
# TESTS
# =============================================================================
class TestGradientDescentBasics:
    """Tests for gradient descent basics."""
    
    def test_1d_quadratic(self):
        """Test 1D gradient descent on simple quadratic."""
        f = lambda x: x**2
        grad_f = lambda x: 2*x
        x_min, history = gradient_descent_1d(f, grad_f, x0=5.0, learning_rate=0.1)
        
        assert abs(x_min) < 0.01, f"Expected x near 0, got {x_min}"
        assert len(history) > 0
    
    def test_1d_quadratic_shifted(self):
        """Test 1D gradient descent with minimum at x=3."""
        f = lambda x: (x - 3)**2
        grad_f = lambda x: 2*(x - 3)
        x_min, history = gradient_descent_1d(f, grad_f, x0=0.0, learning_rate=0.1)
        
        assert abs(x_min - 3) < 0.01, f"Expected x near 3, got {x_min}"
    
    def test_1d_learning_rate_too_large(self):
        """Test that large learning rate causes problems."""
        f = lambda x: x**2
        grad_f = lambda x: 2*x
        # With lr=1.0, we'll oscillate: 5 -> -5 -> 5 -> ...
        x_min, history = gradient_descent_1d(f, grad_f, x0=5.0, learning_rate=1.0, n_iterations=10)
        
        # Should NOT converge properly
        assert abs(x_min) > 0.1 or len(set([abs(h) for h in history[-5:]])) > 1
    
    def test_nd_gradient_descent(self):
        """Test multi-dimensional gradient descent."""
        grad_f = lambda p: 2 * p
        x0 = np.array([5.0, 3.0, -2.0])
        x_min, history = gradient_descent_nd(grad_f, x0, learning_rate=0.1)
        
        assert np.linalg.norm(x_min) < 0.01
    
    def test_linear_regression_gradient_zero(self):
        """Test gradient at optimal point."""
        X = np.array([[1, 0], [1, 1], [1, 2]])
        y = np.array([1, 2, 3])  # y = 1 + 1*x
        w_optimal = np.array([1.0, 1.0])
        
        grad = linear_regression_gradient(X, y, w_optimal)
        # At optimal point, gradient should be near zero
        assert np.linalg.norm(grad) < 0.01
    
    def test_linear_regression_gradient_direction(self):
        """Test that gradient points towards minimum."""
        X = np.array([[1, 1], [1, 2], [1, 3]])
        y = np.array([2, 3, 4])  # y = 1 + 1*x
        
        # Start with w = [0, 0], gradient should point in positive direction
        w = np.array([0.0, 0.0])
        grad = linear_regression_gradient(X, y, w)
        
        # Since y > X @ w (all zeros), gradient should be negative
        # (we want to increase weights)
        assert grad[0] < 0 and grad[1] < 0
    
    def test_fit_linear_regression_simple(self):
        """Test fitting simple linear relationship."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.linspace(0, 10, n)])
        y = 2 + 3 * X[:, 1]  # y = 2 + 3x, no noise
        
        weights, losses = fit_linear_regression(X, y, learning_rate=0.01, n_iterations=1000)
        
        assert abs(weights[0] - 2) < 0.1, f"Intercept should be ~2, got {weights[0]}"
        assert abs(weights[1] - 3) < 0.1, f"Slope should be ~3, got {weights[1]}"
        assert losses[-1] < losses[0], "Loss should decrease"
    
    def test_fit_linear_regression_noisy(self):
        """Test fitting with noise."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.linspace(0, 10, n)])
        y = 2 + 3 * X[:, 1] + np.random.randn(n) * 0.5
        
        weights, losses = fit_linear_regression(X, y, learning_rate=0.01, n_iterations=1000)
        
        # Should be close to true values despite noise
        assert abs(weights[0] - 2) < 0.5
        assert abs(weights[1] - 3) < 0.5
    
    def test_compare_learning_rates(self):
        """Test learning rate comparison."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.linspace(0, 5, n)])
        y = 1 + 2 * X[:, 1]
        
        results = compare_learning_rates(X, y, [0.001, 0.01, 0.1], n_iterations=100)
        
        if results is not None:
            assert 0.001 in results
            assert 0.01 in results
            assert 0.1 in results
            # Smaller lr should have higher final loss (slower convergence)
            assert results[0.001][-1] > results[0.01][-1]


if __name__ == "__main__":
    print("Testing Gradient Descent Basics...")
    print("Run 'pytest level1_basics.py -v' for full tests")
    
    # Quick demo: Minimize f(x) = (x-3)^2
    print("\n--- 1D Gradient Descent Demo ---")
    f = lambda x: (x - 3)**2
    grad_f = lambda x: 2*(x - 3)
    x_min, history = gradient_descent_1d(f, grad_f, x0=0.0, learning_rate=0.1)
    print(f"Minimum found at x = {x_min:.4f} (true minimum: 3)")
    print(f"Iterations: {len(history)}")
