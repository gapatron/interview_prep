"""
GRADIENT DESCENT - LEVEL 2: VARIANTS
====================================

Difficulty: Medium
Time to complete: 25-30 minutes
Topics: Batch, Mini-batch, Stochastic GD, Momentum

BACKGROUND:
-----------
Different variants of gradient descent trade off between:
- Computation per step
- Noise in gradient estimate
- Convergence speed

BATCH GRADIENT DESCENT:
- Uses ALL training samples to compute gradient
- Accurate gradient, but slow for large datasets
- Gradient = (1/n) * Σ gradient_i for all samples

STOCHASTIC GRADIENT DESCENT (SGD):
- Uses ONE random sample per step
- Noisy gradient, but fast updates
- Can escape local minima due to noise

MINI-BATCH GRADIENT DESCENT:
- Uses a SUBSET of samples (e.g., 32, 64, 128)
- Balance between accuracy and speed
- Standard in deep learning

MOMENTUM:
- Adds "velocity" to gradient descent
- Helps accelerate in consistent directions
- Dampens oscillations

INTERVIEW TIP:
--------------
Know the trade-offs! Pinterest might ask:
"When would you use SGD vs mini-batch?"
"What is momentum and why use it?"
"""

import numpy as np
from typing import Tuple, List, Callable, Optional


# =============================================================================
# EXERCISE 1: Stochastic Gradient Descent
# =============================================================================
def sgd_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_epochs: int = 100,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, List[float]]:
    """
    Fit linear regression using Stochastic Gradient Descent.
    
    Updates weights after EACH sample (one at a time).
    
    Args:
        X: Features (n_samples, n_features)
        y: Targets (n_samples,)
        learning_rate: Step size
        n_epochs: Number of passes through the data
        shuffle: Whether to shuffle data each epoch
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (weights, loss_history)
    
    Algorithm:
        For each epoch:
            Shuffle data (optional)
            For each sample (x_i, y_i):
                prediction = x_i @ weights
                error = prediction - y_i
                gradient = 2 * x_i * error
                weights = weights - learning_rate * gradient
            Record loss
    
    Example:
        >>> np.random.seed(42)
        >>> X = np.column_stack([np.ones(100), np.random.randn(100)])
        >>> y = 2 + 3 * X[:, 1]
        >>> weights, losses = sgd_linear_regression(X, y)
    
    INTERVIEW TIP: SGD is noisier but can be faster for large datasets.
    """
    # TODO: Implement this function (15-20 lines)
    # Step 1: Set random seed if provided
    # Step 2: Initialize weights
    # Step 3: For each epoch:
    #         - Shuffle if needed (use indices)
    #         - For each sample:
    #           - Calculate gradient for this sample
    #           - Update weights
    #         - Record average loss for epoch
    # Step 4: Return weights, losses
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 2: Mini-Batch Gradient Descent
# =============================================================================
def minibatch_gd_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    n_epochs: int = 100,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, List[float]]:
    """
    Fit linear regression using Mini-Batch Gradient Descent.
    
    Updates weights after each BATCH of samples.
    
    Args:
        X: Features (n_samples, n_features)
        y: Targets (n_samples,)
        learning_rate: Step size
        batch_size: Number of samples per batch
        n_epochs: Number of passes through data
        random_state: Random seed
    
    Returns:
        Tuple of (weights, loss_history)
    
    Algorithm:
        For each epoch:
            Shuffle data
            Split into batches of size batch_size
            For each batch:
                gradient = (2/batch_size) * X_batch.T @ (X_batch @ w - y_batch)
                weights = weights - learning_rate * gradient
            Record loss
    
    INTERVIEW TIP: Mini-batch is the standard in deep learning.
    Typical batch sizes: 32, 64, 128, 256.
    """
    # TODO: Implement this function (18-22 lines)
    # Step 1: Initialize
    # Step 2: For each epoch:
    #         - Shuffle indices
    #         - Create batches
    #         - For each batch:
    #           - Calculate gradient
    #           - Update weights
    #         - Record loss
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 3: Gradient Descent with Momentum
# =============================================================================
def gd_with_momentum(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    n_iterations: int = 1000
) -> Tuple[np.ndarray, List[float]]:
    """
    Gradient descent with momentum for linear regression.
    
    Momentum adds a "velocity" term that accumulates past gradients:
        velocity = momentum * velocity - learning_rate * gradient
        weights = weights + velocity
    
    This helps:
    - Accelerate convergence in consistent directions
    - Dampen oscillations in inconsistent directions
    
    Args:
        X: Features (n_samples, n_features)
        y: Targets (n_samples,)
        learning_rate: Step size
        momentum: Momentum coefficient (typically 0.9)
        n_iterations: Number of iterations
    
    Returns:
        Tuple of (weights, loss_history)
    
    Example:
        Imagine a ball rolling down a hill:
        - Without momentum: Stops as soon as gradient = 0
        - With momentum: Continues rolling, can overcome small bumps
    
    INTERVIEW TIP: Momentum helps with:
    1. Ravines (narrow valleys where regular GD oscillates)
    2. Saddle points (where gradient is small)
    """
    # TODO: Implement this function (12-15 lines)
    # Step 1: Initialize weights and velocity (both zeros)
    # Step 2: For each iteration:
    #         - Calculate gradient
    #         - Update velocity: v = momentum * v - lr * gradient
    #         - Update weights: w = w + v
    #         - Record loss
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 4: Comparison Function
# =============================================================================
def compare_gd_variants(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 50
) -> dict:
    """
    Compare all gradient descent variants on the same data.
    
    Returns:
        Dictionary with loss histories for each method:
        {
            'batch': [...],
            'sgd': [...],
            'minibatch_32': [...],
            'minibatch_64': [...],
            'momentum': [...]
        }
    
    What to observe:
    - Batch GD: Smooth convergence, potentially slow
    - SGD: Noisy convergence, fast updates
    - Mini-batch: Balance of both
    - Momentum: Faster convergence, less oscillation
    """
    # TODO: Implement this function
    # Run each variant and collect loss histories
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 5: Learning Rate Scheduling
# =============================================================================
def gd_with_lr_schedule(
    X: np.ndarray,
    y: np.ndarray,
    initial_lr: float = 0.1,
    decay_rate: float = 0.95,
    decay_steps: int = 100,
    n_iterations: int = 1000,
    schedule_type: str = "exponential"
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Gradient descent with learning rate scheduling.
    
    Learning rate schedules help:
    - Start with large steps (fast progress)
    - End with small steps (fine-tuning)
    
    Schedule types:
    - "exponential": lr = initial_lr * (decay_rate ^ (iteration / decay_steps))
    - "step": lr = initial_lr * (decay_rate ^ floor(iteration / decay_steps))
    - "inverse": lr = initial_lr / (1 + decay_rate * iteration)
    
    Args:
        X, y: Training data
        initial_lr: Starting learning rate
        decay_rate: How much to decay
        decay_steps: How often to decay (for step schedule)
        n_iterations: Number of iterations
        schedule_type: Type of schedule
    
    Returns:
        Tuple of (weights, loss_history, lr_history)
    
    INTERVIEW TIP: In practice, use adaptive methods (Adam, RMSprop)
    which automatically adjust learning rates per parameter.
    """
    # TODO: Implement this function (15-20 lines)
    # Step 1: Initialize weights
    # Step 2: For each iteration:
    #         - Calculate current learning rate based on schedule
    #         - Calculate gradient
    #         - Update weights with current lr
    #         - Record loss and lr
    
    pass  # Remove this and implement


# =============================================================================
# TESTS
# =============================================================================
class TestGradientDescentVariants:
    """Tests for GD variants."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.n = 200
        self.X = np.column_stack([np.ones(self.n), np.random.randn(self.n)])
        self.y = 2 + 3 * self.X[:, 1] + np.random.randn(self.n) * 0.1
    
    def test_sgd_converges(self):
        """Test that SGD converges."""
        weights, losses = sgd_linear_regression(
            self.X, self.y, 
            learning_rate=0.01, 
            n_epochs=100,
            random_state=42
        )
        
        # Should get close to true weights [2, 3]
        assert abs(weights[0] - 2) < 0.5, f"Intercept: {weights[0]}"
        assert abs(weights[1] - 3) < 0.5, f"Slope: {weights[1]}"
        # Loss should decrease
        assert losses[-1] < losses[0]
    
    def test_sgd_noisy_convergence(self):
        """Test that SGD is noisier than batch GD."""
        _, sgd_losses = sgd_linear_regression(
            self.X, self.y, learning_rate=0.01, n_epochs=50, random_state=42
        )
        
        # SGD losses should have some variance
        if len(sgd_losses) > 10:
            variance = np.var(sgd_losses[-10:])
            # Some variance is expected (may be 0 if converged)
            assert variance >= 0
    
    def test_minibatch_converges(self):
        """Test that mini-batch GD converges."""
        weights, losses = minibatch_gd_linear_regression(
            self.X, self.y,
            learning_rate=0.01,
            batch_size=32,
            n_epochs=100,
            random_state=42
        )
        
        assert abs(weights[0] - 2) < 0.5
        assert abs(weights[1] - 3) < 0.5
        assert losses[-1] < losses[0]
    
    def test_minibatch_different_sizes(self):
        """Test different batch sizes."""
        _, losses_32 = minibatch_gd_linear_regression(
            self.X, self.y, batch_size=32, n_epochs=50, random_state=42
        )
        _, losses_64 = minibatch_gd_linear_regression(
            self.X, self.y, batch_size=64, n_epochs=50, random_state=42
        )
        
        # Both should converge
        if losses_32 and losses_64:
            assert losses_32[-1] < losses_32[0]
            assert losses_64[-1] < losses_64[0]
    
    def test_momentum_converges(self):
        """Test momentum-based GD."""
        weights, losses = gd_with_momentum(
            self.X, self.y,
            learning_rate=0.01,
            momentum=0.9,
            n_iterations=500
        )
        
        assert abs(weights[0] - 2) < 0.5
        assert abs(weights[1] - 3) < 0.5
    
    def test_momentum_helps_convergence(self):
        """Test that momentum speeds up convergence."""
        _, losses_no_momentum = gd_with_momentum(
            self.X, self.y, momentum=0.0, n_iterations=200
        )
        _, losses_with_momentum = gd_with_momentum(
            self.X, self.y, momentum=0.9, n_iterations=200
        )
        
        # Momentum should help reach lower loss faster
        # (This test might be flaky depending on initialization)
        if losses_no_momentum and losses_with_momentum:
            assert losses_with_momentum[-1] < 1.0  # Should converge
    
    def test_lr_schedule_decreases(self):
        """Test that learning rate decreases over time."""
        weights, losses, lrs = gd_with_lr_schedule(
            self.X, self.y,
            initial_lr=0.1,
            decay_rate=0.95,
            n_iterations=100
        )
        
        if lrs:
            assert lrs[-1] < lrs[0], "LR should decrease"
            assert losses[-1] < losses[0], "Loss should decrease"
    
    def test_compare_variants(self):
        """Test comparison function."""
        results = compare_gd_variants(self.X, self.y, n_epochs=30)
        
        if results:
            for key in ['batch', 'sgd', 'minibatch_32']:
                if key in results:
                    assert len(results[key]) > 0


if __name__ == "__main__":
    print("Testing Gradient Descent Variants...")
    print("Run 'pytest level2_variants.py -v' for full tests")
    
    # Quick comparison
    np.random.seed(42)
    n = 200
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    y = 2 + 3 * X[:, 1] + np.random.randn(n) * 0.1
    
    print("\n--- Comparing GD Variants ---")
    results = compare_gd_variants(X, y, n_epochs=50)
    if results:
        for method, losses in results.items():
            if losses:
                print(f"{method}: Final loss = {losses[-1]:.6f}")
