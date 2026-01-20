"""
PCA VS LDA - UNDERSTANDING DIMENSIONALITY REDUCTION
===================================================

Difficulty: Medium
Time to complete: 25-30 minutes
Topics: PCA, LDA, when to use each

THIS IS A HIGH-FREQUENCY PINTEREST INTERVIEW TOPIC!

BACKGROUND:
-----------
Both PCA and LDA are dimensionality reduction techniques, but they have
fundamentally different goals.

PCA (Principal Component Analysis):
- UNSUPERVISED (doesn't use class labels)
- Goal: Maximize variance in projected data
- Finds directions of maximum variance
- Use when: You want to compress data while preserving information

LDA (Linear Discriminant Analysis):
- SUPERVISED (uses class labels)
- Goal: Maximize class separation
- Finds directions that best separate classes
- Use when: You want to improve classification

INTERVIEW QUESTION:
"When would you use PCA vs LDA?"

ANSWER:
- PCA: When you don't have labels, or want general dimensionality reduction
- LDA: When you have labeled data and want to maximize class separability
- PCA first, then LDA: Common pipeline to reduce noise before classification

KEY DIFFERENCES:
| Aspect     | PCA                  | LDA                    |
|------------|----------------------|------------------------|
| Type       | Unsupervised         | Supervised             |
| Goal       | Max variance         | Max class separation   |
| Uses labels| No                   | Yes                    |
| Max dims   | min(n_samples, n_features) | n_classes - 1    |

INSTRUCTIONS:
-------------
1. Complete each implementation
2. Run tests: pytest pca_vs_lda.py -v
3. Be able to explain the difference clearly!
"""

import numpy as np
from typing import Tuple, Optional


# =============================================================================
# EXERCISE 1: PCA Implementation
# =============================================================================
def pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement Principal Component Analysis.
    
    Algorithm:
    1. Center the data (subtract mean)
    2. Compute covariance matrix
    3. Compute eigenvalues and eigenvectors
    4. Sort by eigenvalue (descending)
    5. Select top k eigenvectors
    6. Project data onto new basis
    
    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Number of components to keep
    
    Returns:
        Tuple of:
        - X_transformed: Projected data (n_samples, n_components)
        - components: Principal components (n_components, n_features)
        - explained_variance: Variance explained by each component
    
    Example:
        >>> X = np.random.randn(100, 5)
        >>> X_pca, components, var = pca(X, 2)
        >>> X_pca.shape
        (100, 2)
    
    INTERVIEW TIP:
    "PCA finds the directions of maximum variance in the data,
    without considering any class labels. It's unsupervised."
    """
    # TODO: Implement PCA (15-20 lines)
    # Step 1: Center data (subtract mean)
    #         mean = np.mean(X, axis=0)
    #         X_centered = X - mean
    #
    # Step 2: Compute covariance matrix
    #         cov = (X_centered.T @ X_centered) / (n_samples - 1)
    #
    # Step 3: Eigendecomposition
    #         eigenvalues, eigenvectors = np.linalg.eigh(cov)
    #
    # Step 4: Sort by eigenvalue descending
    #         idx = np.argsort(eigenvalues)[::-1]
    #         eigenvalues = eigenvalues[idx]
    #         eigenvectors = eigenvectors[:, idx]
    #
    # Step 5: Select top components
    #         components = eigenvectors[:, :n_components].T
    #
    # Step 6: Project data
    #         X_transformed = X_centered @ components.T
    
    pass  # Remove this and implement


def pca_using_svd(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA using Singular Value Decomposition.
    
    SVD is more numerically stable than eigendecomposition of covariance.
    This is how sklearn implements PCA!
    
    X = U * S * V^T
    
    The principal components are the rows of V^T (or columns of V).
    The projected data is U * S (truncated to n_components).
    
    INTERVIEW TIP: Mention SVD as the preferred implementation for stability.
    """
    # TODO: Implement PCA using SVD (10-12 lines)
    # Step 1: Center data
    # Step 2: SVD: U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Step 3: Components are rows of Vt[:n_components]
    # Step 4: Projected data is U[:, :n_components] * S[:n_components]
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 2: LDA Implementation
# =============================================================================
def lda(X: np.ndarray, y: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implement Linear Discriminant Analysis.
    
    Algorithm:
    1. Compute class means and overall mean
    2. Compute within-class scatter matrix Sw
    3. Compute between-class scatter matrix Sb
    4. Solve eigenvalue problem for Sw^(-1) * Sb
    5. Select top eigenvectors
    6. Project data
    
    Args:
        X: Data matrix (n_samples, n_features)
        y: Class labels (n_samples,)
        n_components: Number of components (max: n_classes - 1)
    
    Returns:
        Tuple of:
        - X_transformed: Projected data (n_samples, n_components)
        - components: LDA components (n_components, n_features)
    
    Scatter Matrices:
    - Sw (within-class): Sum of covariances within each class
    - Sb (between-class): Scatter of class means around overall mean
    
    Goal: Maximize Sb / Sw ratio
    
    Example:
        >>> X = np.vstack([np.random.randn(50, 4) + [0,0,0,0],
        ...                np.random.randn(50, 4) + [3,3,3,3]])
        >>> y = np.array([0]*50 + [1]*50)
        >>> X_lda, components = lda(X, y, 1)
        >>> X_lda.shape
        (100, 1)
    
    INTERVIEW TIP:
    "LDA maximizes the ratio of between-class scatter to within-class scatter.
    It uses class labels to find projections that separate classes."
    """
    # TODO: Implement LDA (25-30 lines)
    # Step 1: Get unique classes and dimensions
    #         classes = np.unique(y)
    #         n_features = X.shape[1]
    #
    # Step 2: Compute overall mean
    #         mean_overall = np.mean(X, axis=0)
    #
    # Step 3: Compute within-class scatter Sw
    #         Sw = np.zeros((n_features, n_features))
    #         for c in classes:
    #             X_c = X[y == c]
    #             mean_c = np.mean(X_c, axis=0)
    #             Sw += (X_c - mean_c).T @ (X_c - mean_c)
    #
    # Step 4: Compute between-class scatter Sb
    #         Sb = np.zeros((n_features, n_features))
    #         for c in classes:
    #             n_c = np.sum(y == c)
    #             mean_c = np.mean(X[y == c], axis=0)
    #             mean_diff = (mean_c - mean_overall).reshape(-1, 1)
    #             Sb += n_c * (mean_diff @ mean_diff.T)
    #
    # Step 5: Solve eigenvalue problem
    #         eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
    #
    # Step 6: Sort and select (careful: eigenvalues might be complex)
    #
    # Step 7: Project data
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 3: Compare PCA and LDA
# =============================================================================
def compare_pca_lda(X: np.ndarray, y: np.ndarray, n_components: int = 2) -> dict:
    """
    Compare PCA and LDA on the same data.
    
    Returns:
        Dictionary with:
        - 'pca_result': PCA-transformed data
        - 'lda_result': LDA-transformed data
        - 'pca_explained_var': Variance explained by PCA
        - 'pca_class_separation': Some measure of class separation
        - 'lda_class_separation': Same measure for LDA
    
    This helps visualize why LDA is better for classification tasks.
    """
    # TODO: Implement comparison (15-20 lines)
    
    pass  # Remove this and implement


def class_separation_score(X_transformed: np.ndarray, y: np.ndarray) -> float:
    """
    Compute a simple class separation score.
    
    Score = (between-class variance) / (within-class variance)
    
    Higher score = better separation.
    """
    # TODO: Implement (10-12 lines)
    
    pass  # Remove this and implement


# =============================================================================
# EXERCISE 4: When to Use Which
# =============================================================================
def recommend_method(
    has_labels: bool,
    goal: str,  # 'compression', 'classification', 'visualization'
    n_classes: Optional[int] = None,
    n_features: int = 100
) -> str:
    """
    Recommend PCA or LDA based on the scenario.
    
    This function encodes the decision logic you should explain in an interview.
    
    Args:
        has_labels: Whether class labels are available
        goal: What you're trying to achieve
        n_classes: Number of classes (if applicable)
        n_features: Number of features
    
    Returns:
        Recommendation string explaining which method to use and why.
    
    INTERVIEW SCENARIOS:
    
    1. "I have images and want to compress them for storage"
       -> PCA (unsupervised compression)
    
    2. "I have labeled spam/ham emails and want to classify new ones"
       -> LDA (supervised, maximizes class separation)
    
    3. "I have gene expression data and want to find patterns"
       -> PCA first (unsupervised exploration)
    
    4. "I have 1000 features but only 2 classes"
       -> LDA gives max 1 component! Use PCA for more dimensions.
    """
    # TODO: Implement decision logic
    # Return a string explaining the recommendation
    
    pass  # Remove this and implement


# =============================================================================
# DEMO: Visual Comparison
# =============================================================================
def visual_comparison_demo():
    """
    Demonstrate the difference between PCA and LDA visually.
    
    PCA finds directions of maximum variance (may not separate classes well).
    LDA finds directions of maximum class separation.
    """
    np.random.seed(42)
    
    # Create two-class data
    # Class 0: High variance along x, centered at (0, 0)
    # Class 1: High variance along x, centered at (0, 3)
    n_per_class = 100
    
    class_0 = np.random.randn(n_per_class, 2) * [3, 0.5] + [0, 0]
    class_1 = np.random.randn(n_per_class, 2) * [3, 0.5] + [0, 3]
    
    X = np.vstack([class_0, class_1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    
    print("PCA vs LDA Demonstration")
    print("=" * 50)
    print(f"\nData: Two classes with {n_per_class} samples each")
    print("Classes are separated along y-axis but have high variance along x-axis")
    
    # PCA
    X_pca, pca_components, _ = pca(X, 1) if pca(X, 1) else (None, None, None)
    
    # LDA
    X_lda, lda_components = lda(X, y, 1) if lda(X, y, 1) else (None, None)
    
    print("\nPCA (unsupervised):")
    if pca_components is not None:
        print(f"  First component direction: {pca_components[0]}")
        print("  PCA will likely find the x-direction (max variance)")
    else:
        print("  (not implemented)")
    
    print("\nLDA (supervised):")
    if lda_components is not None:
        print(f"  First component direction: {lda_components[0]}")
        print("  LDA will find the y-direction (max class separation)")
    else:
        print("  (not implemented)")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHT:")
    print("- PCA maximizes variance -> might find wrong direction for classification")
    print("- LDA maximizes class separation -> better for classification")
    print("- If classes overlap significantly, LDA might not help much either")


# =============================================================================
# TESTS
# =============================================================================
class TestPCALDA:
    """Tests for PCA and LDA implementations."""
    
    def test_pca_dimensionality(self):
        """Test that PCA reduces dimensions correctly."""
        X = np.random.randn(100, 10)
        result = pca(X, 3)
        
        if result:
            X_transformed, components, explained_var = result
            assert X_transformed.shape == (100, 3)
            assert components.shape == (3, 10)
            assert len(explained_var) == 3
    
    def test_pca_variance_ordering(self):
        """Test that PCA components are ordered by variance."""
        X = np.random.randn(100, 5)
        result = pca(X, 5)
        
        if result:
            _, _, explained_var = result
            # Variance should be in descending order
            assert all(explained_var[i] >= explained_var[i+1] 
                      for i in range(len(explained_var)-1))
    
    def test_pca_reconstruction(self):
        """Test that we can reconstruct from PCA."""
        X = np.random.randn(100, 5)
        result = pca(X, 5)  # Keep all components
        
        if result:
            X_transformed, components, _ = result
            # Reconstruct
            X_reconstructed = X_transformed @ components + np.mean(X, axis=0)
            # Should be close to original
            np.testing.assert_array_almost_equal(X, X_reconstructed, decimal=5)
    
    def test_lda_dimensionality(self):
        """Test that LDA reduces to at most n_classes - 1."""
        np.random.seed(42)
        X = np.vstack([np.random.randn(50, 10) + i for i in range(3)])
        y = np.array([0]*50 + [1]*50 + [2]*50)
        
        result = lda(X, y, 2)  # 3 classes -> max 2 components
        
        if result:
            X_transformed, components = result
            assert X_transformed.shape == (150, 2)
            assert components.shape == (2, 10)
    
    def test_lda_improves_separation(self):
        """Test that LDA improves class separation compared to PCA."""
        np.random.seed(42)
        # Data where classes differ mainly in one direction
        class_0 = np.random.randn(50, 5) * [3, 0.5, 0.5, 0.5, 0.5] + [0, 0, 0, 0, 0]
        class_1 = np.random.randn(50, 5) * [3, 0.5, 0.5, 0.5, 0.5] + [0, 2, 0, 0, 0]
        X = np.vstack([class_0, class_1])
        y = np.array([0]*50 + [1]*50)
        
        pca_result = pca(X, 1)
        lda_result = lda(X, y, 1)
        
        if pca_result and lda_result:
            X_pca, _, _ = pca_result
            X_lda, _ = lda_result
            
            pca_sep = class_separation_score(X_pca, y)
            lda_sep = class_separation_score(X_lda, y)
            
            if pca_sep and lda_sep:
                # LDA should give better separation
                assert lda_sep >= pca_sep
    
    def test_pca_svd_matches_eig(self):
        """Test that SVD-based PCA matches eigenvalue PCA."""
        X = np.random.randn(100, 5)
        
        result_eig = pca(X, 3)
        result_svd = pca_using_svd(X, 3)
        
        if result_eig and result_svd:
            X_eig, _, _ = result_eig
            X_svd, _, _ = result_svd
            
            # Results might differ in sign, so compare absolute values
            np.testing.assert_array_almost_equal(
                np.abs(X_eig), np.abs(X_svd), decimal=5
            )


if __name__ == "__main__":
    print("PCA VS LDA - Understanding Dimensionality Reduction")
    print("=" * 60)
    
    # Run demo
    visual_comparison_demo()
    
    print("\n\nINTERVIEW SUMMARY:")
    print("-" * 60)
    print("""
    PCA (Principal Component Analysis):
    - Unsupervised (no labels needed)
    - Finds directions of maximum VARIANCE
    - Use for: compression, noise reduction, visualization
    
    LDA (Linear Discriminant Analysis):
    - Supervised (needs class labels)
    - Finds directions of maximum CLASS SEPARATION
    - Use for: classification preprocessing
    - Limited to (n_classes - 1) dimensions
    
    When to use which:
    - No labels? -> PCA
    - Classification task? -> Consider LDA
    - Many classes, few components needed? -> PCA (LDA limited)
    - Pipeline: Often PCA first (reduce noise), then LDA
    """)
    
    print("\nRun 'pytest pca_vs_lda.py -v' for tests")
