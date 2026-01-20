"""
NUMBER OF ISLANDS - LEETCODE #200
=================================

Difficulty: Medium
Time to complete: 20-25 minutes
Source: https://leetcode.com/problems/number-of-islands/

THIS PROBLEM IS MENTIONED IN PINTEREST'S OFFICIAL INTERVIEW PREP!

PROBLEM DESCRIPTION:
--------------------
Given an m x n 2D binary grid which represents a map of '1's (land)
and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent
lands horizontally or vertically. You may assume all four edges of
the grid are surrounded by water.

EXAMPLES:
---------
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3

ALGORITHM:
----------
1. Iterate through each cell in the grid
2. When we find a '1' (land), it's a new island - increment counter
3. Use DFS/BFS to mark all connected land as visited (sink the island)
4. Continue until all cells are processed

INTERVIEW TIPS:
---------------
1. Clarify: Can we modify the input grid? (Usually yes)
2. Discuss DFS vs BFS trade-offs
3. Mention Union-Find as alternative
4. Time: O(m*n), Space: O(m*n) for recursion stack / O(min(m,n)) for BFS

INSTRUCTIONS:
-------------
1. Complete each solution variant
2. Run tests: pytest number_of_islands.py -v
3. Time yourself: Try to solve in under 20 minutes
"""

from typing import List
from collections import deque


# =============================================================================
# SOLUTION 1: DFS (Recursive)
# =============================================================================
def num_islands_dfs(grid: List[List[str]]) -> int:
    """
    Count islands using DFS to "sink" each island found.
    
    Args:
        grid: 2D grid of '1' (land) and '0' (water)
    
    Returns:
        Number of islands
    
    Example:
        >>> grid = [["1","1","0"],["0","1","0"],["0","0","1"]]
        >>> num_islands_dfs(grid)
        2
    
    Algorithm:
    1. For each cell (i, j):
       - If it's land ('1'), we found a new island
       - Use DFS to mark all connected land as water ('0')
       - Increment island count
    2. Return count
    
    Time: O(m * n) - visit each cell once
    Space: O(m * n) - worst case recursion depth
    """
    # TODO: Implement this function (15-20 lines)
    # Step 1: Handle edge case (empty grid)
    # Step 2: Get dimensions m, n
    # Step 3: Define DFS helper function:
    #         def dfs(i, j):
    #             - Check bounds and if cell is land
    #             - Mark as visited (set to '0')
    #             - Recursively visit 4 neighbors
    # Step 4: Count islands
    #         count = 0
    #         for i in range(m):
    #             for j in range(n):
    #                 if grid[i][j] == '1':
    #                     dfs(i, j)
    #                     count += 1
    # Step 5: Return count
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 2: BFS
# =============================================================================
def num_islands_bfs(grid: List[List[str]]) -> int:
    """
    Count islands using BFS.
    
    BFS uses a queue instead of recursion, which can be better for
    very large grids (no stack overflow risk).
    
    Algorithm:
    1. When we find land, add it to queue
    2. Process queue: for each land cell, add its land neighbors
    3. Mark cells as visited when adding to queue (not when processing!)
    
    Time: O(m * n)
    Space: O(min(m, n)) - queue size in worst case
    
    INTERVIEW TIP: BFS space is O(min(m,n)) because the queue only
    holds cells at the "frontier" of our search.
    """
    # TODO: Implement this function (20-25 lines)
    # Step 1: Handle edge case
    # Step 2: Define BFS helper:
    #         def bfs(start_i, start_j):
    #             queue = deque([(start_i, start_j)])
    #             grid[start_i][start_j] = '0'  # Mark visited immediately!
    #             while queue:
    #                 i, j = queue.popleft()
    #                 for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
    #                     ni, nj = i + di, j + dj
    #                     if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == '1':
    #                         queue.append((ni, nj))
    #                         grid[ni][nj] = '0'  # Mark visited immediately!
    # Step 3: Count islands
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 3: Union-Find (Disjoint Set)
# =============================================================================
class UnionFind:
    """
    Union-Find data structure with path compression and union by rank.
    
    Operations:
    - find(x): Find root of x's set with path compression
    - union(x, y): Unite sets containing x and y
    - count: Number of distinct sets
    
    INTERVIEW TIP: Union-Find is O(α(n)) per operation where α is the
    inverse Ackermann function - effectively O(1) for practical inputs.
    """
    
    def __init__(self, n: int):
        """Initialize n elements in separate sets."""
        # TODO: Implement (3 lines)
        # self.parent = list(range(n))  # Each element is its own parent
        # self.rank = [0] * n           # For union by rank
        # self.count = n                # Number of distinct sets
        pass
    
    def find(self, x: int) -> int:
        """Find root with path compression."""
        # TODO: Implement (4-6 lines)
        # Path compression: make each node point directly to root
        # if self.parent[x] != x:
        #     self.parent[x] = self.find(self.parent[x])
        # return self.parent[x]
        pass
    
    def union(self, x: int, y: int) -> None:
        """Unite sets containing x and y."""
        # TODO: Implement (8-12 lines)
        # 1. Find roots
        # 2. If same root, already united
        # 3. Union by rank: attach smaller tree under larger
        # 4. Decrement count
        pass


def num_islands_union_find(grid: List[List[str]]) -> int:
    """
    Count islands using Union-Find.
    
    Algorithm:
    1. Initialize Union-Find with all land cells
    2. For each land cell, union with adjacent land cells
    3. Return number of distinct sets (minus water cells)
    
    This approach is powerful when we need to track connectivity
    dynamically (e.g., as islands are added).
    
    Time: O(m * n * α(m * n)) ≈ O(m * n)
    Space: O(m * n)
    """
    # TODO: Implement this function (20-25 lines)
    # Step 1: Count land cells
    # Step 2: Create Union-Find with land cells only
    # Step 3: Union adjacent land cells
    # Step 4: Return count of distinct sets
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 4: Without Modifying Input
# =============================================================================
def num_islands_no_modify(grid: List[List[str]]) -> int:
    """
    Count islands WITHOUT modifying the input grid.
    
    Use a separate visited set instead of modifying grid.
    
    INTERVIEW TIP: Ask if you can modify the input! If not, use this approach.
    
    Time: O(m * n)
    Space: O(m * n) for visited set
    """
    # TODO: Implement this function (15-20 lines)
    # Use set visited instead of modifying grid
    
    pass  # Remove this and implement


# =============================================================================
# INTERVIEW PRACTICE
# =============================================================================
def num_islands_interview(grid: List[List[str]]) -> int:
    """
    INTERVIEW VERSION: Write this from scratch in 15-20 minutes.
    
    Steps:
    1. Clarify: Can we modify the grid? What defines an island?
    2. Explain approach: "Sink" islands as we find them
    3. Code clean DFS or BFS solution
    4. Discuss complexity
    
    Common follow-ups:
    - What if we can't modify the grid? (Use visited set)
    - What if grid is very large? (BFS to avoid stack overflow)
    - What if we need to handle dynamic updates? (Union-Find)
    """
    # TODO: Implement from scratch!
    
    pass  # Remove this and implement


# =============================================================================
# TESTS
# =============================================================================
class TestNumberOfIslands:
    """Tests for number of islands implementations."""
    
    def get_grid_1(self):
        """Grid with 1 island."""
        return [
            ["1", "1", "1", "1", "0"],
            ["1", "1", "0", "1", "0"],
            ["1", "1", "0", "0", "0"],
            ["0", "0", "0", "0", "0"]
        ]
    
    def get_grid_3(self):
        """Grid with 3 islands."""
        return [
            ["1", "1", "0", "0", "0"],
            ["1", "1", "0", "0", "0"],
            ["0", "0", "1", "0", "0"],
            ["0", "0", "0", "1", "1"]
        ]
    
    def test_dfs_one_island(self):
        """Test DFS with one island."""
        grid = self.get_grid_1()
        result = num_islands_dfs(grid)
        if result is not None:
            assert result == 1
    
    def test_dfs_three_islands(self):
        """Test DFS with three islands."""
        grid = self.get_grid_3()
        result = num_islands_dfs(grid)
        if result is not None:
            assert result == 3
    
    def test_dfs_empty_grid(self):
        """Test DFS with empty grid."""
        result = num_islands_dfs([])
        if result is not None:
            assert result == 0
    
    def test_dfs_all_water(self):
        """Test DFS with all water."""
        grid = [["0", "0"], ["0", "0"]]
        result = num_islands_dfs(grid)
        if result is not None:
            assert result == 0
    
    def test_dfs_all_land(self):
        """Test DFS with all land (one big island)."""
        grid = [["1", "1"], ["1", "1"]]
        result = num_islands_dfs(grid)
        if result is not None:
            assert result == 1
    
    def test_bfs_matches_dfs(self):
        """Test that BFS gives same result as DFS."""
        # Need fresh grids since we modify them
        grid_dfs = self.get_grid_3()
        grid_bfs = self.get_grid_3()
        
        dfs_result = num_islands_dfs(grid_dfs)
        bfs_result = num_islands_bfs(grid_bfs)
        
        if dfs_result is not None and bfs_result is not None:
            assert dfs_result == bfs_result
    
    def test_union_find_matches(self):
        """Test that Union-Find gives same result."""
        grid1 = self.get_grid_3()
        grid2 = self.get_grid_3()
        
        dfs_result = num_islands_dfs(grid1)
        uf_result = num_islands_union_find(grid2)
        
        if dfs_result is not None and uf_result is not None:
            assert dfs_result == uf_result
    
    def test_single_cell_island(self):
        """Test single cell island."""
        grid = [["1"]]
        result = num_islands_dfs(grid)
        if result is not None:
            assert result == 1
    
    def test_diagonal_not_connected(self):
        """Test that diagonal cells are NOT connected."""
        grid = [
            ["1", "0"],
            ["0", "1"]
        ]
        result = num_islands_dfs(grid)
        if result is not None:
            assert result == 2  # Two separate islands


if __name__ == "__main__":
    print("NUMBER OF ISLANDS - Practice Problem")
    print("=" * 50)
    
    # Example
    grid = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"]
    ]
    
    print("\nInput grid:")
    for row in grid:
        print("  " + " ".join(row))
    
    # Make a copy since DFS modifies the grid
    grid_copy = [row[:] for row in grid]
    result = num_islands_dfs(grid_copy)
    
    print(f"\nNumber of islands: {result if result is not None else '(not implemented)'}")
    print("Expected: 3")
    
    print("\n" + "=" * 50)
    print("KEY CONCEPTS:")
    print("1. Grid traversal using DFS/BFS")
    print("2. 'Sinking' islands to mark as visited")
    print("3. Alternative: Union-Find for dynamic connectivity")
    
    print("\nRun 'pytest number_of_islands.py -v' for tests")
