# 🌳 BFS, DFS & Tree Algorithms - Complete Master Guide
## Interactive Workbook for Offline Study

**Author:** Interview Prep Materials  
**Last Updated:** 2025  
**Estimated Study Time:** 8-10 hours

**This comprehensive guide covers:**
- BFS (Breadth-First Search) fundamentals & applications
- DFS (Depth-First Search) fundamentals & applications  
- Tree algorithms & traversal techniques
- Binary tree problem-solving patterns
- Advanced tree algorithms

---

# Table of Contents

1. [BFS Core Concepts](#bfs-core-concepts)
2. [BFS Exercises](#bfs-exercises)
3. [DFS Core Concepts](#dfs-core-concepts)
4. [DFS Exercises](#dfs-exercises)
5. [Tree Algorithms Fundamentals](#tree-algorithms-fundamentals)
6. [Tree Traversal Techniques](#tree-traversal-techniques)
7. [Tree Problem Patterns](#tree-problem-patterns)
8. [Advanced Tree Algorithms](#advanced-tree-algorithms)
9. [Comparison & Decision Making](#comparison--decision-making)
10. [Practice Problems](#practice-problems)
11. [Answer Key](#answer-key)

---

# Part 1: BFS Core Concepts

## 🌊 Understanding BFS: The Ripple Analogy

**Fill in the blanks:**

1. Imagine dropping a stone in a pond. BFS works like ripples:
   - First, the ripple touches everything **______ steps away**
   - Then, everything **______ steps away**
   - Then **______ steps away**
   - And so on...

2. BFS explores nodes **______ by ______**, in order of **______** from the start.

3. BFS is perfect for finding the **______** path in unweighted graphs.

4. BFS uses a **______** (FIFO - First In, First Out).

5. BFS explores **______** (level by level / as deep as possible).

## 📝 BFS Algorithm Steps

**Complete the steps:**

1. **Initialize:**
   - Create a **______** (queue)
   - Create a **______** (visited set)
   - Add the **______** node to both

2. **Main Loop:**
   - While queue is **______**:
     - **______** a node from the queue (this is the oldest node)
     - **______** the node (do your work here)
     - For each **______** of the current node:
       - If neighbor is **______** visited:
         - **______** it as visited (BEFORE adding to queue!)
         - Add it to the **______**

3. **Key Rule:** Always mark nodes as visited **______** (before/after) adding to the queue to avoid duplicates.

## 🆚 BFS vs DFS Comparison Table

**Fill in the table:**

| Aspect | BFS | DFS |
|--------|-----|-----|
| Data Structure | **______** | **______** |
| Exploration Order | **______** | **______** |
| Finds | **______** path | **______** path |
| Memory Usage | **______** | **______** |
| Good For | **______** | **______** |

---

# Part 2: BFS Exercises

## Exercise 1: Basic BFS Template - Fill in the Code

```python
def bfs_basic(graph: Dict, start) -> Set:
    """
    Basic BFS traversal of a graph.
    
    TODO: Fill in the missing code
    """
    visited = {start}
    queue = ________________  # Fill in: initialize queue with start
    
    while ________________:  # Fill in: condition
        node = queue.________________  # Fill in: remove node from queue
        
        # Process node here (optional)
        # ...
        
        for neighbor in graph[node]:
            if neighbor ________________ visited:  # Fill in: check condition
                visited.________________  # Fill in: mark as visited
                queue.________________  # Fill in: add to queue
    
    return visited
```

## Exercise 2: BFS with Level Tracking - Complete the Code

```python
def bfs_with_levels(graph: Dict, start) -> Dict:
    """
    BFS that tracks distance/level for each node.
    
    TODO: Complete the implementation
    """
    levels = {start: 0}
    queue = deque([start])
    
    while queue:
        node = ________________
        current_level = levels[node]
        
        for neighbor in graph[node]:
            if neighbor ________________ levels:
                levels[neighbor] = ________________  # Fill in: calculate level
                queue.________________  # Fill in: add to queue
    
    return levels
```

## Exercise 3: BFS on Grid - Complete the Code

```python
def bfs_grid(grid: List[List[int]], start: Tuple[int, int]) -> int:
    """
    BFS on a 2D grid to find shortest path.
    
    TODO: Fill in the missing parts
    """
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    visited = {start}
    queue = deque([(start[0], start[1], 0)])  # (row, col, distance)
    
    while queue:
        r, c, dist = ________________
        
        # Check if reached target (example: value == 9)
        if grid[r][c] == 9:
            return ________________
        
        # Explore neighbors
        for dr, dc in directions:
            nr, nc = ________________, ________________  # Fill in: new row, new col
            
            # Check bounds and validity
            if (0 <= nr < rows and 0 <= nc < cols and 
                (nr, nc) ________________ visited and 
                grid[nr][nc] != 0):  # 0 is obstacle
                
                visited.add((nr, nc))
                queue.append((________________, ________________, ________________))  # Fill in
    
    return -1  # Not found
```

## Exercise 4: BFS Shortest Path with Reconstruction

```python
def bfs_shortest_path(graph: Dict, start, end) -> List:
    """
    Find shortest path between start and end using BFS.
    
    TODO: Complete the path reconstruction
    """
    if start == end:
        return [start]
    
    visited = {start}
    queue = deque([start])
    parent = {}  # Track parent: {node: parent_node}
    
    while queue:
        current = queue.popleft()
        
        for neighbor in graph.get(current, []):
            if neighbor ________________ visited:
                visited.________________
                parent[neighbor] = ________________  # Fill in: track parent
                queue.________________
                
                if neighbor == end:
                    # Reconstruct path
                    path = []
                    node = end
                    while node ________________ None:  # Fill in: condition
                        path.append(node)
                        node = parent.________________  # Fill in: get parent
                    
                    return path[::-1]  # Reverse
    
    return []  # No path found
```

## Quiz 1: BFS Understanding

**Answer the following questions:**

1. **Why does BFS guarantee the shortest path in unweighted graphs?**
   
   Your answer:
   ___________________________________________________________________________
   ___________________________________________________________________________

2. **When should you mark a node as visited in BFS?**
   - [ ] Before adding to queue
   - [ ] After removing from queue
   - [ ] Both are correct
   - [ ] Neither matters

   Your answer: **______**

3. **What is the time complexity of BFS?**
   - [ ] O(V)
   - [ ] O(E)
   - [ ] O(V + E)
   - [ ] O(V × E)

   Your answer: **______**

4. **What is the space complexity of BFS?**
   - [ ] O(V)
   - [ ] O(E)
   - [ ] O(V + E)
   - [ ] O(V²)

   Your answer: **______**

5. **Draw the BFS traversal order for this graph starting at A:**
   ```
        A
       / \
      B   C
     / \ / \
    D   E   F
   ```
   
   Traversal order: **______ → ______ → ______ → ______ → ______ → ______**

---

# Part 3: DFS Core Concepts

## 🕳️ Understanding DFS: The Maze Explorer Analogy

**Fill in the blanks:**

1. DFS works like exploring a maze:
   - Pick a path and go as **______** as possible
   - When you hit a dead end, **______** to the last choice point
   - Keep going **______** until you've explored everything

2. DFS explores nodes by going **______** first, then **______**.

3. DFS is perfect for:
   - **______** all paths
   - **______** problems (permutations, combinations)
   - **______** cycles
   - **______** sorting

4. DFS uses a **______** (LIFO - Last In, First Out) or **______**.

5. DFS stores only the **______** path, making it more memory-efficient.

## 📝 DFS Algorithm Steps (Recursive)

**Complete the steps:**

1. **Base Case:**
   - If node is **______** or **______**: return

2. **Process Current Node:**
   - **______** node as visited
   - **______** the node (do your work here)

3. **Recursive Step:**
   - For each **______**:
     - If neighbor is **______** visited:
       - Recursively call **______** on neighbor

## 📝 DFS Algorithm Steps (Iterative)

**Complete the steps:**

1. **Initialize:**
   - Create a **______** (stack)
   - Create a **______** (visited set)
   - Push the **______** node

2. **Main Loop:**
   - While stack is **______**:
     - **______** a node from the stack
     - If node is **______** visited:
       - **______** it as visited
       - **______** the node
       - Push all **______** to the stack

## 🔑 Key DFS Patterns

**Match the pattern with its use case:**

1. **Inorder (Left → Root → Right)** → **______**
2. **Preorder (Root → Left → Right)** → **______**
3. **Postorder (Left → Right → Root)** → **______**

Choices:
- A. Copy/serialize tree
- B. BST gives sorted order
- C. Delete tree/calculate size

---

# Part 4: DFS Exercises

## Exercise 5: Recursive DFS Template - Fill in the Code

```python
def dfs_recursive(graph: Dict, node, visited: Set) -> None:
    """
    Recursive DFS traversal.
    
    TODO: Complete the implementation
    """
    # Base case
    if node ________________ visited:
        return
    
    # Mark as visited
    visited.________________
    
    # Process node (optional)
    # process(node)
    
    # Recursive call on neighbors
    for neighbor in graph.get(node, []):
        if neighbor ________________ visited:
            ________________  # Fill in: recursive call
```

## Exercise 6: Iterative DFS Template - Fill in the Code

```python
def dfs_iterative(graph: Dict, start) -> Set:
    """
    Iterative DFS using a stack.
    
    TODO: Complete the implementation
    """
    visited = set()
    stack = ________________  # Fill in: initialize stack with start
    
    while ________________:  # Fill in: condition
        node = stack.________________  # Fill in: remove from stack
        
        if node ________________ visited:  # Fill in: check
            visited.________________  # Fill in: mark as visited
            
            # Process node (optional)
            # process(node)
            
            # Add neighbors to stack
            for neighbor in graph.get(node, []):
                if neighbor ________________ visited:
                    stack.________________  # Fill in: add to stack
    
    return visited
```

## Exercise 7: DFS on Grid - Complete the Code

```python
def dfs_grid(grid: List[List[int]], r: int, c: int, visited: Set) -> int:
    """
    DFS on grid to count connected cells or find islands.
    
    TODO: Fill in the missing parts
    """
    rows, cols = len(grid), len(grid[0])
    
    # Base case: out of bounds or invalid
    if (r < 0 or r >= rows or 
        c < 0 or c >= cols or
        (r, c) ________________ visited or
        grid[r][c] == 0):  # 0 is water/obstacle
        return 0
    
    # Mark as visited
    visited.add((r, c))
    count = 1  # Count current cell
    
    # Explore all 4 directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        count += dfs_grid(grid, ________________, ________________, visited)  # Fill in
    
    return count
```

## Exercise 8: DFS with Path Tracking - Complete the Code

```python
def dfs_with_path(graph: Dict, start, end, path: List, visited: Set) -> bool:
    """
    DFS that tracks the path from start to end.
    
    TODO: Complete the path tracking
    """
    # Base case: reached target
    if start == end:
        path.append(end)
        return ________________  # Fill in
    
    # Mark as visited
    visited.add(start)
    path.append(start)
    
    # Explore neighbors
    for neighbor in graph.get(start, []):
        if neighbor ________________ visited:
            if dfs_with_path(________________, ________________, ________________, path, visited):  # Fill in
                return True
    
    # Backtrack: remove from path
    path.________________  # Fill in: remove last element
    return False
```

## Exercise 9: DFS Backtracking Template - Complete the Code

```python
def dfs_backtrack(candidates: List, path: List, result: List) -> None:
    """
    DFS backtracking template (for permutations/combinations).
    
    TODO: Complete the backtracking logic
    """
    # Base case: found a solution
    if is_complete(path):
        result.append(path[:])  # Make a copy
        return
    
    # Try all candidates
    for candidate in candidates:
        # Skip invalid candidates
        if not is_valid(candidate, path):
            continue
        
        # Choose
        path.append(candidate)
        
        # Explore
        dfs_backtrack(________________, ________________, ________________)  # Fill in
        
        # Unchoose (backtrack)
        path.________________  # Fill in: remove last element
```

## Exercise 10: DFS Cycle Detection - Complete the Code

```python
def has_cycle_directed(graph: Dict) -> bool:
    """
    Detect cycle in a directed graph using DFS.
    
    TODO: Complete the cycle detection logic
    Uses 3 states: WHITE (unvisited), GRAY (in current path), BLACK (finished)
    """
    # Color states: 0 = WHITE (unvisited), 1 = GRAY (visiting), 2 = BLACK (done)
    color = {node: 0 for node in graph}
    
    def dfs(node):
        if color[node] == 1:  # GRAY = cycle found!
            return ________________  # Fill in
        
        if color[node] == 2:  # BLACK = already processed
            return ________________  # Fill in
        
        # Mark as GRAY (in current path)
        color[node] = ________________  # Fill in
        
        # Explore neighbors
        for neighbor in graph.get(node, []):
            if dfs(neighbor):
                return ________________  # Fill in
        
        # Mark as BLACK (finished)
        color[node] = ________________  # Fill in
        return False
    
    # Check all nodes (for disconnected components)
    for node in graph:
        if color[node] == 0:  # WHITE
            if dfs(node):
                return True
    
    return False
```

## Exercise 11: DFS Topological Sort - Complete the Code

```python
def topological_sort(graph: Dict) -> List:
    """
    Topological sort using DFS.
    
    TODO: Complete the topological sort
    Returns nodes in order such that if u → v, then u comes before v.
    """
    visited = set()
    result = []
    
    def dfs(node):
        if node ________________ visited:  # Fill in: check if already in result
            return
        
        visited.add(node)
        
        # Explore dependencies (neighbors that must come AFTER)
        for neighbor in graph.get(node, []):
            if neighbor ________________ visited:  # Fill in
                ________________  # Fill in: recursive call
        
        # Add to result (postorder: after exploring dependencies)
        result.________________  # Fill in: add node
    
    # Process all nodes
    for node in graph:
        if node ________________ visited:  # Fill in
            dfs(node)
    
    return result[::-1]  # Reverse (we added in reverse order)
```

## Exercise 12: DFS Tree Traversals - Complete the Code

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """
    Inorder: Left → Root → Right
    
    TODO: Complete the inorder traversal
    """
    result = []
    
    def inorder(node):
        if node:
            inorder(________________)  # Fill in: traverse left
            result.append(________________)  # Fill in: process root
            inorder(________________)  # Fill in: traverse right
    
    inorder(root)
    return result

def preorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """
    Preorder: Root → Left → Right
    
    TODO: Complete the preorder traversal
    """
    result = []
    
    def preorder(node):
        if node:
            result.append(________________)  # Fill in: process root FIRST
            preorder(________________)  # Fill in: traverse left
            preorder(________________)  # Fill in: traverse right
    
    preorder(root)
    return result

def postorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """
    Postorder: Left → Right → Root
    
    TODO: Complete the postorder traversal
    """
    result = []
    
    def postorder(node):
        if node:
            postorder(________________)  # Fill in: traverse left
            postorder(________________)  # Fill in: traverse right
            result.append(________________)  # Fill in: process root LAST
    
    postorder(root)
    return result
```

## Exercise 13: DFS Generate Parentheses - Complete the Code

```python
def generate_parentheses(n: int) -> List[str]:
    """
    Generate all valid combinations of n pairs of parentheses using DFS backtracking.
    
    Example: n=2 → ["(())", "()()"]
    
    TODO: Complete the backtracking logic
    """
    result = []
    
    def dfs(open_count: int, close_count: int, current: str):
        # Base case: valid combination found
        if len(current) == ________________:  # Fill in: target length
            result.append(current)
            return
        
        # Can add opening parenthesis if we haven't used all
        if open_count < n:
            dfs(________________, ________________, ________________)  # Fill in: add '('
        
        # Can add closing parenthesis if we have more open than close
        if close_count < ________________:  # Fill in: condition
            dfs(________________, ________________, ________________)  # Fill in: add ')'
    
    dfs(0, 0, "")
    return result
```

## Exercise 14: DFS Word Search - Complete the Code

```python
def word_search(board: List[List[str]], word: str) -> bool:
    """
    Check if word exists in grid by moving adjacent (up, down, left, right).
    
    TODO: Complete the DFS backtracking logic
    """
    rows, cols = len(board), len(board[0])
    
    def dfs(r: int, c: int, index: int) -> bool:
        # Base case: found entire word
        if index == ________________:  # Fill in: target index
            return ________________  # Fill in
        
        # Out of bounds or doesn't match
        if (r < 0 or r >= rows or 
            c < 0 or c >= cols or
            board[r][c] != word[index]):
            return ________________  # Fill in
        
        # Mark as visited (use special char)
        temp = board[r][c]
        board[r][c] = ________________  # Fill in: mark visited
        
        # Explore 4 directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        found = False
        for dr, dc in directions:
            if dfs(________________, ________________, ________________):  # Fill in
                found = True
                break
        
        # Backtrack: restore original value
        board[r][c] = ________________  # Fill in: restore
        
        return found
    
    # Try starting from each cell
    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    
    return False
```

## Exercise 15: DFS Number of Islands - Complete the Code

```python
def num_islands(grid: List[List[str]]) -> int:
    """
    Count number of islands (connected '1's) using DFS.
    
    TODO: Complete the island counting logic
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r: int, c: int):
        # Base case: out of bounds or water
        if (r < 0 or r >= rows or 
            c < 0 or c >= cols or
            grid[r][c] != '1'):
            return
        
        # Mark as visited (sink the island)
        grid[r][c] = '0'
        
        # Explore all 4 directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            dfs(________________, ________________)  # Fill in
    
    # Find all islands
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += ________________  # Fill in
                dfs(r, c)
    
    return count
```

## Exercise 16: DFS Permutations - Complete the Code

```python
def permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all permutations using DFS backtracking.
    
    Example: [1,2] → [[1,2], [2,1]]
    
    TODO: Complete the backtracking logic
    """
    result = []
    
    def dfs(current: List[int], remaining: List[int]):
        # Base case: permutation complete
        if len(remaining) == 0:
            result.append(________________)  # Fill in: add copy of current
            return
        
        # Try each remaining number
        for i in range(len(remaining)):
            # Choose
            current.append(________________)  # Fill in: add number
            new_remaining = remaining[:i] + remaining[________________:]  # Fill in: remove chosen
            
            # Explore
            dfs(current, new_remaining)
            
            # Unchoose (backtrack)
            current.________________  # Fill in: remove last
    
    dfs([], nums)
    return result
```

## Quiz 2: DFS Understanding

**Answer the following questions:**

1. **When should you use DFS over BFS?**
   
   Your answer:
   ___________________________________________________________________________
   ___________________________________________________________________________

2. **What is the main difference between recursive and iterative DFS?**
   - [ ] Recursive uses a stack, iterative uses a queue
   - [ ] Recursive uses recursion stack, iterative uses explicit stack
   - [ ] They're completely different algorithms
   - [ ] Iterative is always better

   Your answer: **______**

3. **What is the time complexity of DFS?**
   - [ ] O(V)
   - [ ] O(E)
   - [ ] O(V + E)
   - [ ] O(V × E)

   Your answer: **______**

4. **When backtracking, what must you remember to do?**
   - [ ] Nothing special
   - [ ] Unmark as visited
   - [ ] Remove from path and unmark
   - [ ] Clear the entire path

   Your answer: **______**

5. **For tree traversal, which DFS order gives sorted order for BST?**
   - [ ] Preorder
   - [ ] Inorder
   - [ ] Postorder
   - [ ] Level-order

   Your answer: **______**

6. **What's the key difference between cycle detection in directed vs undirected graphs?**
   
   Your answer:
   ___________________________________________________________________________

7. **Why do we reverse the result in topological sort?**
   
   Your answer:
   ___________________________________________________________________________

8. **In DFS backtracking, why do we make a copy before adding to result?**
   - [ ] To save memory
   - [ ] Because lists are mutable and we'll backtrack
   - [ ] Because it's faster
   - [ ] It doesn't matter

   Your answer: **______**

---

## Quiz 3: DFS Advanced Concepts

**Answer these advanced DFS questions:**

1. **What are the three states used in cycle detection for directed graphs?**
   - [ ] Unvisited, Visiting, Visited
   - [ ] White, Gray, Black
   - [ ] 0, 1, 2
   - [ ] All of the above

   Your answer: **______**

2. **Why does topological sort use postorder traversal?**
   
   Your answer:
   ___________________________________________________________________________

3. **What happens if we mark visited nodes when popping from stack instead of pushing?**
   
   Your answer:
   ___________________________________________________________________________

4. **When would you use iterative DFS instead of recursive DFS?**
   
   Your answer:
   ___________________________________________________________________________

---

# Part 5: Comparison & Decision Making

## Exercise 10: Choose BFS or DFS

**For each problem, choose BFS or DFS and explain why:**

1. **Finding shortest path in unweighted graph**
   - [ ] BFS
   - [ ] DFS
   
   Why: _______________________________________________________________________

2. **Finding all permutations of a string**
   - [ ] BFS
   - [ ] DFS
   
   Why: _______________________________________________________________________

3. **Counting connected components in a graph**
   - [ ] BFS
   - [ ] DFS
   
   Why: _______________________________________________________________________

4. **Finding level-order traversal of a tree**
   - [ ] BFS
   - [ ] DFS
   
   Why: _______________________________________________________________________

5. **Detecting cycles in a directed graph**
   - [ ] BFS
   - [ ] DFS
   
   Why: _______________________________________________________________________

## Exercise 11: Problem Recognition

**Circle the keywords that indicate BFS or DFS:**

1. "Find the minimum number of steps"
   - BFS / DFS

2. "Generate all possible combinations"
   - BFS / DFS

3. "Count the number of islands"
   - BFS / DFS

4. "Print tree level by level"
   - BFS / DFS

5. "Find if a path exists (any path)"
   - BFS / DFS

---

# Part 6: Practice Problems

## Problem 1: Number of Islands (BFS or DFS)

**Given a 2D grid of '1's (land) and '0's (water), count the number of islands.**

**Your solution:**

```python
def num_islands(grid: List[List[str]]) -> int:
    """
    TODO: Implement using BFS or DFS
    """
    # Your code here:
    










```

**Which approach did you choose?** **______** (BFS/DFS)

**Why?** _______________________________________________________________________

## Problem 2: Word Ladder (BFS)

**Given two words and a dictionary, find the shortest transformation sequence.**

**Your solution:**

```python
def word_ladder(begin_word: str, end_word: str, word_list: List[str]) -> int:
    """
    TODO: Implement using BFS
    """
    # Your code here:
    










```

## Problem 3: Binary Tree Maximum Depth (DFS)

**Find the maximum depth of a binary tree.**

**Your solution:**

```python
def max_depth(root: Optional[TreeNode]) -> int:
    """
    TODO: Implement using DFS (recursive or iterative)
    """
    # Your code here:
    










```

## Problem 4: Course Schedule (DFS)

**Check if you can finish all courses given prerequisites (cycle detection).**

**Your solution:**

```python
def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    TODO: Implement using DFS for cycle detection
    """
    # Your code here:
    










```

## Problem 5: Maximum Depth of Binary Tree (DFS)

**Find the maximum depth of a binary tree using DFS.**

**Your solution:**

```python
def max_depth(root: Optional[TreeNode]) -> int:
    """
    TODO: Implement using DFS (recursive or iterative)
    """
    # Your code here:
    










```

## Problem 6: Same Tree (DFS)

**Check if two binary trees are the same.**

**Your solution:**

```python
def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    TODO: Implement using DFS
    """
    # Your code here:
    










```

## Problem 7: Invert Binary Tree (DFS)

**Invert a binary tree (swap left and right children for every node).**

**Your solution:**

```python
def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    TODO: Implement using DFS
    """
    # Your code here:
    










```

## Problem 8: Path Sum (DFS)

**Check if there exists a root-to-leaf path with sum equal to target.**

**Your solution:**

```python
def has_path_sum(root: Optional[TreeNode], target_sum: int) -> bool:
    """
    TODO: Implement using DFS
    """
    # Your code here:
    










```

## Problem 9: Validate Binary Search Tree (DFS)

**Check if a binary tree is a valid BST.**

**Your solution:**

```python
def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    TODO: Implement using DFS inorder traversal
    """
    # Your code here:
    










```

## Problem 10: Symmetric Tree (DFS)

**Check if a binary tree is symmetric (mirror of itself).**

**Your solution:**

```python
def is_symmetric(root: Optional[TreeNode]) -> bool:
    """
    TODO: Implement using DFS
    """
    # Your code here:
    










```

---

# Part 6: DFS Solved Problems - Learn from Solutions

## Solved Problem 1: Number of Islands (DFS)

**Problem:** Given a 2D grid of '1's (land) and '0's (water), count the number of islands.

**Solution:**

```python
def num_islands(grid: List[List[str]]) -> int:
    """
    Count islands using DFS flood fill.
    
    Strategy:
    1. Iterate through grid
    2. When we find '1', it's a new island
    3. Use DFS to mark all connected '1's as visited (sink them)
    4. Count total islands
    
    Time: O(m × n) where m, n are grid dimensions
    Space: O(m × n) for recursion stack (worst case: entire grid is land)
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        # Base case: out of bounds or water
        if (r < 0 or r >= rows or 
            c < 0 or c >= cols or
            grid[r][c] != '1'):
            return
        
        # Mark as visited (sink the island)
        grid[r][c] = '0'
        
        # Explore all 4 directions
        dfs(r - 1, c)  # up
        dfs(r + 1, c)  # down
        dfs(r, c - 1)  # left
        dfs(r, c + 1)  # right
    
    # Find all islands
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)  # Sink entire island
    
    return count

# Example:
# grid = [
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]
# Result: 1 island
```

**Key Points:**
- Use DFS to "sink" (mark as visited) all connected land cells
- Each DFS call from a '1' cell represents one island
- Mark visited by changing '1' to '0' (or use visited set)

## Solved Problem 2: Word Search (DFS Backtracking)

**Problem:** Given a 2D board and a word, find if the word exists by moving adjacent.

**Solution:**

```python
def exist(board: List[List[str]], word: str) -> bool:
    """
    Word search using DFS backtracking.
    
    Strategy:
    1. Try starting from each cell
    2. Use DFS to explore paths
    3. Mark visited cells temporarily (backtrack)
    4. If word found, return True
    
    Time: O(m × n × 4^L) where L is word length
    Space: O(L) for recursion stack
    """
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, index):
        # Base case: found entire word
        if index == len(word):
            return True
        
        # Out of bounds or doesn't match
        if (r < 0 or r >= rows or 
            c < 0 or c >= cols or
            board[r][c] != word[index]):
            return False
        
        # Mark as visited temporarily (use special char)
        temp = board[r][c]
        board[r][c] = '#'  # Mark visited
        
        # Explore 4 directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        found = False
        for dr, dc in directions:
            if dfs(r + dr, c + dc, index + 1):
                found = True
                break
        
        # Backtrack: restore original value
        board[r][c] = temp
        return found
    
    # Try starting from each cell
    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    
    return False

# Example:
# board = [
#   ["A","B","C","E"],
#   ["S","F","C","S"],
#   ["A","D","E","E"]
# ]
# word = "ABCCED"
# Result: True (path: A→B→C→C→E→D)
```

**Key Points:**
- Must backtrack: restore cell value after exploring
- Use temporary marker ('#') to prevent revisiting in current path
- Try starting from every cell

## Solved Problem 3: Generate Parentheses (DFS Backtracking)

**Problem:** Generate all combinations of n pairs of valid parentheses.

**Solution:**

```python
def generate_parenthesis(n: int) -> List[str]:
    """
    Generate valid parentheses using DFS backtracking.
    
    Strategy:
    1. Track open_count and close_count
    2. Can add '(' if open_count < n
    3. Can add ')' if close_count < open_count
    4. When length = 2*n, add to result
    
    Time: O(4^n / sqrt(n)) - Catalan number
    Space: O(4^n / sqrt(n)) for result + O(n) for recursion
    """
    result = []
    
    def dfs(open_count, close_count, current):
        # Base case: valid combination found
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Can add opening parenthesis
        if open_count < n:
            dfs(open_count + 1, close_count, current + '(')
        
        # Can add closing parenthesis (must have more opens)
        if close_count < open_count:
            dfs(open_count, close_count + 1, current + ')')
    
    dfs(0, 0, "")
    return result

# Example: n = 3
# Result: ["((()))", "(()())", "(())()", "()(())", "()()()"]
```

**Key Points:**
- Two constraints: open_count ≤ n, close_count ≤ open_count
- Track counts, not just string length
- When both counts = n, we have valid combination

## Solved Problem 4: Course Schedule (DFS Cycle Detection)

**Problem:** Check if you can finish all courses given prerequisites (no cycles).

**Solution:**

```python
def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    Check for cycles using DFS with color states.
    
    Strategy:
    1. Build graph: course → list of prerequisites
    2. Use DFS with 3 states: WHITE(0), GRAY(1), BLACK(2)
    3. GRAY = in current path, BLACK = processed
    4. If we find GRAY node, cycle detected!
    
    Time: O(V + E)
    Space: O(V + E) for graph + recursion
    """
    # Build graph: course → [prerequisites]
    graph = {i: [] for i in range(num_courses)}
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # 0 = WHITE (unvisited), 1 = GRAY (visiting), 2 = BLACK (done)
    color = {i: 0 for i in range(num_courses)}
    
    def dfs(course):
        if color[course] == 1:  # GRAY = cycle!
            return True
        if color[course] == 2:  # BLACK = already processed
            return False
        
        # Mark as GRAY (in current path)
        color[course] = 1
        
        # Check prerequisites
        for prereq in graph[course]:
            if dfs(prereq):  # Cycle found
                return True
        
        # Mark as BLACK (finished)
        color[course] = 2
        return False
    
    # Check all courses
    for course in range(num_courses):
        if color[course] == 0:
            if dfs(course):
                return False  # Cycle found = can't finish
    
    return True  # No cycles = can finish

# Example:
# num_courses = 2
# prerequisites = [[1,0]]
# Result: True (0 → 1, no cycle)
#
# prerequisites = [[1,0], [0,1]]
# Result: False (0 → 1 → 0, cycle!)
```

**Key Points:**
- Three color states for cycle detection in directed graphs
- GRAY = in current recursion path (back edge = cycle!)
- BLACK = completely processed (skip in future)

## Solved Problem 5: Maximum Depth of Binary Tree (DFS)

**Problem:** Find maximum depth of binary tree.

**Solution:**

```python
def max_depth(root: Optional[TreeNode]) -> int:
    """
    Maximum depth using DFS postorder.
    
    Strategy:
    1. If None, depth = 0
    2. Calculate depth of left subtree
    3. Calculate depth of right subtree
    4. Return 1 + max(left_depth, right_depth)
    
    Why postorder? Need children's depths before calculating parent's!
    
    Time: O(n) - visit each node once
    Space: O(h) - recursion stack, h = height
    """
    if not root:
        return 0
    
    # Postorder: process children first
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    # Process root: 1 (current) + max of children
    return 1 + max(left_depth, right_depth)

# Example:
#     3
#    / \
#   9  20
#     /  \
#    15   7
# Result: 3
```

**Key Points:**
- Use postorder: need children's depths first
- Base case: None node has depth 0
- Return 1 + max(left, right)

## Solved Problem 6: Same Tree (DFS)

**Problem:** Check if two binary trees are identical.

**Solution:**

```python
def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Check if trees are same using DFS.
    
    Strategy:
    1. Both None? → Same
    2. One None? → Different
    3. Values different? → Different
    4. Recursively check left and right subtrees
    
    Time: O(min(m, n)) where m, n are number of nodes
    Space: O(min(m, n)) for recursion
    """
    # Both None
    if not p and not q:
        return True
    
    # One is None
    if not p or not q:
        return False
    
    # Values different
    if p.val != q.val:
        return False
    
    # Recursively check subtrees
    return (is_same_tree(p.left, q.left) and 
            is_same_tree(p.right, q.right))
```

**Key Points:**
- Handle None cases first
- Check value equality
- Recursively check both subtrees

## Solved Problem 7: Validate Binary Search Tree (DFS Inorder)

**Problem:** Check if binary tree is valid BST.

**Solution:**

```python
def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    Validate BST using DFS inorder traversal.
    
    Strategy:
    Inorder traversal of BST gives sorted order!
    If inorder traversal is not strictly increasing → not BST
    
    Time: O(n)
    Space: O(h) for recursion
    """
    prev = None
    
    def inorder(node):
        nonlocal prev
        if not node:
            return True
        
        # Traverse left
        if not inorder(node.left):
            return False
        
        # Process root: check if value > previous
        if prev is not None and node.val <= prev:
            return False
        prev = node.val
        
        # Traverse right
        return inorder(node.right)
    
    return inorder(root)

# Alternative: Pass min/max bounds
def is_valid_bst_bounds(root: Optional[TreeNode]) -> bool:
    def dfs(node, min_val, max_val):
        if not node:
            return True
        
        # Check bounds
        if node.val <= min_val or node.val >= max_val:
            return False
        
        # Recursively check with updated bounds
        return (dfs(node.left, min_val, node.val) and
                dfs(node.right, node.val, max_val))
    
    return dfs(root, float('-inf'), float('inf'))
```

**Key Points:**
- Inorder traversal gives sorted order for BST
- Can use min/max bounds approach too
- Must be strictly increasing (no duplicates allowed)

---

# Part 5: Tree Algorithms Fundamentals

## 🌳 Understanding Binary Trees

**Fill in the blanks:**

1. A binary tree is a tree data structure where each node has at most **______** children.

2. Tree terminology:
   - **Root:** The **______** node (has no parent)
   - **Leaf:** Node with **______** children
   - **Depth:** Number of edges from **______** to node
   - **Height:** Maximum **______** in the tree
   - **Level:** **______** + 1 (root is level 1)

3. Tree properties:
   - **Full Binary Tree:** Every node has **______** or **______** children
   - **Complete Binary Tree:** All levels filled except possibly last, filled **______**
   - **Perfect Binary Tree:** All levels are **______** filled
   - **Balanced Binary Tree:** Height difference between subtrees ≤ **______**

## 📊 Tree Representation

**Complete the code:**

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = ________________  # Fill in
        self.right = ________________  # Fill in
```

---

# Part 6: Tree Traversal Techniques (BFS & DFS)

## Exercise 17: Binary Tree Level Order (BFS) - Complete the Code

```python
def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Level-order traversal using BFS (each level as separate list).
    
    TODO: Complete the BFS level-order traversal
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = ________________  # Fill in: nodes at current level
        current_level = []
        
        # Process all nodes at current level
        for _ in range(level_size):
            node = queue.________________  # Fill in: remove from queue
            current_level.append(node.val)
            
            # Add children to queue (next level)
            if node.left:
                queue.________________  # Fill in
            if node.right:
                queue.________________  # Fill in
        
        result.append(current_level)
    
    return result
```

## Exercise 18: Binary Tree Zigzag Level Order (BFS) - Complete the Code

```python
def zigzag_level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Zigzag level-order: alternate left-to-right and right-to-left.
    
    TODO: Complete the zigzag traversal
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if left_to_right:
                current_level.________________  # Fill in: append
            else:
                current_level.________________  # Fill in: insert at start
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
        left_to_right = ________________  # Fill in: toggle direction
    
    return result
```

## Exercise 19: Binary Tree Right Side View (BFS) - Complete the Code

```python
def right_side_view(root: Optional[TreeNode]) -> List[int]:
    """
    Return values of nodes visible from right side.
    
    TODO: Complete the right side view using BFS
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Rightmost node at this level
            if i == ________________:  # Fill in: last node condition
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result
```

## Exercise 20: Binary Tree Maximum Depth (DFS Postorder) - Complete the Code

```python
def max_depth(root: Optional[TreeNode]) -> int:
    """
    Maximum depth using DFS postorder.
    
    TODO: Complete the depth calculation
    """
    if not root:
        return ________________  # Fill in: base case
    
    # Postorder: process children first
    left_depth = max_depth(________________)  # Fill in
    right_depth = max_depth(________________)  # Fill in
    
    # Process root: 1 (current) + max of children
    return ________________ + ________________(left_depth, right_depth)  # Fill in
```

## Exercise 21: Binary Tree Minimum Depth (BFS) - Complete the Code

```python
def min_depth(root: Optional[TreeNode]) -> int:
    """
    Minimum depth using BFS (shortest path to leaf).
    
    TODO: Complete the BFS minimum depth
    """
    if not root:
        return 0
    
    queue = deque([(root, 1)])  # (node, depth)
    
    while queue:
        node, depth = queue.________________  # Fill in
        
        # Found a leaf (minimum depth)
        if not node.left and not node.right:
            return ________________  # Fill in
        
        if node.left:
            queue.append((________________, ________________))  # Fill in
        if node.right:
            queue.append((________________, ________________))  # Fill in
    
    return 0
```

## Exercise 22: Invert Binary Tree (DFS) - Complete the Code

```python
def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Invert binary tree (swap left and right children).
    
    TODO: Complete the inversion using DFS
    """
    if not root:
        return None
    
    # Swap children
    root.left, root.right = ________________, ________________  # Fill in
    
    # Recursively invert subtrees
    invert_tree(root.left)
    invert_tree(root.right)
    
    return root
```

## Exercise 23: Same Tree (DFS) - Complete the Code

```python
def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Check if two trees are identical.
    
    TODO: Complete the comparison logic
    """
    # Both None
    if ________________:  # Fill in: both None condition
        return True
    
    # One is None
    if ________________ or ________________:  # Fill in: one None condition
        return False
    
    # Values different
    if p.val != q.val:
        return False
    
    # Recursively check subtrees
    return (is_same_tree(________________, ________________) and  # Fill in
            is_same_tree(________________, ________________))  # Fill in
```

## Exercise 24: Symmetric Tree (DFS) - Complete the Code

```python
def is_symmetric(root: Optional[TreeNode]) -> bool:
    """
    Check if tree is symmetric (mirror of itself).
    
    TODO: Complete the symmetric check
    """
    if not root:
        return True
    
    def is_mirror(left, right):
        # Both None
        if not left and not right:
            return True
        
        # One is None
        if not left or not right:
            return False
        
        # Values must match and subtrees must be mirror
        return (left.val == right.val and
                is_mirror(left.left, ________________) and  # Fill in
                is_mirror(left.right, ________________))  # Fill in
    
    return is_mirror(________________, ________________)  # Fill in
```

---

# Part 7: Tree Problem Patterns

## Exercise 25: Path Sum (DFS) - Complete the Code

```python
def has_path_sum(root: Optional[TreeNode], target_sum: int) -> bool:
    """
    Check if there exists a root-to-leaf path with sum equal to target.
    
    TODO: Complete the path sum check
    """
    if not root:
        return False
    
    # Leaf node: check if sum matches
    if not root.left and not root.right:
        return target_sum == ________________  # Fill in
    
    # Recursively check left and right subtrees
    remaining = target_sum - ________________  # Fill in: subtract current value
    return (has_path_sum(root.left, ________________) or  # Fill in
            has_path_sum(root.right, ________________))  # Fill in
```

## Exercise 26: All Paths from Root to Leaves (DFS Backtracking) - Complete the Code

```python
def binary_tree_paths(root: Optional[TreeNode]) -> List[str]:
    """
    Return all root-to-leaf paths.
    
    Example: ["1->2->5", "1->3"]
    
    TODO: Complete the path collection
    """
    result = []
    
    def dfs(node, path):
        if not node:
            return
        
        # Add current node to path
        path.append(str(node.val))
        
        # Leaf node: add path to result
        if not node.left and not node.right:
            result.append("->".join(________________))  # Fill in
        else:
            # Continue DFS
            dfs(node.left, path)
            dfs(node.right, path)
        
        # Backtrack: remove current node
        path.________________  # Fill in
    
    dfs(root, [])
    return result
```

## Exercise 27: Path Sum II (DFS Backtracking) - Complete the Code

```python
def path_sum(root: Optional[TreeNode], target_sum: int) -> List[List[int]]:
    """
    Return all root-to-leaf paths where sum equals target.
    
    TODO: Complete the backtracking logic
    """
    result = []
    
    def dfs(node, remaining, path):
        if not node:
            return
        
        # Add current node
        path.append(node.val)
        remaining -= node.val
        
        # Leaf node: check if sum matches
        if not node.left and not node.right and remaining == 0:
            result.append(________________)  # Fill in: add copy of path
        
        # Continue DFS
        dfs(node.left, remaining, path)
        dfs(node.right, remaining, path)
        
        # Backtrack
        path.________________  # Fill in
    
    dfs(root, target_sum, [])
    return result
```

## Exercise 28: Lowest Common Ancestor (DFS) - Complete the Code

```python
def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Find lowest common ancestor of two nodes.
    
    Strategy:
    - If both nodes in left subtree → LCA in left
    - If both nodes in right subtree → LCA in right
    - Otherwise, root is LCA
    
    TODO: Complete the LCA logic
    """
    if not root or root == p or root == q:
        return ________________  # Fill in
    
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    # Both found in different subtrees → root is LCA
    if left and right:
        return ________________  # Fill in
    
    # Both in same subtree
    return ________________ or ________________  # Fill in: left or right
```

## Exercise 29: Construct Binary Tree from Preorder and Inorder (DFS) - Complete the Code

```python
def build_tree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Build tree from preorder and inorder traversals.
    
    Strategy:
    1. Preorder[0] is root
    2. Find root in inorder → split into left and right subtrees
    3. Recursively build subtrees
    
    TODO: Complete the tree construction
    """
    if not preorder or not inorder:
        return None
    
    # Root is first in preorder
    root_val = preorder[0]
    root = TreeNode(root_val)
    
    # Find root in inorder
    root_index = inorder.index(________________)  # Fill in
    
    # Split inorder into left and right subtrees
    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]
    
    # Split preorder (skip root)
    left_preorder = preorder[1:1 + len(left_inorder)]
    right_preorder = preorder[1 + len(left_inorder):]
    
    # Recursively build subtrees
    root.left = build_tree(________________, ________________)  # Fill in
    root.right = build_tree(________________, ________________)  # Fill in
    
    return root
```

## Exercise 30: Serialize and Deserialize Binary Tree (DFS Preorder) - Complete the Code

```python
def serialize(root: Optional[TreeNode]) -> str:
    """
    Serialize tree to string using DFS preorder.
    
    TODO: Complete the serialization
    """
    result = []
    
    def dfs(node):
        if not node:
            result.append("null")
        else:
            result.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
    
    dfs(root)
    return ",".join(result)

def deserialize(data: str) -> Optional[TreeNode]:
    """
    Deserialize string to tree.
    
    TODO: Complete the deserialization
    """
    values = data.split(",")
    index = 0
    
    def dfs():
        nonlocal index
        if index >= len(values) or values[index] == "null":
            index += 1
            return None
        
        root = TreeNode(int(values[index]))
        index += 1
        root.left = dfs()
        root.right = dfs()
        return root
    
    return dfs()
```

---

# Part 8: Advanced Tree Algorithms

## Exercise 31: Validate Binary Search Tree (DFS Inorder) - Complete the Code

```python
def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    Validate BST using inorder traversal.
    
    Strategy: Inorder traversal of BST gives sorted order!
    
    TODO: Complete the validation
    """
    prev = None
    
    def inorder(node):
        nonlocal prev
        if not node:
            return True
        
        # Traverse left
        if not inorder(node.left):
            return False
        
        # Process root: check if value > previous
        if prev is not None and node.val <= prev:
            return ________________  # Fill in
        prev = node.val
        
        # Traverse right
        return inorder(node.right)
    
    return inorder(root)
```

## Exercise 32: Kth Smallest Element in BST (DFS Inorder) - Complete the Code

```python
def kth_smallest(root: Optional[TreeNode], k: int) -> int:
    """
    Find kth smallest element in BST.
    
    Strategy: Inorder traversal gives sorted order → kth element = answer
    
    TODO: Complete the kth smallest search
    """
    count = 0
    result = None
    
    def inorder(node):
        nonlocal count, result
        if not node:
            return
        
        inorder(node.left)
        
        count += 1
        if count == ________________:  # Fill in: kth element
            result = node.val
            return
        
        inorder(node.right)
    
    inorder(root)
    return result
```

## Exercise 33: Count Complete Tree Nodes (DFS with Optimization) - Complete the Code

```python
def count_nodes(root: Optional[TreeNode]) -> int:
    """
    Count nodes in complete binary tree efficiently.
    
    Strategy: Check if left and right heights are equal
    
    TODO: Complete the optimized count
    """
    if not root:
        return 0
    
    # Calculate left and right heights
    left_height = 0
    right_height = 0
    
    node = root.left
    while node:
        left_height += 1
        node = node.left
    
    node = root.right
    while node:
        right_height += 1
        node = node.right
    
    # If heights equal, tree is perfect (use formula)
    if left_height == right_height:
        return (2 ** (left_height + 1)) - 1  # 2^height - 1
    
    # Otherwise, recursively count
    return 1 + count_nodes(________________) + count_nodes(________________)  # Fill in
```

## Exercise 34: Binary Tree Maximum Path Sum (DFS) - Complete the Code

```python
def max_path_sum(root: Optional[TreeNode]) -> int:
    """
    Find maximum path sum (path can start and end anywhere).
    
    Strategy:
    - For each node, calculate max path through it
    - Path can go through node: node.val + left_max + right_max
    - Return max contribution from subtree: node.val + max(left, right)
    
    TODO: Complete the max path sum
    """
    max_sum = float('-inf')
    
    def dfs(node):
        nonlocal max_sum
        if not node:
            return 0
        
        # Max contributions from left and right subtrees (can be negative)
        left_max = max(0, dfs(node.left))
        right_max = max(0, dfs(node.right))
        
        # Max path through current node
        current_max = node.val + left_max + right_max
        max_sum = max(max_sum, ________________)  # Fill in
        
        # Return max contribution from this subtree
        return node.val + max(________________, ________________)  # Fill in
    
    dfs(root)
    return max_sum
```

## Exercise 35: Diameter of Binary Tree (DFS Postorder) - Complete the Code

```python
def diameter_of_binary_tree(root: Optional[TreeNode]) -> int:
    """
    Find diameter (longest path between any two nodes).
    
    Strategy: Diameter = max(left_height + right_height + 1) for all nodes
    
    TODO: Complete the diameter calculation
    """
    max_diameter = 0
    
    def dfs(node):
        nonlocal max_diameter
        if not node:
            return 0
        
        # Get heights of subtrees
        left_height = dfs(node.left)
        right_height = dfs(node.right)
        
        # Update diameter (path through current node)
        max_diameter = max(max_diameter, ________________)  # Fill in
        
        # Return height of subtree
        return 1 + max(________________, ________________)  # Fill in
    
    dfs(root)
    return max_diameter
```

---

# Part 9: Comparison & Decision Making

## BFS Core Concepts Answers

1. 1, 2, 3
2. level by level, distance
3. shortest
4. queue
5. level by level

## BFS Algorithm Steps Answers

1. queue, visited set, start
2. not empty, dequeue/popleft, process, neighbor, not, mark, queue
3. before

## BFS vs DFS Comparison Answers

| Aspect | BFS | DFS |
|--------|-----|-----|
| Data Structure | Queue | Stack |
| Exploration Order | Level by level | As deep as possible |
| Finds | Shortest path | A path |
| Memory Usage | More (stores level) | Less (stores path) |
| Good For | Shortest path, level-order | Backtracking, all paths |

## Exercise 1 Answers

```python
queue = deque([start])
while queue:
    node = queue.popleft()
    if neighbor not in visited:
        visited.add(neighbor)
        queue.append(neighbor)
```

## Exercise 2 Answers

```python
node = queue.popleft()
if neighbor not in levels:
    levels[neighbor] = current_level + 1
    queue.append(neighbor)
```

## Exercise 3 Answers

```python
r, c, dist = queue.popleft()
return dist
nr, nc = r + dr, c + dc
if (nr, nc) not in visited:
    queue.append((nr, nc, dist + 1))
```

## Exercise 4 Answers

```python
if neighbor not in visited:
    visited.add(neighbor)
    parent[neighbor] = current
    queue.append(neighbor)
    while node is not None:
        node = parent.get(node)
```

## Quiz 1 Answers

1. BFS explores level by level, so the first time we reach a node is via the shortest path.
2. Before adding to queue
3. O(V + E)
4. O(V)
5. A → B → C → D → E → F

## DFS Core Concepts Answers

1. deep, backtrack, deep
2. deep, backtracking
3. Exploring, Backtracking, Detecting, Topological
4. stack, recursion
5. current

## DFS Algorithm Steps Answers

**Recursive:**
1. None, visited
2. Mark, process
3. neighbor, not, DFS

**Iterative:**
1. stack, visited set, start
2. not empty, pop, not, mark, process, neighbors

## DFS Patterns Answers

1. B (BST gives sorted order)
2. A (Copy/serialize tree)
3. C (Delete tree/calculate size)

## Exercise 5-9 Answers

**Exercise 5:**
```python
if node in visited:
    return
visited.add(node)
if neighbor not in visited:
    dfs_recursive(graph, neighbor, visited)
```

**Exercise 6:**
```python
stack = [start]
while stack:
    node = stack.pop()
    if node not in visited:
        visited.add(node)
        if neighbor not in visited:
            stack.append(neighbor)
```

**Exercise 7:**
```python
if (r, c) in visited or:
    return 0
count += dfs_grid(grid, r + dr, c + dc, visited)
```

**Exercise 8:**
```python
return True
if neighbor not in visited:
    if dfs_with_path(graph, neighbor, end, path, visited):
        return True
path.pop()
```

**Exercise 9:**
```python
dfs_backtrack(candidates, path, result)
path.pop()
```

## Exercise 10-16 Answers (New DFS Exercises)

**Exercise 10: Cycle Detection**
```python
if color[node] == 1:  # GRAY = cycle found!
    return True
if color[node] == 2:  # BLACK = already processed
    return False
color[node] = 1  # Mark as GRAY
if dfs(neighbor):
    return True
color[node] = 2  # Mark as BLACK
```

**Exercise 11: Topological Sort**
```python
if node in visited:  # Already in result
    return
visited.add(node)
if neighbor not in visited:
    dfs(neighbor)
result.append(node)  # Postorder: add after exploring dependencies
```

**Exercise 12: Tree Traversals**
```python
# Inorder: Left → Root → Right
inorder(node.left)
result.append(node.val)
inorder(node.right)

# Preorder: Root → Left → Right
result.append(node.val)
preorder(node.left)
preorder(node.right)

# Postorder: Left → Right → Root
postorder(node.left)
postorder(node.right)
result.append(node.val)
```

**Exercise 13: Generate Parentheses**
```python
if len(current) == 2 * n:  # Target length
    result.append(current)
    return
if open_count < n:
    dfs(open_count + 1, close_count, current + '(')
if close_count < open_count:  # Can close if more opens than closes
    dfs(open_count, close_count + 1, current + ')')
```

**Exercise 14: Word Search**
```python
if index == len(word):  # Found entire word
    return True
return False  # Out of bounds or doesn't match
board[r][c] = '#'  # Mark visited
if dfs(r + dr, c + dc, index + 1):  # Explore with next index
    found = True
    break
board[r][c] = temp  # Restore (backtrack)
```

**Exercise 15: Number of Islands**
```python
dfs(r + dr, c + dc)  # Explore neighbors
count += 1  # New island found
```

**Exercise 16: Permutations**
```python
result.append(current[:])  # Make a copy
current.append(remaining[i])  # Add chosen number
new_remaining = remaining[:i] + remaining[i+1:]  # Remove chosen
current.pop()  # Backtrack
```

## Quiz 2 Answers

1. When you need to explore all paths, backtrack, or detect cycles
2. Recursive uses recursion stack, iterative uses explicit stack
3. O(V + E)
4. Remove from path and unmark
5. Inorder
6. Directed graphs need 3 states (WHITE/GRAY/BLACK) to detect back edges. Undirected graphs can use 2 states (visited/unvisited) and check if neighbor is parent.
7. We add nodes in postorder (after exploring dependencies), so they're added in reverse topological order. We reverse to get correct order.
8. Because lists are mutable and we'll backtrack, modifying the list. Without copy, all entries would reference the same modified list.

## Quiz 3 Answers

1. All of the above - they're the same concepts with different names
2. Postorder ensures we process dependencies before dependents. After exploring all dependencies (neighbors), we add the node, ensuring it comes after all its dependencies.
3. Nodes can be added multiple times to the stack, causing duplicates and potential infinite loops in cyclic graphs.
4. When recursion depth might be too deep (stack overflow), or when you need more control over the traversal order.

## Exercise 10 Answers

1. BFS - finds shortest path
2. DFS - backtracking problem
3. Either works, but DFS is simpler
4. BFS - level by level
5. DFS - easier to detect back edges

## Exercise 11 Answers

1. BFS
2. DFS
3. Either (DFS is common)
4. BFS
5. Either (DFS is simpler)

## Exercise 17-35 Answers (Tree Algorithms)

**Exercise 17: Level Order**
```python
level_size = len(queue)
node = queue.popleft()
queue.append(node.left)
queue.append(node.right)
```

**Exercise 18: Zigzag Level Order**
```python
current_level.append(node.val)  # left to right
current_level.insert(0, node.val)  # right to left
left_to_right = not left_to_right  # toggle
```

**Exercise 19: Right Side View**
```python
if i == level_size - 1:  # last node at level
```

**Exercise 20: Maximum Depth**
```python
return 0  # base case
left_depth = max_depth(root.left)
right_depth = max_depth(root.right)
return 1 + max(left_depth, right_depth)
```

**Exercise 21: Minimum Depth (BFS)**
```python
node, depth = queue.popleft()
return depth
queue.append((node.left, depth + 1))
queue.append((node.right, depth + 1))
```

**Exercise 22: Invert Tree**
```python
root.left, root.right = root.right, root.left
```

**Exercise 23: Same Tree**
```python
if not p and not q:  # both None
if not p or not q:  # one None
is_same_tree(p.left, q.left)
is_same_tree(p.right, q.right)
```

**Exercise 24: Symmetric Tree**
```python
is_mirror(left.left, right.right)  # outer
is_mirror(left.right, right.left)  # inner
return is_mirror(root.left, root.right)
```

**Exercise 25: Path Sum**
```python
return target_sum == root.val
remaining = target_sum - root.val
has_path_sum(root.left, remaining)
has_path_sum(root.right, remaining)
```

**Exercise 26: Binary Tree Paths**
```python
result.append("->".join(path))
path.pop()
```

**Exercise 27: Path Sum II**
```python
result.append(path[:])  # copy of path
path.pop()
```

**Exercise 28: Lowest Common Ancestor**
```python
return root  # base case
return root  # both found in different subtrees
return left or right  # both in same subtree
```

**Exercise 29: Build Tree from Preorder/Inorder**
```python
root_index = inorder.index(root_val)
root.left = build_tree(left_preorder, left_inorder)
root.right = build_tree(right_preorder, right_inorder)
```

**Exercise 30: Serialize/Deserialize**
```python
# Already complete in template - just review the structure!
```

**Exercise 31: Validate BST**
```python
return False  # invalid BST
```

**Exercise 32: Kth Smallest**
```python
if count == k:  # kth element found
```

**Exercise 33: Count Complete Tree Nodes**
```python
return 1 + count_nodes(root.left) + count_nodes(root.right)
```

**Exercise 34: Maximum Path Sum**
```python
max_sum = max(max_sum, current_max)
return node.val + max(left_max, right_max)
```

**Exercise 35: Diameter**
```python
max_diameter = max(max_diameter, left_height + right_height)
return 1 + max(left_height, right_height)
```

---

# Study Checklist

Use this checklist to track your progress:

## BFS Mastery
- [ ] Can explain BFS intuitively (ripple analogy)
- [ ] Can write basic BFS from memory
- [ ] Can implement BFS with level tracking
- [ ] Can implement BFS on a grid
- [ ] Can find shortest path with BFS
- [ ] Can identify BFS problems in interviews

## DFS Mastery
- [ ] Can explain DFS intuitively (maze analogy)
- [ ] Can write recursive DFS from memory
- [ ] Can write iterative DFS from memory
- [ ] Can implement DFS on a grid
- [ ] Can implement DFS with backtracking
- [ ] Can detect cycles in directed graphs (3-state DFS)
- [ ] Can implement topological sort using DFS
- [ ] Can do tree traversals (inorder, preorder, postorder)
- [ ] Can solve backtracking problems (permutations, combinations)
- [ ] Can solve grid backtracking (word search)
- [ ] Can identify DFS problems in interviews

## Tree Algorithms Mastery
- [ ] Understand binary tree structure and terminology
- [ ] Can implement level-order traversal (BFS)
- [ ] Can implement inorder, preorder, postorder (DFS)
- [ ] Can find maximum depth (DFS postorder)
- [ ] Can find minimum depth (BFS)
- [ ] Can invert a binary tree
- [ ] Can check if trees are same/symmetric
- [ ] Can solve path sum problems
- [ ] Can find all root-to-leaf paths
- [ ] Can find lowest common ancestor
- [ ] Can construct tree from traversals
- [ ] Can serialize/deserialize tree
- [ ] Can validate BST
- [ ] Can find kth smallest in BST
- [ ] Can find maximum path sum
- [ ] Can find diameter of tree

## Problem Solving
- [ ] Can choose between BFS and DFS for a problem
- [ ] Can implement Number of Islands (DFS)
- [ ] Can implement Word Search (DFS backtracking)
- [ ] Can implement Word Ladder (BFS)
- [ ] Can implement tree traversals (inorder, preorder, postorder)
- [ ] Can detect cycles with DFS (directed graphs)
- [ ] Can implement topological sort
- [ ] Can solve Generate Parentheses (DFS backtracking)
- [ ] Can solve tree problems (max depth, same tree, BST validation)
- [ ] Can implement permutations/combinations using DFS

---

**Good luck with your interview prep! 🚀**

*Remember: Practice implementing these from memory until it becomes muscle memory.*
