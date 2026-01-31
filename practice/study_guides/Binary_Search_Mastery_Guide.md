# 🔍 Binary Search - Complete Mastery Guide
## Interactive Workbook for Offline Study

**Author:** Interview Prep Materials  
**Last Updated:** 2025  
**Estimated Study Time:** 6-8 hours

**This comprehensive guide covers:**
- Binary Search fundamentals & intuition
- All Binary Search templates (exact, lower bound, upper bound, answer space)
- When to use each template
- Common patterns and pitfalls
- Advanced binary search techniques

---

# Table of Contents

1. [Binary Search Core Concepts](#binary-search-core-concepts)
2. [Template 1: Classic Binary Search (Exact Value)](#template-1-classic-binary-search)
3. [Template 2: Lower Bound (First Position)](#template-2-lower-bound)
4. [Template 3: Upper Bound (Last Position)](#template-3-upper-bound)
5. [Template 4: Binary Search on Answer Space](#template-4-binary-search-on-answer)
6. [Template 5: Binary Search on Condition](#template-5-binary-search-on-condition)
7. [Template Summary & Decision Guide](#part-7-template-summary--decision-guide)
8. [Practice Problems](#practice-problems)
9. [Additional Practice Problems (Minimize Maximum / Maximize Minimum)](#additional-practice-problems-minimize-maximum--maximize-minimum)
10. [Answer Key](#answer-key)

---

# Part 1: Binary Search Core Concepts

## 🎯 Understanding Binary Search: The Guessing Game Analogy

**Fill in the blanks:**

1. Binary search works like guessing a number between 1 and 100:
   - **Linear search:** "Is it 1? No. Is it 2? No..." (100 guesses!)
   - **Binary search:** "Is it 50? Too high. Is it 25? Too low..." (**______** guesses!)

2. Binary search requires the array to be **______** (ascending or descending).

3. Binary search reduces time complexity from **______** to **______**.

4. At each step, binary search eliminates **______** of the remaining elements.

5. Binary search uses the **______** property to eliminate half the space.

## 🔑 Key Insight

**Binary Search = Halve the search space at each step until you find what you're looking for.**

Instead of checking every element (O(n)), binary search achieves O(log n)!

## 📋 Prerequisites for Binary Search

**Fill in the table:**

| Requirement | Why? |
|------------|------|
| **Sorted array** | Need to know if elements are before/after |
| **Monotonic property** | Condition that allows halving |
| **Search space** | Can be divided in half |
| **Comparable elements** | Need to compare with target |

## 💡 Why Binary Search Works

**Complete the explanation:**

Binary search works because:
1. If `array[mid] < target`, then **all elements to the left** are also **______** than target (because array is sorted!)
2. This allows us to **______** the entire left half
3. We only need to search the **______** half
4. Each step reduces search space by **______**

---

# Part 2: Template 1: Classic Binary Search (Exact Value)

## 🎯 When to Use This Template

**Use Template 1 when:**
- ✅ You need to find the **exact value** in a sorted array
- ✅ You want to know if a value **exists**
- ✅ You need the **index** of a specific element

**Example problems:**
- Binary Search (LeetCode #704)
- Search in Rotated Sorted Array
- Search a 2D Matrix

## 📝 The Algorithm

**Complete the steps:**

1. **Initialize:**
   - `left = 0`, `right = len(array) - 1`
   - Loop condition: **______** (fill in: <= or <)

2. **Main Loop:**
   - Calculate `mid = left + (right - left) // 2` (prevents **______**)
   - If `array[mid] == target`: **______**
   - If `array[mid] < target`: search **______** (`left = mid + 1`)
   - If `array[mid] > target`: search **______** (`right = mid - 1`)

3. **Exit:**
   - If loop ends, target **______** found → return -1

## 📋 Template 1: Complete the Code

```python
def binary_search(nums: List[int], target: int) -> int:
    """
    Classic binary search: find exact value in sorted array.
    
    Returns index of target, or -1 if not found.
    
    TODO: Fill in the missing code
    """
    left, right = ________________, ________________  # Fill in: initialize
    
    while ________________:  # Fill in: loop condition
        mid = ________________  # Fill in: calculate mid (prevent overflow!)
        
        if nums[mid] == target:
            return ________________  # Fill in
        
        elif nums[mid] < target:
            ________________  # Fill in: search right
        
        else:  # nums[mid] > target
            ________________  # Fill in: search left
    
    return ________________  # Fill in: not found
```

## 🎓 Key Points for Template 1

**Answer these questions:**

1. **Why use `left <= right` instead of `left < right`?**
   
   Your answer:
   ___________________________________________________________________________

2. **Why `left = mid + 1` instead of `left = mid`?**
   
   Your answer:
   ___________________________________________________________________________

3. **Why use `left + (right - left) // 2` instead of `(left + right) // 2`?**
   
   Your answer:
   ___________________________________________________________________________

## 📚 Example 1: Basic Binary Search

**Problem:** Find target 7 in `[1, 3, 5, 7, 9, 11, 13, 15]`

**Trace through the algorithm:**

| Step | left | right | mid | nums[mid] | Action |
|------|------|-------|-----|-----------|--------|
| 1 | 0 | 7 | **______** | **______** | **______** |
| 2 | **______** | **______** | **______** | **______** | **______** |

**Final result:** **______**

## 📚 Example 2: Target Not Found

**Problem:** Find target 6 in `[1, 3, 5, 7, 9, 11]`

**What happens?**
- We search until `left > right`
- Return: **______**

---

# Part 3: Template 2: Lower Bound (First Position)

## 🎯 When to Use This Template

**Use Template 2 when:**
- ✅ Finding **first occurrence** of a value
- ✅ Finding **insertion position** for a value
- ✅ Finding first position where **condition is true**
- ✅ Implementing **lower_bound** (C++)

**Example problems:**
- Search Insert Position (LeetCode #35)
- Find First and Last Position (LeetCode #34)
- First Bad Version (LeetCode #278)

## 🔑 Key Insight

**Lower Bound = First position where `array[pos] >= target`**

This is useful for:
- Insertion position: where to insert target to maintain sorted order
- First occurrence: first position where target appears

## 📝 The Algorithm

**Complete the steps:**

1. **Initialize:**
   - `left = 0`, `right = len(array) - 1`
   - Loop condition: **______** (fill in: <= or <)

2. **Main Loop:**
   - Calculate `mid`
   - If `array[mid] >= target`: **keep mid** (could be answer), `right = mid`
   - Else: **exclude mid**, `left = mid + 1`

3. **Exit:**
   - `left` points to **first position** where condition is true
   - Check if `array[left] == target` (if we need exact match)

## 📋 Template 2: Complete the Code

```python
def lower_bound(nums: List[int], target: int) -> int:
    """
    Find first position where nums[pos] >= target (lower bound).
    
    This gives insertion position or first occurrence.
    
    TODO: Fill in the missing code
    """
    left, right = 0, len(nums) - 1
    
    while ________________:  # Fill in: loop condition (hint: different from Template 1!)
        mid = left + (right - left) // 2
        
        if nums[mid] >= target:
            right = ________________  # Fill in: keep mid (hint: not mid - 1!)
        else:
            left = ________________  # Fill in: exclude mid
    
    # Optional: Check if target exists
    # return left if nums[left] == target else -1
    return ________________  # Fill in: insertion position
```

## 🎓 Key Differences from Template 1

**Fill in the comparison:**

| Aspect | Template 1 (Exact) | Template 2 (Lower Bound) |
|--------|-------------------|------------------------|
| Loop condition | `left <= right` | **______** |
| When condition true | `return mid` | **______** (keep mid!) |
| When condition false | `left = mid + 1` | **______** |
| Updates | Both `left` and `right` move | Only **______** moves or **______** stays |

## 📚 Example 1: Find Insertion Position

**Problem:** Find insertion position for target 5 in `[1, 3, 5, 5, 5, 6, 7]`

**What should it return?**
- Answer: **______** (first position where `>= 5`)

**Trace:**

| Step | left | right | mid | nums[mid] | Action |
|------|------|-------|-----|-----------|--------|
| 1 | 0 | 6 | 3 | 5 | **______** |
| 2 | **______** | **______** | **______** | **______** | **______** |

## 📚 Example 2: First Occurrence

**Problem:** Find first occurrence of 5 in `[1, 3, 5, 5, 5, 6, 7]`

**After lower_bound returns 2, check:**
```python
if nums[left] == target:
    return left  # Found at position 2
else:
    return -1  # Not found
```

**Result:** **______**

## 📚 Example 3: Search Insert Position

**Problem:** Find insertion position for 2 in `[1, 3, 5, 6]`

**Result:** Position **______** (insert between 1 and 3)

---

# Part 4: Template 3: Upper Bound (Last Position)

## 🎯 When to Use This Template

**Use Template 3 when:**
- ✅ Finding **last occurrence** of a value
- ✅ Finding last position where **condition is true**
- ✅ Implementing **upper_bound** (C++)

**Example problems:**
- Find First and Last Position (LeetCode #34) - for last position
- Last occurrence problems

## 🔑 Key Insight

**Upper Bound = Last position where `array[pos] <= target`**

Or: **First position where `array[pos] > target`** (minus 1)

## 📝 The Algorithm

**Complete the steps:**

1. **Initialize:**
   - `left = 0`, `right = len(array) - 1`
   - Loop condition: **______** (fill in: <= or <)

2. **Main Loop:**
   - **IMPORTANT:** Use `mid = left + (right - left + 1) // 2` (the **+1** is critical!)
   - If `array[mid] <= target`: **keep mid** (could be answer), `left = mid`
   - Else: **exclude mid**, `right = mid - 1`

3. **Exit:**
   - `left` points to **last position** where condition is true

## 📋 Template 3: Complete the Code

```python
def upper_bound(nums: List[int], target: int) -> int:
    """
    Find last position where nums[pos] <= target (upper bound).
    
    This gives last occurrence or last valid position.
    
    TODO: Fill in the missing code
    """
    left, right = 0, len(nums) - 1
    
    while ________________:  # Fill in: loop condition
        # CRITICAL: Use +1 in mid calculation to avoid infinite loop!
        mid = left + (right - left + 1) // 2  # Fill in: why +1?
        
        if nums[mid] <= target:
            left = ________________  # Fill in: keep mid (hint: not mid + 1!)
        else:
            right = ________________  # Fill in: exclude mid
    
    # Optional: Check if target exists
    # return left if nums[left] == target else -1
    return ________________  # Fill in: last position
```

## 🎓 Critical Understanding: Why +1 in Mid Calculation?

**Answer this question:**

**Why do we use `mid = left + (right - left + 1) // 2` instead of `mid = left + (right - left) // 2`?**

Your explanation:
___________________________________________________________________________
___________________________________________________________________________
___________________________________________________________________________

**Key insight:** When `left = mid` and `left = right - 1`, we need mid to move **forward**!

## 📚 Example 1: Last Occurrence

**Problem:** Find last occurrence of 5 in `[1, 3, 5, 5, 5, 6, 7]`

**What should it return?**
- Answer: **______** (last position where `<= 5`)

**Trace with +1 mid:**

| Step | left | right | mid (with +1) | nums[mid] | Action |
|------|------|-------|---------------|-----------|--------|
| 1 | 0 | 6 | 4 | 5 | **______** |
| 2 | **______** | **______** | **______** | **______** | **______** |

**What if we didn't use +1?**
- Mid would be 3, `left = mid` → `left = 3`
- Next iteration: `left = 3, right = 6`, mid = 4, `left = mid` → `left = 4`
- This works, but when `left = 4, right = 5`, without +1:
  - Mid = 4, `left = mid` → `left = 4` (no progress! **infinite loop!**)

## 📚 Example 2: Find Range

**Problem:** Find range of 5 in `[1, 3, 5, 5, 5, 6, 7]`

**Solution:**
```python
first = lower_bound(nums, 5)  # Returns 2
last = upper_bound(nums, 5)   # Returns 4
return [first, last]  # [2, 4]
```

---

# Part 5: Template 4: Binary Search on Answer Space

## 🎯 When to Use This Template

**Use Template 4 when:**
- ✅ Problem asks for **minimize maximum** or **maximize minimum**
- ✅ Answer is not in array, but in a **range of possible answers**
- ✅ You can **verify** if an answer is valid

**Example problems:**
- Capacity To Ship Packages Within D Days (LeetCode #1011)
- Split Array Largest Sum (LeetCode #410)
- Koko Eating Bananas (LeetCode #875)
- Minimum Time to Complete Trips (LeetCode #2187)

## 🔑 Key Insight

**Binary Search on Answer = Search in the answer space, not the array!**

Instead of searching the array, we:
1. Identify the **range of possible answers** `[min_answer, max_answer]`
2. Binary search in this range
3. For each candidate answer, check if it's **valid**
4. Find the **minimum valid answer** or **maximum valid answer**

## 📝 The Algorithm

**Complete the steps:**

1. **Find Answer Space:**
   - `left = min_possible_answer`
   - `right = max_possible_answer`

2. **Main Loop:**
   - `mid = left + (right - left) // 2` (for minimize)
   - OR `mid = left + (right - left + 1) // 2` (for maximize)
   - If `is_valid(mid)`: 
     - **Minimize:** `right = mid` (try smaller)
     - **Maximize:** `left = mid` (try larger)
   - Else:
     - **Minimize:** `left = mid + 1` (too small)
     - **Maximize:** `right = mid - 1` (too large)

3. **Exit:**
   - Return `left` (or `right` depending on problem)

## 📋 Template 4A: Minimize Maximum (Complete the Code)

```python
def minimize_maximum(nums: List[int], k: int) -> int:
    """
    Minimize the maximum value (e.g., minimize max sum of k subarrays).
    
    TODO: Fill in the binary search on answer
    """
    # Define answer space
    left = ________________  # Fill in: minimum possible answer
    right = ________________  # Fill in: maximum possible answer
    
    def is_valid(answer):
        """
        Check if 'answer' is a valid solution.
        
        TODO: Implement validation logic
        """
        # Example: Check if we can split array into k parts
        # where each part has sum <= answer
        count = 1
        current_sum = 0
        
        for num in nums:
            if current_sum + num > answer:
                count += 1
                current_sum = num
            else:
                current_sum += num
        
        return count <= k  # Can we split into k or fewer parts?
    
    # Binary search on answer
    while ________________:  # Fill in: loop condition
        mid = left + (right - left) // 2  # No +1 for minimize
        
        if is_valid(mid):
            right = ________________  # Fill in: try smaller
        else:
            left = ________________  # Fill in: need larger
    
    return ________________  # Fill in: minimum valid answer
```

## 📋 Template 4B: Maximize Minimum (Complete the Code)

```python
def maximize_minimum(nums: List[int], k: int) -> int:
    """
    Maximize the minimum value (e.g., maximize min distance).
    
    TODO: Fill in the binary search on answer
    """
    # Define answer space
    left = ________________  # Fill in: minimum possible answer
    right = ________________  # Fill in: maximum possible answer
    
    def is_valid(answer):
        """
        Check if 'answer' is a valid solution.
        
        TODO: Implement validation logic
        """
        # Example: Check if we can place k items with min distance >= answer
        count = 1
        last_pos = nums[0]
        
        for i in range(1, len(nums)):
            if nums[i] - last_pos >= answer:
                count += 1
                last_pos = nums[i]
        
        return count >= k  # Can we place k or more items?
    
    # Binary search on answer
    while ________________:  # Fill in: loop condition
        mid = left + (right - left + 1) // 2  # +1 for maximize!
        
        if is_valid(mid):
            left = ________________  # Fill in: try larger
        else:
            right = ________________  # Fill in: too large
    
    return ________________  # Fill in: maximum valid answer
```

## 📚 Example 1: Minimize Maximum (Split Array Largest Sum)

**Problem:** Split array into k non-empty subarrays, minimize the largest sum.

**Array:** `[7, 2, 5, 10, 8]`, `k = 2`

**Answer space:**
- Minimum: **______** (largest single element)
- Maximum: **______** (sum of all elements)

**Binary search in `[10, 32]`:**
- Try 21: Can we split into ≤2 parts each ≤21? **______**
- Try 18: Can we split? **______**
- Try 14: Can we split? **______**

**Result:** **______**

## 📚 Example 2: Maximize Minimum (Aggressive Cows)

**Problem:** Place k cows in positions, maximize minimum distance between cows.

**Positions:** `[1, 2, 4, 8, 9]`, `k = 3`

**Answer space:**
- Minimum: **______** (minimum possible distance)
- Maximum: **______** (maximum possible distance)

**Binary search to find maximum minimum distance!**

---

# Part 6: Template 5: Binary Search on Condition

## 🎯 When to Use This Template

**Use Template 5 when:**
- ✅ Finding **boundary** between two conditions
- ✅ Finding **transition point** (e.g., first bad version)
- ✅ Finding **peak** or **valley**
- ✅ Any monotonic condition

**Example problems:**
- First Bad Version (LeetCode #278)
- Find Peak Element (LeetCode #162)
- Search in Rotated Sorted Array (LeetCode #33)
- H-Index II (LeetCode #275)

## 🔑 Key Insight

**Binary Search on Condition = Find the boundary where condition changes from false to true (or vice versa)**

## 📝 The Algorithm

**Complete the steps:**

1. **Identify the condition:**
   - Define `condition(mid)` that returns True/False
   - Condition should be **monotonic** (all false then all true, or vice versa)

2. **Choose direction:**
   - If finding **first True**: when True, keep mid (`right = mid`), else exclude (`left = mid + 1`)
   - If finding **last False**: when False, keep mid (`left = mid`), else exclude (`right = mid - 1`)

3. **Adjust mid calculation:**
   - Use `+1` when `left = mid` to avoid infinite loop

## 📋 Template 5: Complete the Code

```python
def first_bad_version(n: int) -> int:
    """
    Find first bad version using binary search on condition.
    
    Condition: isBadVersion(mid) returns True for bad versions.
    Find first position where condition is True.
    
    TODO: Fill in the binary search on condition
    """
    left, right = 1, n
    
    while left < right:
        mid = left + (right - left) // 2
        
        if isBadVersion(mid):  # Condition is True
            right = ________________  # Fill in: keep mid (could be first bad)
        else:  # Condition is False
            left = ________________  # Fill in: exclude mid (search right)
    
    return ________________  # Fill in: first bad version

def find_peak_element(nums: List[int]) -> int:
    """
    Find peak element (nums[i] > nums[i-1] and nums[i] > nums[i+1]).
    
    TODO: Complete the peak finding logic
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        # Condition: Is mid on an ascending slope?
        if nums[mid] < nums[mid + 1]:
            # Peak is on the right
            left = ________________  # Fill in
        else:
            # Peak is on the left (or mid is peak)
            right = ________________  # Fill in
    
    return ________________  # Fill in: peak index
```

## 📚 Example 1: First Bad Version

**Problem:** Find first bad version in `[1, 2, 3, 4, 5]` where versions 4 and 5 are bad.

**Condition:** `isBadVersion(version)` returns True for bad versions.

**Trace:**

| Step | left | right | mid | isBadVersion(mid) | Action |
|------|------|-------|-----|-------------------|--------|
| 1 | 1 | 5 | 3 | False | **______** |
| 2 | **______** | **______** | **______** | **______** | **______** |

**Result:** First bad version is **______**

## 📚 Example 2: Find Peak Element

**Problem:** Find peak in `[1, 2, 1, 3, 5, 6, 4]`

**Peaks:** Index 1 (value 2) or index 5 (value 6)

**Strategy:** If `nums[mid] < nums[mid+1]`, peak is on the right!

---

# Part 7: Template Summary & Decision Guide

## 📊 Template Comparison Table

**Fill in the table:**

| Template | Loop Condition | When True | When False | Use Case |
|----------|---------------|-----------|------------|----------|
| **Template 1** (Exact) | `left <= right` | `return mid` | `left = mid + 1` or `right = mid - 1` | **______** |
| **Template 2** (Lower Bound) | **______** | `right = mid` | `left = mid + 1` | **______** |
| **Template 3** (Upper Bound) | **______** | `left = mid` (+1 in mid) | `right = mid - 1` | **______** |
| **Template 4** (Answer Space) | `left < right` | Depends on min/max | Depends on min/max | **______** |
| **Template 5** (Condition) | `left < right` | `right = mid` or `left = mid` | Opposite | **______** |

## 🎯 Quick Decision Guide

**Choose the template based on problem:**

1. **"Find exact value"** → **Template 1** (Classic Binary Search)
2. **"Find first occurrence"** or **"Find insertion position"** → **Template 2** (Lower Bound)
3. **"Find last occurrence"** → **Template 3** (Upper Bound)
4. **"Minimize maximum"** or **"Maximize minimum"** → **Template 4** (Answer Space)
5. **"Find transition point"** or **"Find boundary"** → **Template 5** (Condition)

---

# Part 8: Practice Problems

## Problem 1: Binary Search (Template 1)

**Your solution:**

```python
def search(nums: List[int], target: int) -> int:
    """
    TODO: Implement classic binary search
    """
    # Your code here:
    










```

## Problem 2: Search Insert Position (Template 2)

**Your solution:**

```python
def search_insert(nums: List[int], target: int) -> int:
    """
    TODO: Implement using lower bound
    """
    # Your code here:
    










```

## Problem 3: Find First and Last Position (Templates 2 & 3)

**Your solution:**

```python
def search_range(nums: List[int], target: int) -> List[int]:
    """
    TODO: Use both lower and upper bound
    """
    # Your code here:
    










```

## Problem 4: Capacity To Ship Packages (Template 4)

**Your solution:**

```python
def ship_within_days(weights: List[int], days: int) -> int:
    """
    Minimize ship capacity to ship all packages in 'days'.
    
    TODO: Binary search on answer space
    """
    # Your code here:
    










```

## Problem 5: First Bad Version (Template 5)

**Your solution:**

```python
def first_bad_version(n: int) -> int:
    """
    TODO: Binary search on condition
    """
    # Your code here:
    










```

---

## Additional Practice Problems: Minimize Maximum & Maximize Minimum

### 🟢 Minimize Maximum (First True) - Additional Problems

#### Problem 6: Smallest Divisor Given Threshold (LC 1283)

**Problem:** Given an array `nums` and an integer `threshold`, find the **smallest** positive divisor such that the sum of `ceil(nums[i] / divisor)` for all elements is **≤ threshold**.

**Your solution:**

```python
def smallest_divisor(nums: List[int], threshold: int) -> int:
    """
    TODO: Find smallest divisor using binary search on answer space
    """
    # Your code here:
    






```

#### Problem 7: Min Days to Make Bouquets (LC 1482)

**Problem:** You have `n` flowers; `bloomDay[i]` = day flower `i` blooms. Make `m` bouquets, each of `k` **adjacent** flowers. Find the **minimum** number of days to wait so you can make all bouquets.

**Your solution:**

```python
def min_days(bloomDay: List[int], m: int, k: int) -> int:
    """
    TODO: Find minimum days using binary search on answer space
    """
    # Your code here:
    






```

### 🔵 Maximize Minimum (Last True) - Additional Problems

#### Problem 8: Aggressive Cows (Classic)

**Problem:** Place `c` cows in `positions` (sorted) so the **minimum distance** between any two cows is as **large** as possible.

**Your solution:**

```python
def aggressive_cows(positions: List[int], c: int) -> int:
    """
    TODO: Maximize minimum distance using binary search on answer space
    """
    # Your code here:
    






```

#### Problem 9: Magnetic Force Between Two Balls (LC 1552)

**Problem:** Place `m` balls in `position` (sorted) so the **minimum** magnetic force (distance) between any two balls is as **large** as possible.

**Your solution:**

```python
def max_distance(position: List[int], m: int) -> int:
    """
    TODO: Maximize minimum distance (same pattern as Aggressive Cows)
    """
    # Your code here:
    






```

#### Problem 10: Maximum Tastiness of Candy Basket (LC 2517)

**Problem:** Pick `k` candies from `price` so the **minimum** absolute difference between any two chosen prices is as **large** as possible.

**Your solution:**

```python
def maximum_tastiness(price: List[int], k: int) -> int:
    """
    TODO: Maximize minimum difference using binary search on answer space
    """
    # Your code here:
    






```

#### Problem 11: Maximum Minimum Distance in Grid

**Problem:** Place `k` objects in a **grid** (some cells blocked) so the **minimum** Euclidean or Manhattan distance between any two objects is as **large** as possible.

**Note:** This is a concept problem - practice the 1D version first (same as Aggressive Cows), then extend to 2D.

#### Problem 12: Allocate Mailboxes (Variant)

**Variant:** Place `k` mailboxes among `houses` (sorted) so the **minimum** distance between any two **mailboxes** is as **large** as possible. (Same structure as Aggressive Cows.)

**Note:** LC 1478 "Allocate Mailboxes" minimizes **total** distance (DP). This variant uses **maximize minimum** (binary search) for practice.

---

# Part 9: Answer Key

## Binary Search Core Concepts Answers

1. 7 guesses (log₂(100) ≈ 6.64, rounded up)
2. sorted
3. O(n) to O(log n)
4. half
5. monotonic

## Template 1 Answers

**Complete code:**
```python
left, right = 0, len(nums) - 1
while left <= right:
    mid = left + (right - left) // 2
    if nums[mid] == target:
        return mid
    elif nums[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
return -1
```

**Key Points:**
1. `left <= right` includes the case when `left == right` (single element to check)
2. `left = mid + 1` because we know `nums[mid] < target`, so mid can't be the answer
3. `left + (right - left) // 2` prevents integer overflow when `left + right` is very large

## Template 2 Answers

**Complete code:**
```python
while left < right:
    mid = left + (right - left) // 2
    if nums[mid] >= target:
        right = mid  # Keep mid
    else:
        left = mid + 1  # Exclude mid
return left
```

**Comparison table:**
- Loop condition: `left < right`
- When true: `right = mid` (keep mid!)
- When false: `left = mid + 1`
- Updates: Only left moves or right stays

## Template 3 Answers

**Complete code:**
```python
while left < right:
    mid = left + (right - left + 1) // 2  # +1 critical!
    if nums[mid] <= target:
        left = mid  # Keep mid
    else:
        right = mid - 1  # Exclude mid
return left
```

**Why +1?** When `left = mid` and `left = right - 1`, without +1:
- `mid = (left + right) // 2 = left`
- `left = mid` → `left = left` (no progress! infinite loop!)
- With +1: `mid = left + 1 = right`, `left = mid` → `left = right` (loop ends)

## Template 4 Answers

**Template 4A (Minimize):**
```python
left = max(nums)  # Minimum possible answer
right = sum(nums)  # Maximum possible answer
while left < right:
    mid = left + (right - left) // 2
    if is_valid(mid):
        right = mid  # Try smaller
    else:
        left = mid + 1  # Need larger
return left
```

**Template 4B (Maximize):**
```python
left = min_distance  # Minimum possible
right = max_distance  # Maximum possible
while left < right:
    mid = left + (right - left + 1) // 2  # +1 for maximize!
    if is_valid(mid):
        left = mid  # Try larger
    else:
        right = mid - 1  # Too large
return left
```

## Template 5 Answers

**First Bad Version:**
```python
while left < right:
    mid = left + (right - left) // 2
    if isBadVersion(mid):
        right = mid  # Keep mid (could be first bad)
    else:
        left = mid + 1  # Exclude mid (search right)
return left
```

**Find Peak:**
```python
while left < right:
    mid = left + (right - left) // 2
    if nums[mid] < nums[mid + 1]:
        left = mid + 1  # Peak on right
    else:
        right = mid  # Peak on left or mid is peak
return left
```

## Template Summary Answers

| Template | Loop Condition | When True | When False | Use Case |
|----------|---------------|-----------|------------|----------|
| **Template 1** | `left <= right` | `return mid` | `left = mid + 1` or `right = mid - 1` | Find exact value |
| **Template 2** | `left < right` | `right = mid` | `left = mid + 1` | Find first position |
| **Template 3** | `left < right` | `left = mid` | `right = mid - 1` | Find last position |
| **Template 4** | `left < right` | Depends on min/max | Depends | Minimize/maximize |
| **Template 5** | `left < right` | `right = mid` or `left = mid` | Opposite | Find boundary |

---

# Study Checklist

Use this checklist to track your progress:

## Binary Search Fundamentals
- [ ] Can explain binary search intuitively (guessing game)
- [ ] Understand why sorted array is required
- [ ] Understand time complexity O(log n)
- [ ] Can explain monotonic property

## Template 1: Classic Binary Search
- [ ] Can write Template 1 from memory
- [ ] Understand why `left <= right`
- [ ] Understand why `left = mid + 1`
- [ ] Can prevent integer overflow

## Template 2: Lower Bound
- [ ] Can write Template 2 from memory
- [ ] Understand `left < right` vs `left <= right`
- [ ] Understand why `right = mid` (keep mid!)
- [ ] Can find insertion position

## Template 3: Upper Bound
- [ ] Can write Template 3 from memory
- [ ] Understand why `+1` in mid calculation
- [ ] Understand why `left = mid` (keep mid!)
- [ ] Can find last occurrence

## Template 4: Answer Space
- [ ] Can identify answer space problems
- [ ] Can implement minimize maximum
- [ ] Can implement maximize minimum
- [ ] Can write validation function

## Template 5: Condition
- [ ] Can identify condition-based problems
- [ ] Can write condition function
- [ ] Can find transition points
- [ ] Can find peaks/valleys

## Problem Solving
- [ ] Can choose correct template for problem
- [ ] Can implement Binary Search (Template 1)
- [ ] Can implement Search Insert Position (Template 2)
- [ ] Can implement Find First and Last Position (Templates 2 & 3)
- [ ] Can solve minimize/maximize problems (Template 4)
- [ ] Can solve condition-based problems (Template 5)

---

# Part 10: Solved Problems - Learn from Complete Solutions

## Solved Problem 1: Binary Search (Template 1)

**Problem:** Find target in sorted array, return index or -1.

**Solution:**

```python
def search(nums: List[int], target: int) -> int:
    """
    Classic binary search: find exact value.
    
    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Example: nums = [-1,0,3,5,9,12], target = 9
# Step 1: left=0, right=5, mid=2, nums[2]=3 < 9 → left=3
# Step 2: left=3, right=5, mid=4, nums[4]=9 == 9 → return 4
```

**Key Points:**
- Use `left <= right` to include single element case
- Update `left = mid + 1` and `right = mid - 1` (exclude mid)
- Prevent overflow with `left + (right - left) // 2`

## Solved Problem 2: Search Insert Position (Template 2)

**Problem:** Find insertion position for target in sorted array.

**Solution:**

```python
def search_insert(nums: List[int], target: int) -> int:
    """
    Find insertion position using lower bound.
    
    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] >= target:
            right = mid  # Keep mid (could be insertion position)
        else:
            left = mid + 1  # Exclude mid (search right)
    
    # Handle case when target > all elements
    if nums[left] < target:
        return left + 1
    return left

# Example: nums = [1,3,5,6], target = 5
# Result: 2 (first position where >= 5)

# Example: nums = [1,3,5,6], target = 2
# Result: 1 (insert at position 1)
```

**Key Points:**
- Use `left < right` (not <=)
- When `nums[mid] >= target`: `right = mid` (keep mid!)
- This finds lower bound = insertion position

## Solved Problem 3: Find First and Last Position (Templates 2 & 3)

**Problem:** Find first and last position of target in sorted array.

**Solution:**

```python
def search_range(nums: List[int], target: int) -> List[int]:
    """
    Find first and last position using lower and upper bound.
    
    Time: O(log n)
    Space: O(1)
    """
    def find_first(nums, target):
        """Lower bound: first position where >= target"""
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] >= target:
                right = mid
            else:
                left = mid + 1
        return left if nums[left] == target else -1
    
    def find_last(nums, target):
        """Upper bound: last position where <= target"""
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left + 1) // 2  # +1 critical!
            if nums[mid] <= target:
                left = mid  # Keep mid
            else:
                right = mid - 1  # Exclude mid
        return left if nums[left] == target else -1
    
    if not nums:
        return [-1, -1]
    
    first = find_first(nums, target)
    if first == -1:
        return [-1, -1]
    
    last = find_last(nums, target)
    return [first, last]

# Example: nums = [5,7,7,8,8,10], target = 8
# First: 3 (lower bound)
# Last: 4 (upper bound)
# Result: [3, 4]
```

**Key Points:**
- First: Use lower bound (Template 2)
- Last: Use upper bound (Template 3) with +1 in mid!
- Combine both for complete range

## Solved Problem 4: Capacity To Ship Packages (Template 4)

**Problem:** Minimize ship capacity to ship all packages within D days.

**Solution:**

```python
def ship_within_days(weights: List[int], days: int) -> int:
    """
    Minimize ship capacity using binary search on answer space.
    
    Strategy:
    1. Answer space: [max(weights), sum(weights)]
    2. For each capacity, check if we can ship in <= days
    3. Minimize valid capacity
    
    Time: O(n * log(sum(weights)))
    Space: O(1)
    """
    def can_ship(capacity):
        """Check if we can ship with given capacity in <= days"""
        days_needed = 1
        current_weight = 0
        
        for weight in weights:
            if current_weight + weight > capacity:
                days_needed += 1
                current_weight = weight
            else:
                current_weight += weight
        
        return days_needed <= days
    
    # Answer space: minimum = max single package, maximum = sum of all
    left = max(weights)
    right = sum(weights)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_ship(mid):
            right = mid  # Try smaller capacity
        else:
            left = mid + 1  # Need larger capacity
    
    return left  # Minimum valid capacity

# Example: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
# Answer space: [10, 55]
# Result: 15 (minimum capacity that works)
```

**Key Points:**
- Binary search on answer space (not the array!)
- Validation function checks if answer is feasible
- Minimize: `right = mid` when valid, `left = mid + 1` when invalid

## Solved Problem 5: Koko Eating Bananas (Template 4)

**Problem:** Minimize eating speed to finish all bananas within h hours.

**Solution:**

```python
def min_eating_speed(piles: List[int], h: int) -> int:
    """
    Minimize eating speed using binary search on answer space.
    
    Strategy:
    1. Answer space: [1, max(piles)]
    2. For each speed, calculate hours needed
    3. Minimize speed that allows finishing in <= h hours
    
    Time: O(n * log(max(piles)))
    Space: O(1)
    """
    def can_finish(speed):
        """Check if we can finish all bananas with given speed in <= h hours"""
        hours = 0
        for pile in piles:
            hours += (pile + speed - 1) // speed  # Ceiling division
            if hours > h:
                return False
        return hours <= h
    
    # Answer space: minimum = 1, maximum = max pile
    left = 1
    right = max(piles)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_finish(mid):
            right = mid  # Try slower speed
        else:
            left = mid + 1  # Need faster speed
    
    return left

# Example: piles = [3,6,7,11], h = 8
# Answer space: [1, 11]
# Result: 4 (minimum speed that works)
```

**Key Points:**
- Answer space: `[1, max(piles)]`
- Validation: calculate total hours needed
- Minimize: smaller valid answer is better

## Solved Problem 6: First Bad Version (Template 5)

**Problem:** Find first bad version in sorted versions.

**Solution:**

```python
def first_bad_version(n: int) -> int:
    """
    Find first bad version using binary search on condition.
    
    Condition: isBadVersion(version) returns True for bad versions.
    Find first position where condition is True.
    
    Time: O(log n)
    Space: O(1)
    """
    left, right = 1, n
    
    while left < right:
        mid = left + (right - left) // 2
        
        if isBadVersion(mid):
            right = mid  # Keep mid (could be first bad)
        else:
            left = mid + 1  # Exclude mid (search right)
    
    return left  # First bad version

# Example: n = 5, versions 4 and 5 are bad
# Step 1: left=1, right=5, mid=3, isBadVersion(3)=False → left=4
# Step 2: left=4, right=5, mid=4, isBadVersion(4)=True → right=4
# Step 3: left=4, right=4, loop ends → return 4
```

**Key Points:**
- Binary search on condition (not exact value)
- When condition True: keep mid (could be answer)
- When condition False: exclude mid (search right)

## Solved Problem 7: Find Peak Element (Template 5)

**Problem:** Find any peak element (element > neighbors).

**Solution:**

```python
def find_peak_element(nums: List[int]) -> int:
    """
    Find peak element using binary search on condition.
    
    Strategy:
    - If nums[mid] < nums[mid+1]: peak is on the right
    - Else: peak is on the left or mid is peak
    
    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        # Check if we're on ascending slope
        if nums[mid] < nums[mid + 1]:
            # Peak is on the right (ascending means peak ahead)
            left = mid + 1
        else:
            # Peak is on the left or mid is peak (descending means peak behind)
            right = mid
    
    return left  # Peak index

# Example: nums = [1,2,1,3,5,6,4]
# Peaks: index 1 (value 2) or index 5 (value 6)
# Result: 5 (one of the peaks)
```

**Key Points:**
- Binary search on condition (ascending vs descending)
- Ascending slope → peak on right
- Descending slope → peak on left or current is peak

## Solved Problem 8: Search in Rotated Sorted Array (Template 5)

**Problem:** Search target in rotated sorted array (no duplicates).

**Solution:**

```python
def search(nums: List[int], target: int) -> int:
    """
    Search in rotated sorted array.
    
    Strategy:
    1. Find which side is sorted
    2. Check if target is in sorted side
    3. Otherwise, search other side
    
    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            # Target is in sorted left half
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            # Target is in sorted right half
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# Example: nums = [4,5,6,7,0,1,2], target = 0
# Rotated at index 4
# Result: 4
```

**Key Points:**
- Determine which half is sorted
- Check if target is in sorted half
- Otherwise, search the other half

## Additional Practice Problems Answers

### Problem 6: Smallest Divisor Given Threshold (LC 1283)

**Complete code:**
```python
def smallest_divisor(nums: List[int], threshold: int) -> int:
    def ok(d: int) -> bool:
        return sum((x + d - 1) // d for x in nums) <= threshold
    
    lo, hi = 1, max(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

**Key Points:**
- Answer space: `[1, max(nums)]`
- Ceiling division: `(x + d - 1) // d`
- Minimize: `right = mid` when valid

### Problem 7: Min Days to Make Bouquets (LC 1482)

**Complete code:**
```python
def min_days(bloomDay: List[int], m: int, k: int) -> int:
    n = len(bloomDay)
    if m * k > n:
        return -1
    
    def ok(day: int) -> bool:
        bouquets, streak = 0, 0
        for d in bloomDay:
            if d <= day:
                streak += 1
                if streak == k:
                    bouquets += 1
                    streak = 0
            else:
                streak = 0
        return bouquets >= m
    
    lo, hi = min(bloomDay), max(bloomDay)
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

**Key Points:**
- Answer space: `[min(bloomDay), max(bloomDay)]`
- Track **adjacent** groups (streak)
- When streak == k: make bouquet, reset streak

### Problem 8: Aggressive Cows

**Complete code:**
```python
def aggressive_cows(positions: List[int], c: int) -> int:
    positions = sorted(positions)
    lo, hi = 1, positions[-1] - positions[0]
    
    def ok(d: int) -> bool:
        count, last = 1, positions[0]
        for p in positions[1:]:
            if p - last >= d:
                count += 1
                last = p
        return count >= c
    
    while lo < hi:
        mid = (lo + hi + 1) // 2  # +1 for maximize!
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo
```

**Key Points:**
- **Maximize:** Use `+1` in mid calculation!
- When valid: `left = mid` (try larger)
- Greedy: place first at `positions[0]`, then next at first position >= `last + d`

### Problem 9: Magnetic Force Between Two Balls (LC 1552)

**Complete code:**
```python
def max_distance(position: List[int], m: int) -> int:
    position = sorted(position)
    lo, hi = 1, position[-1] - position[0]
    
    def ok(d: int) -> bool:
        count, last = 1, position[0]
        for p in position[1:]:
            if p - last >= d:
                count += 1
                last = p
        return count >= m
    
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo
```

**Key Points:**
- **Identical to Aggressive Cows!**
- Same pattern: maximize minimum distance

### Problem 10: Maximum Tastiness of Candy Basket (LC 2517)

**Complete code:**
```python
def maximum_tastiness(price: List[int], k: int) -> int:
    price = sorted(price)
    lo, hi = 0, price[-1] - price[0]
    
    def ok(d: int) -> bool:
        count, last = 1, price[0]
        for p in price[1:]:
            if p - last >= d:
                count += 1
                last = p
        return count >= k
    
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo
```

**Key Points:**
- **Sort first!**
- Answer space: `[0, max(price) - min(price)]`
- Greedy selection: pick first valid candy

### Problem 11: Maximum Minimum Distance in Grid

**Concept:** Extend Aggressive Cows pattern to 2D:
1. Enumerate all valid grid cells (not blocked)
2. Use same binary search on distance
3. In `ok(d)`: use Manhattan `|x1-x2|+|y1-y2|` or Euclidean `sqrt((x1-x2)²+(y1-y2)²)`
4. Greedily place points with min distance >= d

### Problem 12: Allocate Mailboxes (Variant)

**Complete code:**
```python
def allocate_mailboxes_max_min(houses: List[int], k: int) -> int:
    houses = sorted(houses)
    lo, hi = 1, houses[-1] - houses[0]
    
    def ok(d: int) -> bool:
        count, last = 1, houses[0]
        for h in houses[1:]:
            if h - last >= d:
                count += 1
                last = h
        return count >= k
    
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo
```

**Key Points:**
- **Identical to Aggressive Cows!**
- Note: LC 1478 minimizes total distance (DP), this variant maximizes minimum distance (binary search)

---

**Good luck mastering Binary Search! 🚀**

*Remember: The key is recognizing which template to use and understanding the subtle differences!*
