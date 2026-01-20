"""
MERGE K SORTED LISTS - LEETCODE #23
===================================

Difficulty: Hard
Time to complete: 25-30 minutes
Source: https://leetcode.com/problems/merge-k-sorted-lists/

THIS PROBLEM IS MENTIONED IN PINTEREST'S OFFICIAL INTERVIEW PREP!

PROBLEM DESCRIPTION:
--------------------
You are given an array of k linked-lists, each linked-list is sorted
in ascending order. Merge all the linked-lists into one sorted linked-list
and return it.

EXAMPLES:
---------
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6

APPROACHES:
-----------
1. Brute Force: Collect all values, sort, create list - O(N log N)
2. Compare One by One: Compare k heads - O(kN)
3. Merge Lists One by One: Merge pairs - O(kN)
4. Divide and Conquer: Merge in pairs - O(N log k)
5. Min-Heap: Use heap to get minimum - O(N log k)

INTERVIEW TIPS:
---------------
1. Ask: Can I use extra space? Which approach do you prefer?
2. Min-Heap is the most elegant for interviews
3. Divide and Conquer is also excellent
4. Discuss time/space trade-offs

INSTRUCTIONS:
-------------
1. Complete each solution variant
2. Run tests: pytest merge_k_sorted_lists.py -v
3. Time yourself: Try to solve in under 25 minutes
"""

from typing import List, Optional
import heapq


# =============================================================================
# ListNode Definition
# =============================================================================
class ListNode:
    """Definition for singly-linked list."""
    
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next
    
    def __lt__(self, other: 'ListNode') -> bool:
        """Enable comparison for heap operations."""
        return self.val < other.val
    
    @staticmethod
    def from_list(values: List[int]) -> Optional['ListNode']:
        """Create linked list from Python list."""
        if not values:
            return None
        head = ListNode(values[0])
        current = head
        for val in values[1:]:
            current.next = ListNode(val)
            current = current.next
        return head
    
    def to_list(self) -> List[int]:
        """Convert linked list to Python list."""
        result = []
        current = self
        while current:
            result.append(current.val)
            current = current.next
        return result


# =============================================================================
# SOLUTION 1: Min-Heap (Priority Queue) - RECOMMENDED
# =============================================================================
def merge_k_lists_heap(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge k sorted lists using a min-heap.
    
    Algorithm:
    1. Add first node of each list to min-heap
    2. Pop minimum, add to result
    3. If popped node has next, add next to heap
    4. Repeat until heap is empty
    
    Time: O(N log k) - N total nodes, each heap operation is O(log k)
    Space: O(k) - heap stores at most k nodes
    
    INTERVIEW TIP: This is the cleanest solution!
    
    Example:
        >>> lists = [ListNode.from_list(l) for l in [[1,4,5],[1,3,4],[2,6]]]
        >>> result = merge_k_lists_heap(lists)
        >>> result.to_list()
        [1, 1, 2, 3, 4, 4, 5, 6]
    """
    # TODO: Implement this function (15-20 lines)
    # Step 1: Handle edge cases (empty lists)
    # Step 2: Initialize min-heap with (value, index, node) tuples
    #         Use index to break ties and avoid comparing nodes
    # Step 3: Create dummy head for result
    # Step 4: While heap is not empty:
    #         - Pop minimum
    #         - Add to result list
    #         - If node has next, push next to heap
    # Step 5: Return dummy.next
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 2: Divide and Conquer
# =============================================================================
def merge_k_lists_divide_conquer(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge k sorted lists using divide and conquer.
    
    Algorithm:
    1. Pair up lists and merge each pair
    2. Repeat until only one list remains
    
    This is like merge sort, but for linked lists!
    
    Time: O(N log k) - merge all N nodes log k times
    Space: O(log k) - recursion depth
    
    Example:
        Round 1: Merge pairs [0,1], [2,3], [4,5], ...
        Round 2: Merge pairs of merged lists
        Continue until one list remains
    """
    # TODO: Implement this function (20-25 lines)
    # Step 1: Handle edge cases
    # Step 2: Define merge_two_lists helper (merge 2 sorted lists)
    # Step 3: While len(lists) > 1:
    #         - Merge pairs
    #         - Replace lists with merged results
    # Step 4: Return lists[0]
    
    pass  # Remove this and implement


def merge_two_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Helper: Merge two sorted linked lists.
    
    This is a classic problem on its own (LeetCode #21).
    
    Time: O(n + m)
    Space: O(1)
    """
    # TODO: Implement this helper (12-15 lines)
    # Use dummy node technique
    # Compare heads, append smaller, advance pointer
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 3: Brute Force (for comparison)
# =============================================================================
def merge_k_lists_brute(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge k sorted lists using brute force.
    
    Algorithm:
    1. Collect all values into array
    2. Sort array
    3. Create new linked list
    
    Time: O(N log N) - sorting dominates
    Space: O(N) - store all values
    
    INTERVIEW TIP: Mention this as baseline, then improve!
    """
    # TODO: Implement this function (10-15 lines)
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 4: Sequential Merge
# =============================================================================
def merge_k_lists_sequential(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge lists one by one sequentially.
    
    Merge list 0 with list 1, result with list 2, etc.
    
    Time: O(kN) - worst case when lists are unbalanced
    Space: O(1)
    
    This is simpler but less efficient than divide and conquer.
    """
    # TODO: Implement this function (8-12 lines)
    
    pass  # Remove this and implement


# =============================================================================
# INTERVIEW PRACTICE
# =============================================================================
def merge_k_lists_interview(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    INTERVIEW VERSION: Write this from scratch in 20-25 minutes.
    
    Recommended approach: Min-Heap
    
    Steps:
    1. Clarify: Are lists always sorted? Can they be empty?
    2. Explain heap approach: always get minimum efficiently
    3. Implement with clean code
    4. Discuss: O(N log k) time, O(k) space
    
    Common follow-ups:
    - What if k is very large? (Heap is still O(log k) per operation)
    - What if lists are stored on disk? (External merge sort)
    - How would you parallelize this? (Divide and conquer naturally parallel)
    """
    # TODO: Implement from scratch!
    
    pass  # Remove this and implement


# =============================================================================
# TESTS
# =============================================================================
class TestMergeKSortedLists:
    """Tests for merge k sorted lists implementations."""
    
    def test_heap_example(self):
        """Test heap solution with example."""
        lists = [
            ListNode.from_list([1, 4, 5]),
            ListNode.from_list([1, 3, 4]),
            ListNode.from_list([2, 6])
        ]
        result = merge_k_lists_heap(lists)
        if result:
            assert result.to_list() == [1, 1, 2, 3, 4, 4, 5, 6]
    
    def test_heap_empty(self):
        """Test heap with empty input."""
        result = merge_k_lists_heap([])
        assert result is None
    
    def test_heap_single_list(self):
        """Test heap with single list."""
        lists = [ListNode.from_list([1, 2, 3])]
        result = merge_k_lists_heap(lists)
        if result:
            assert result.to_list() == [1, 2, 3]
    
    def test_heap_all_empty(self):
        """Test heap with all empty lists."""
        lists = [None, None, None]
        result = merge_k_lists_heap(lists)
        assert result is None
    
    def test_divide_conquer_example(self):
        """Test divide and conquer solution."""
        lists = [
            ListNode.from_list([1, 4, 5]),
            ListNode.from_list([1, 3, 4]),
            ListNode.from_list([2, 6])
        ]
        result = merge_k_lists_divide_conquer(lists)
        if result:
            assert result.to_list() == [1, 1, 2, 3, 4, 4, 5, 6]
    
    def test_merge_two_lists(self):
        """Test helper function."""
        l1 = ListNode.from_list([1, 3, 5])
        l2 = ListNode.from_list([2, 4, 6])
        result = merge_two_lists(l1, l2)
        if result:
            assert result.to_list() == [1, 2, 3, 4, 5, 6]
    
    def test_brute_force(self):
        """Test brute force solution."""
        lists = [
            ListNode.from_list([1, 4, 5]),
            ListNode.from_list([1, 3, 4]),
            ListNode.from_list([2, 6])
        ]
        result = merge_k_lists_brute(lists)
        if result:
            assert result.to_list() == [1, 1, 2, 3, 4, 4, 5, 6]
    
    def test_large_k(self):
        """Test with larger k."""
        lists = [ListNode.from_list([i, i+10, i+20]) for i in range(5)]
        result = merge_k_lists_heap(lists)
        if result:
            expected = sorted([i + j for i in range(5) for j in [0, 10, 20]])
            assert result.to_list() == expected


if __name__ == "__main__":
    print("MERGE K SORTED LISTS - Practice Problem")
    print("=" * 50)
    
    # Example
    lists = [
        ListNode.from_list([1, 4, 5]),
        ListNode.from_list([1, 3, 4]),
        ListNode.from_list([2, 6])
    ]
    
    print("\nInput lists:")
    for i, lst in enumerate(lists):
        print(f"  List {i}: {lst.to_list() if lst else '[]'}")
    
    result = merge_k_lists_heap(lists)
    print(f"\nMerged: {result.to_list() if result else '(not implemented)'}")
    print("Expected: [1, 1, 2, 3, 4, 4, 5, 6]")
    
    print("\n" + "=" * 50)
    print("APPROACHES:")
    print("1. Min-Heap: O(N log k) time, O(k) space - RECOMMENDED")
    print("2. Divide & Conquer: O(N log k) time, O(log k) space")
    print("3. Brute Force: O(N log N) time, O(N) space")
    print("4. Sequential: O(kN) time, O(1) space")
    
    print("\nRun 'pytest merge_k_sorted_lists.py -v' for tests")
