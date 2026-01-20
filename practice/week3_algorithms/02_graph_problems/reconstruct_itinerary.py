"""
RECONSTRUCT ITINERARY - LEETCODE #332
=====================================

Difficulty: Hard
Time to complete: 25-30 minutes
Source: https://leetcode.com/problems/reconstruct-itinerary/

THIS PROBLEM HAS BEEN SEEN IN PINTEREST INTERVIEWS!

PROBLEM DESCRIPTION:
--------------------
You are given a list of airline tickets where tickets[i] = [from_i, to_i]
represent the departure and arrival airports of one flight. Reconstruct
the itinerary in order and return it.

All of the tickets belong to a man who departs from "JFK", thus the
itinerary must begin with "JFK". If there are multiple valid itineraries,
you should return the itinerary that has the smallest lexical order when
read as a single string.

You may assume all tickets form at least one valid itinerary. You must
use all the tickets once and only once.

EXAMPLES:
---------
Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
Output: ["JFK","MUC","LHR","SFO","SJC"]

Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]

ALGORITHM:
----------
This is an Eulerian path problem! We need to visit every edge exactly once.

Hierholzer's Algorithm:
1. Build adjacency list (sorted for lexical order)
2. DFS from "JFK", marking edges as used
3. When stuck (no more edges), add current node to result
4. Reverse the result at the end

INTERVIEW TIPS:
---------------
1. Recognize this as a graph problem
2. Know Eulerian path concept (visit every edge once)
3. Explain why we build result backwards
4. Discuss time complexity: O(E log E) for sorting

INSTRUCTIONS:
-------------
1. Complete each solution variant
2. Run tests: pytest reconstruct_itinerary.py -v
3. Time yourself: Try to solve in under 25 minutes
"""

from typing import List, Dict
from collections import defaultdict
import heapq


# =============================================================================
# SOLUTION 1: DFS with Sorted Adjacency List
# =============================================================================
def find_itinerary_dfs(tickets: List[List[str]]) -> List[str]:
    """
    Find itinerary using DFS with backtracking.
    
    Args:
        tickets: List of [from, to] airport pairs
    
    Returns:
        List of airports in order of visit
    
    Example:
        >>> tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
        >>> find_itinerary_dfs(tickets)
        ['JFK', 'ATL', 'JFK', 'SFO', 'ATL', 'SFO']
    
    Algorithm:
    1. Build adjacency list with sorted destinations
    2. DFS from "JFK", consuming edges
    3. Add to result when backtracking (no more edges)
    4. Reverse result
    
    Why build backwards?
    - When we reach a dead end, we add it to result
    - The dead end must be the LAST airport in our path
    - So we build from end to start, then reverse
    
    Time: O(E log E) for sorting
    Space: O(E) for graph and result
    """
    # TODO: Implement this function (15-20 lines)
    # Step 1: Build adjacency list
    #         graph = defaultdict(list)
    #         for src, dst in tickets:
    #             graph[src].append(dst)
    #
    # Step 2: Sort destinations in REVERSE order
    #         (so we can pop from end in O(1))
    #         for src in graph:
    #             graph[src].sort(reverse=True)
    #
    # Step 3: DFS
    #         result = []
    #         def dfs(airport):
    #             while graph[airport]:
    #                 next_airport = graph[airport].pop()
    #                 dfs(next_airport)
    #             result.append(airport)
    #         dfs("JFK")
    #
    # Step 4: Return reversed result
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 2: Iterative with Stack
# =============================================================================
def find_itinerary_iterative(tickets: List[List[str]]) -> List[str]:
    """
    Find itinerary using iterative approach with explicit stack.
    
    This avoids recursion, which is good for large inputs (no stack overflow).
    
    Algorithm:
    1. Build sorted adjacency list
    2. Use stack to simulate DFS
    3. When no more edges from current node, add to result
    4. Reverse result
    
    INTERVIEW TIP: Mention this as an alternative if interviewer asks
    about recursion depth limits.
    """
    # TODO: Implement this function (15-20 lines)
    # Step 1: Build adjacency list (sorted reverse for pop efficiency)
    # Step 2: Initialize stack with "JFK"
    # Step 3: While stack is not empty:
    #         - current = top of stack
    #         - If current has neighbors:
    #           - Push next neighbor onto stack
    #         - Else:
    #           - Pop current and add to result
    # Step 4: Return reversed result
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 3: Using Min-Heap for Lexical Order
# =============================================================================
def find_itinerary_heap(tickets: List[List[str]]) -> List[str]:
    """
    Find itinerary using min-heap for automatic lexical ordering.
    
    Instead of sorting adjacency lists, use a min-heap to always
    get the smallest destination first.
    
    This is cleaner when you need to frequently get the minimum.
    
    Time: O(E log E) - same as sorting approach
    Space: O(E)
    """
    # TODO: Implement this function (12-18 lines)
    # Step 1: Build graph with lists
    # Step 2: Convert lists to heaps (heapify)
    # Step 3: DFS using heappop instead of list.pop()
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 4: Backtracking Approach (More Explicit)
# =============================================================================
def find_itinerary_backtrack(tickets: List[List[str]]) -> List[str]:
    """
    Find itinerary using explicit backtracking.
    
    This approach is more explicit about trying different paths
    and backtracking when stuck.
    
    Algorithm:
    1. Build adjacency list with edge counts
    2. Try each destination in sorted order
    3. If we use all edges, we found the answer
    4. Otherwise, backtrack and try next destination
    
    INTERVIEW TIP: This shows clear backtracking logic but is less efficient.
    The Hierholzer approach (Solution 1) is O(E), this is potentially O(E!).
    """
    # TODO: Implement this function (20-25 lines)
    
    pass  # Remove this and implement


# =============================================================================
# HELPER: Visualize the Graph
# =============================================================================
def visualize_graph(tickets: List[List[str]]) -> None:
    """
    Print the graph structure for debugging.
    
    Helpful during interview to verify your understanding.
    """
    graph = defaultdict(list)
    for src, dst in tickets:
        graph[src].append(dst)
    
    for src in sorted(graph.keys()):
        dests = sorted(graph[src])
        print(f"{src} -> {', '.join(dests)}")


# =============================================================================
# INTERVIEW PRACTICE
# =============================================================================
def find_itinerary_interview(tickets: List[List[str]]) -> List[str]:
    """
    INTERVIEW VERSION: Write this from scratch in 25 minutes.
    
    Requirements:
    1. Start from "JFK"
    2. Use all tickets exactly once
    3. Return lexically smallest itinerary
    
    Before coding, explain:
    1. This is an Eulerian path problem
    2. We need to visit every EDGE once
    3. We use Hierholzer's algorithm
    4. Build result backwards, then reverse
    """
    # TODO: Implement from scratch without looking at other solutions!
    
    pass  # Remove this and implement


# =============================================================================
# TESTS
# =============================================================================
class TestReconstructItinerary:
    """Tests for reconstruct itinerary implementations."""
    
    def test_simple_path(self):
        """Test simple linear path."""
        tickets = [["JFK", "A"], ["A", "B"], ["B", "C"]]
        result = find_itinerary_dfs(tickets)
        if result:
            assert result == ["JFK", "A", "B", "C"]
    
    def test_example_1(self):
        """Test LeetCode example 1."""
        tickets = [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
        result = find_itinerary_dfs(tickets)
        if result:
            assert result == ["JFK", "MUC", "LHR", "SFO", "SJC"]
    
    def test_example_2(self):
        """Test LeetCode example 2 (with cycle)."""
        tickets = [["JFK", "SFO"], ["JFK", "ATL"], ["SFO", "ATL"], 
                   ["ATL", "JFK"], ["ATL", "SFO"]]
        result = find_itinerary_dfs(tickets)
        if result:
            assert result == ["JFK", "ATL", "JFK", "SFO", "ATL", "SFO"]
    
    def test_lexical_order(self):
        """Test that lexically smallest path is chosen."""
        tickets = [["JFK", "ZZZ"], ["JFK", "AAA"], ["AAA", "JFK"]]
        result = find_itinerary_dfs(tickets)
        if result:
            # Should go to AAA first (lexically smaller), then back to JFK, then ZZZ
            assert result == ["JFK", "AAA", "JFK", "ZZZ"]
    
    def test_iterative_matches_dfs(self):
        """Test that iterative solution matches DFS solution."""
        tickets = [["JFK", "SFO"], ["JFK", "ATL"], ["SFO", "ATL"], 
                   ["ATL", "JFK"], ["ATL", "SFO"]]
        
        dfs_result = find_itinerary_dfs(tickets)
        iter_result = find_itinerary_iterative(tickets)
        
        if dfs_result and iter_result:
            assert dfs_result == iter_result
    
    def test_heap_matches_dfs(self):
        """Test that heap solution matches DFS solution."""
        tickets = [["JFK", "SFO"], ["JFK", "ATL"], ["SFO", "ATL"], 
                   ["ATL", "JFK"], ["ATL", "SFO"]]
        
        dfs_result = find_itinerary_dfs(tickets)
        heap_result = find_itinerary_heap(tickets)
        
        if dfs_result and heap_result:
            assert dfs_result == heap_result
    
    def test_single_ticket(self):
        """Test with single ticket."""
        tickets = [["JFK", "A"]]
        result = find_itinerary_dfs(tickets)
        if result:
            assert result == ["JFK", "A"]
    
    def test_multiple_edges_same_pair(self):
        """Test with multiple tickets between same airports."""
        tickets = [["JFK", "A"], ["A", "JFK"], ["JFK", "A"], ["A", "B"]]
        result = find_itinerary_dfs(tickets)
        if result:
            # Should use all tickets
            assert len(result) == 5  # 4 tickets + 1 for starting airport


if __name__ == "__main__":
    print("RECONSTRUCT ITINERARY - Practice Problem")
    print("=" * 50)
    
    # Example walkthrough
    tickets = [["JFK", "SFO"], ["JFK", "ATL"], ["SFO", "ATL"], 
               ["ATL", "JFK"], ["ATL", "SFO"]]
    
    print("\nInput tickets:")
    for src, dst in tickets:
        print(f"  {src} -> {dst}")
    
    print("\nGraph structure:")
    visualize_graph(tickets)
    
    result = find_itinerary_dfs(tickets)
    print(f"\nItinerary: {result if result else '(not implemented)'}")
    print("Expected:  ['JFK', 'ATL', 'JFK', 'SFO', 'ATL', 'SFO']")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHT:")
    print("- This is an Eulerian path problem")
    print("- We must visit every EDGE exactly once")
    print("- Build result backwards when we hit dead ends")
    print("- Reverse at the end")
    
    print("\nRun 'pytest reconstruct_itinerary.py -v' for tests")
