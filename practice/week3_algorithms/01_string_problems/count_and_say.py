"""
COUNT AND SAY - LEETCODE #38
============================

Difficulty: Medium
Time to complete: 15-20 minutes
Source: https://leetcode.com/problems/count-and-say/

THIS PROBLEM HAS BEEN SEEN IN PINTEREST INTERVIEWS!

PROBLEM DESCRIPTION:
--------------------
The count-and-say sequence is a sequence of digit strings defined by the
recursive formula:

- countAndSay(1) = "1"
- countAndSay(n) is the way you would "say" the digit string from 
  countAndSay(n-1), which is then converted into a different digit string.

To determine how you "say" a digit string, split it into the minimal number
of substrings such that each substring contains exactly one unique digit.
Then for each substring, say the number of digits, then say the digit.
Finally, concatenate every said digit.

EXAMPLES:
---------
countAndSay(1) = "1"
countAndSay(2) = "11" (one 1)
countAndSay(3) = "21" (two 1s)
countAndSay(4) = "1211" (one 2, one 1)
countAndSay(5) = "111221" (one 1, one 2, two 1s)

INTERVIEW TIPS:
---------------
1. Start by walking through examples manually
2. Identify the two-step process:
   a) Count consecutive identical digits
   b) Build new string: count + digit
3. Discuss iterative vs recursive approaches
4. Mention time/space complexity

INSTRUCTIONS:
-------------
1. Complete each solution variant
2. Run tests: pytest count_and_say.py -v
3. Time yourself: Try to solve in under 15 minutes
"""

from typing import List


# =============================================================================
# SOLUTION 1: Iterative Approach (Recommended for Interview)
# =============================================================================
def count_and_say_iterative(n: int) -> str:
    """
    Generate the nth term of count-and-say sequence iteratively.
    
    Args:
        n: The term to generate (1-indexed)
    
    Returns:
        The nth count-and-say string
    
    Example:
        >>> count_and_say_iterative(1)
        '1'
        >>> count_and_say_iterative(4)
        '1211'
        >>> count_and_say_iterative(5)
        '111221'
    
    Algorithm:
    1. Start with "1"
    2. For each step from 2 to n:
       - Iterate through current string
       - Count consecutive identical characters
       - Build new string: str(count) + digit
    3. Return final string
    
    Time Complexity: O(n * m) where m is the average length of strings
    Space Complexity: O(m) for the current and next strings
    """
    # TODO: Implement this function (12-18 lines)
    # Step 1: Handle base case n=1, return "1"
    # Step 2: Initialize result = "1"
    # Step 3: Loop n-1 times:
    #         - Initialize next_result as empty string
    #         - Initialize count = 1, current_char = result[0]
    #         - For each subsequent character:
    #           - If same as current_char: increment count
    #           - Else: append str(count) + current_char to next_result
    #                   reset count = 1, current_char = new char
    #         - Don't forget the last group!
    #         - Set result = next_result
    # Step 4: Return result
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 2: Helper Function Approach (Cleaner Code)
# =============================================================================
def get_next_sequence(s: str) -> str:
    """
    Generate the next sequence from the current one.
    
    Args:
        s: Current sequence string
    
    Returns:
        Next sequence in count-and-say
    
    Example:
        >>> get_next_sequence("1")
        "11"
        >>> get_next_sequence("11")
        "21"
        >>> get_next_sequence("21")
        "1211"
    """
    # TODO: Implement this helper function (10-15 lines)
    # This handles the "say" logic for a single string
    
    pass  # Remove this and implement


def count_and_say_with_helper(n: int) -> str:
    """
    Generate nth term using helper function.
    
    This approach separates concerns:
    - Main function handles iteration
    - Helper function handles the "say" logic
    
    INTERVIEW TIP: This shows good code organization!
    """
    # TODO: Implement using get_next_sequence helper
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 3: Recursive Approach
# =============================================================================
def count_and_say_recursive(n: int) -> str:
    """
    Generate nth term recursively.
    
    Base case: n = 1 -> return "1"
    Recursive case: count_and_say(n) = say(count_and_say(n-1))
    
    INTERVIEW TIP: Discuss the trade-off:
    - Recursive is elegant but uses O(n) call stack
    - Iterative is more memory efficient
    - Both have same time complexity
    
    Args:
        n: The term to generate
    
    Returns:
        The nth count-and-say string
    """
    # TODO: Implement recursive solution (8-12 lines)
    # Base case: n == 1 -> return "1"
    # Recursive case: get_next_sequence(count_and_say_recursive(n-1))
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 4: Using itertools.groupby (Python-specific)
# =============================================================================
def count_and_say_groupby(n: int) -> str:
    """
    Generate nth term using itertools.groupby.
    
    itertools.groupby groups consecutive identical elements.
    
    Example:
        groupby("1211") -> [('1', ['1']), ('2', ['2']), ('1', ['1', '1'])]
    
    This is a Pythonic solution, but explain that you know the manual approach too!
    
    INTERVIEW TIP: Show this as an alternative, but be ready to implement without it.
    """
    from itertools import groupby
    
    # TODO: Implement using groupby (6-10 lines)
    
    pass  # Remove this and implement


# =============================================================================
# SOLUTION 5: Two-pointer Approach
# =============================================================================
def count_and_say_two_pointers(n: int) -> str:
    """
    Generate nth term using two pointers.
    
    This is another way to think about the problem:
    - Use two pointers to find groups of identical characters
    - i = start of group, j = end of group
    
    Algorithm:
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[j] == s[i]:
                j += 1
            # Now s[i:j] is a group of identical chars
            count = j - i
            result += str(count) + s[i]
            i = j
    
    INTERVIEW TIP: This shows algorithmic thinking about string processing.
    """
    # TODO: Implement using two pointers (15-20 lines)
    
    pass  # Remove this and implement


# =============================================================================
# INTERVIEW PRACTICE: Complete Under Time Pressure
# =============================================================================
def count_and_say_interview(n: int) -> str:
    """
    INTERVIEW VERSION: Write this from scratch in 15 minutes.
    
    Requirements:
    1. Clean, readable code
    2. Handle edge cases
    3. Explain time/space complexity
    
    Start by:
    1. Clarifying the problem
    2. Walking through an example
    3. Coding the solution
    4. Testing with examples
    """
    # TODO: Implement from scratch without looking at other solutions!
    
    pass  # Remove this and implement


# =============================================================================
# TESTS
# =============================================================================
class TestCountAndSay:
    """Tests for count-and-say implementations."""
    
    def test_iterative_base_cases(self):
        """Test base cases."""
        assert count_and_say_iterative(1) == "1"
        assert count_and_say_iterative(2) == "11"
        assert count_and_say_iterative(3) == "21"
    
    def test_iterative_longer(self):
        """Test longer sequences."""
        assert count_and_say_iterative(4) == "1211"
        assert count_and_say_iterative(5) == "111221"
        assert count_and_say_iterative(6) == "312211"
    
    def test_helper_function(self):
        """Test the helper function."""
        if get_next_sequence:
            assert get_next_sequence("1") == "11"
            assert get_next_sequence("11") == "21"
            assert get_next_sequence("21") == "1211"
            assert get_next_sequence("1211") == "111221"
    
    def test_with_helper(self):
        """Test solution with helper."""
        if count_and_say_with_helper:
            assert count_and_say_with_helper(1) == "1"
            assert count_and_say_with_helper(5) == "111221"
    
    def test_recursive(self):
        """Test recursive solution."""
        if count_and_say_recursive:
            assert count_and_say_recursive(1) == "1"
            assert count_and_say_recursive(4) == "1211"
            assert count_and_say_recursive(5) == "111221"
    
    def test_groupby(self):
        """Test groupby solution."""
        if count_and_say_groupby:
            assert count_and_say_groupby(1) == "1"
            assert count_and_say_groupby(5) == "111221"
    
    def test_two_pointers(self):
        """Test two pointer solution."""
        if count_and_say_two_pointers:
            assert count_and_say_two_pointers(1) == "1"
            assert count_and_say_two_pointers(5) == "111221"
    
    def test_all_match(self):
        """Test that all implementations give same result."""
        for n in range(1, 8):
            results = []
            
            if count_and_say_iterative:
                results.append(count_and_say_iterative(n))
            if count_and_say_with_helper:
                results.append(count_and_say_with_helper(n))
            if count_and_say_recursive:
                results.append(count_and_say_recursive(n))
            if count_and_say_groupby:
                results.append(count_and_say_groupby(n))
            
            # All non-None results should be equal
            results = [r for r in results if r is not None]
            if len(results) > 1:
                assert all(r == results[0] for r in results), f"Mismatch at n={n}"


if __name__ == "__main__":
    print("COUNT AND SAY - Practice Problem")
    print("=" * 50)
    
    # Show the sequence
    print("\nFirst 8 terms of the sequence:")
    for i in range(1, 9):
        result = count_and_say_iterative(i)
        if result:
            print(f"n={i}: {result}")
        else:
            print(f"n={i}: (not implemented)")
    
    print("\n" + "=" * 50)
    print("WALKTHROUGH:")
    print("n=1: '1' (base case)")
    print("n=2: '11' (one 1)")
    print("n=3: '21' (two 1s)")
    print("n=4: '1211' (one 2, one 1)")
    print("n=5: '111221' (one 1, one 2, two 1s)")
    
    print("\nRun 'pytest count_and_say.py -v' for tests")
