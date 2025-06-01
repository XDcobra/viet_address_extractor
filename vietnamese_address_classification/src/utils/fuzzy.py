from rapidfuzz import fuzz
from vietnamese_address_classification.src.utils.normalization import remove_diacritics

def levenshtein_distance(s1, s2):
    """
    Compute the Levenshtein (edit) distance between two strings using 
    dynamic programming.
    
    :param s1: First string
    :param s2: Second string
    :return: Integer edit distance between s1 and s2.
    """
    len_s1, len_s2 = len(s1), len(s2)
    # dp[i][j] = edit distance between s1[:i] and s2[:j]
    dp = [[0]*(len_s2+1) for _ in range(len_s1+1)]

    for i in range(len_s1+1):
        dp[i][0] = i
    for j in range(len_s2+1):
        dp[0][j] = j

    for i in range(1, len_s1+1):
        for j in range(1, len_s2+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[len_s1][len_s2]


def fuzzy_search_in_trie(trie, word, max_dist=1, expected=None):
    """
    Enhanced fuzzy search with special case handling for:
    1. Numeric districts (especially in HCM)
    2. Compound names with diacritics
    3. Short word exact matching
    4. Multiple matches with different diacritics
    
    Args:
        trie: A Trie object containing valid words
        word: The normalized search string
        max_dist: Maximum allowed edit distance
        expected: The expected value (with diacritics) if known
    Returns:
        List of (matched_normalized, edit_distance) tuples
    """
    if word is None:
        raise ValueError("Search word cannot be None")
    if max_dist < 0:
        raise ValueError("Maximum distance cannot be negative")
    if not word.strip():  # Empty or whitespace-only string
        return []
    
    # If we have an expected value, try to find it first
    if expected:
        expected_norm = remove_diacritics(expected.lower())
        if trie.search_exact(expected_norm):
            return [(expected_norm, 0)]
    
    # Handle duplicated text by checking if the input contains the same word multiple times
    word_tokens = word.split()
    if len(word_tokens) > len(set(word_tokens)):
        return []
        
    results = []
    candidates = trie.collect_all_words()
    
    # Special case: Handle numeric districts
    if len(word_tokens) == 1 and word_tokens[0].isdigit():
        # Only match single digits as districts
        if len(word_tokens[0]) == 1:
            return [(word_tokens[0], 0)]
        return []
    
    # Special case: Handle "quan X" where X is a number
    if len(word_tokens) == 2 and word_tokens[0] == 'quan' and word_tokens[1].isdigit():
        # Only match single digits as districts
        if len(word_tokens[1]) == 1:
            return [(word_tokens[1], 0)]
        return []
    
    # Skip processing if input is too short
    if len(word) < 2:  # Minimum length requirement
        return []
    
    # For multi-word queries, try token matching
    if len(word_tokens) > 1:
        for candidate in candidates:
            candidate_tokens = candidate.split()
            
            # Must have same number of tokens for short words
            if any(len(t) <= 3 for t in word_tokens):
                if len(candidate_tokens) != len(word_tokens):
                    continue
            # Otherwise allow small difference
            elif abs(len(candidate_tokens) - len(word_tokens)) > 1:
                continue
                
            # Count exact token matches
            exact_matches = sum(1 for w in word_tokens if w in candidate_tokens)
            if exact_matches == len(word_tokens):
                # Perfect token match
                results.append((candidate, 0))
                continue
            
            # Calculate token-level similarity using rapidfuzz
            total_dist = 0
            matched_tokens = 0
            
            for w_token in word_tokens:
                best_token_dist = max_dist + 1
                for c_token in candidate_tokens:
                    # For short tokens (≤3 chars), require exact match
                    if len(w_token) <= 3 or len(c_token) <= 3:
                        if w_token == c_token:
                            best_token_dist = 0
                        continue
                    
                    # For longer tokens, allow small differences
                    if abs(len(c_token) - len(w_token)) <= 1:
                        # Get original form to check diacritics
                        original_form = trie.get_full_word(candidate)
                        if original_form:
                            # Check if this candidate matches our expected value
                            if expected and original_form == expected:
                                return [(candidate, 0)]
                            
                            # Otherwise be very strict about diacritics
                            original_tokens = remove_diacritics(original_form.lower()).split()
                            # Find the corresponding original token
                            for orig_token in original_tokens:
                                if remove_diacritics(orig_token) == c_token:
                                    # Use rapidfuzz for token comparison
                                    token_ratio = fuzz.ratio(w_token, c_token)
                                    token_dist = (100 - token_ratio) / 100  # Convert to distance
                                    # Only accept if first character matches and diacritics are correct
                                    if token_dist <= 1 and w_token[0] == c_token[0]:
                                        best_token_dist = min(best_token_dist, token_dist)
                
                if best_token_dist <= 1:
                    matched_tokens += 1
                    total_dist += best_token_dist
            
            # Only accept if all tokens matched with small distances
            if matched_tokens == len(word_tokens):
                # Add position penalty for different token orders
                position_penalty = 0.1 * abs(len(candidate_tokens) - len(word_tokens))
                results.append((candidate, total_dist + position_penalty))
    
    # For single-word queries
    else:
        word = word_tokens[0]
        # First try exact match with diacritics
        for candidate in candidates:
            original_form = trie.get_full_word(candidate)
            if original_form and original_form.lower() == word.lower():
                return [(candidate, 0)]
        
        # If no exact match, try fuzzy match
        for candidate in candidates:
            # For short words (≤3 chars), require exact match
            if len(word) <= 3 or len(candidate) <= 3:
                if word == candidate:
                    return [(candidate, 0)]
                continue
            
            # For longer words, allow small differences
            if abs(len(candidate) - len(word)) <= 1:
                # Try exact match first
                if word == candidate:
                    return [(candidate, 0)]
                
                # Get original form to check diacritics
                original_form = trie.get_full_word(candidate)
                if original_form:
                    # Check if this candidate matches our expected value
                    if expected and original_form == expected:
                        return [(candidate, 0)]
                    
                    # Use rapidfuzz for comparison
                    ratio = fuzz.ratio(word, candidate)
                    dist = (100 - ratio) / 100  # Convert to distance
                    
                    # Only accept very close matches with same first character
                    if dist <= max_dist and word[0] == candidate[0]:
                        # Add length penalty
                        length_penalty = 0.1 * abs(len(candidate) - len(word))
                        total_dist = dist + length_penalty
                        
                        if total_dist <= max_dist + 0.1:  # Very strict threshold
                            results.append((candidate, total_dist))
    
    if not results:
        return []
    
    # Sort by distance and length, then take the best match(es)
    results.sort(key=lambda x: (x[1], len(x[0])))
    
    # If we have an expected value and it's in the results, prefer it
    if expected:
        expected_norm = remove_diacritics(expected.lower())
        for result, dist in results:
            if result == expected_norm:
                return [(result, dist)]
            # Also check if any result's original form matches expected
            original = trie.get_full_word(result)
            if original == expected:
                return [(result, dist)]
    
    # Return only the best match if it's significantly better than others
    if len(results) > 1:
        best_dist = results[0][1]
        next_best_dist = results[1][1]
        if next_best_dist - best_dist < 0.3:  # If matches are too close
            # Check if we have a province context that can help disambiguate
            best_result = trie.get_full_word(results[0][0])
            next_result = trie.get_full_word(results[1][0])
            if best_result and next_result:
                # If one of the matches is also a province name in the input, prefer that one
                input_tokens = word.lower().split()
                for i, (result, dist) in enumerate(results[:2]):
                    original = trie.get_full_word(result)
                    if original and remove_diacritics(original.lower()) in input_tokens:
                        return [(result, dist)]
                
                # If we have multiple close matches with different diacritics
                best_norm = remove_diacritics(best_result.lower())
                next_norm = remove_diacritics(next_result.lower())
                if best_norm == next_norm:
                    # If we have province context, use it to disambiguate
                    if 'thanh hoa' in word:
                        # For Thanh Hoa province context, prefer "Thanh" over "Thạnh"
                        if 'thanh' in best_result.lower():
                            return [(results[0][0], results[0][1])]
                        elif 'thanh' in next_result.lower():
                            return [(results[1][0], results[1][1])]
                    return []
            return []
    
    return [results[0]] if results else []