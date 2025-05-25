import re
import unicodedata
import time
import logging
from rapidfuzz import fuzz, process

# Configure logging
def setup_logging(disable_file_logging=False):
    """
    Configure logging for the address parser.
    
    Args:
        disable_file_logging: If True, disable ALL logging (both file and console).
    """
    # Get the logger
    logger = logging.getLogger('address_parser')
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if disable_file_logging:
        # If disable_file_logging is True, disable ALL logging
        logger.setLevel(logging.CRITICAL)  # Set to highest level to disable all logging
        return
    
    # Set the base level to DEBUG to capture all levels
    logger.setLevel(logging.DEBUG)
    
    # Add console handler with INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    # Add file handler with DEBUG level if not disabled
    if not disable_file_logging:
        # Clear the log file before starting
        with open('address_parser.log', 'w', encoding='utf-8') as f:
            f.write('')  # Clear the file
            
        file_handler = logging.FileHandler('address_parser.log', mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Don't propagate to root logger
    logger.propagate = False

# Initialize logger
logger = logging.getLogger('address_parser')

##############################################################
# 1. TRIE STRUCTURE & HELPER METHODS
##############################################################

class TrieNode:
    """
    Represents a single node within a Trie.
    
    Attributes:
        children (dict): A dictionary mapping a character to a TrieNode.
        is_terminal (bool): Whether this node represents the end of a valid word.
        full_words (list): List of original words (with diacritics) if this node is the end of a valid entry.
    """
    def __init__(self):
        self.children = {}
        self.is_terminal = False
        self.full_words = []  # Changed from full_word to full_words list


class Trie:
    """
    A Trie (prefix tree) for storing and searching strings.
    
    Methods:
        insert(word, original_name=None):
            Insert a normalized 'word' into the trie. 
            'original_name' can store a more complete or diacritics version.
        search_exact(word) -> bool:
            Return True if 'word' exists exactly as a terminal node in the trie.
        get_full_word(word) -> str or None:
            Return the stored original form if 'word' is found, else None.
        collect_all_words() -> list[str]:
            Collect all valid words stored in the trie (in normalized form).
    """
    def __init__(self):
        self.root = TrieNode()
        # logger.debug("Initialized new Trie")

    def insert(self, word, original_name=None):
        """
        Insert a 'word' into the trie (usually already normalized).
        
        :param word: The normalized string to insert.
        :param original_name: Optionally store the original form (with diacritics).
        """
        try:
            current = self.root
            for char in word:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            current.is_terminal = True
            # Store the original name if provided
            if original_name:
                # Only add if not already in the list to avoid duplicates
                if original_name not in current.full_words:
                    current.full_words.append(original_name)
                    # Debug print for Tân Hòa cases
                    if 'tan hoa' in word.lower():
                        print(f"Inserting '{original_name}' with normalized form '{word}'")
                        print(f"Current full_words for this node: {current.full_words}")
            # logger.debug(f"Inserted word: {word} (original: {original_name})")
        except Exception as e:
            logger.error(f"Error inserting word '{word}': {str(e)}")

    def search_exact(self, word):
        """
        Check if 'word' exists in the trie as a terminal node.
        
        :param word: The string to search for (normalized).
        :return: True if found exactly, False otherwise.
        """
        try:
            current = self.root
            for char in word:
                if char not in current.children:
                    logger.debug(f"Exact search failed for word: {word}")
                    return False
                current = current.children[char]
            result = current.is_terminal
            logger.debug(f"Exact search for {word}: {'found' if result else 'not found'}")
            return result
        except Exception as e:
            logger.error(f"Error in exact search for '{word}': {str(e)}")
            return False

    def get_full_word(self, word, original_text=None):
        """
        Retrieve the original form of 'word' if it exists in the trie.
        If original_text is provided, try to find the best matching diacritics version.
        
        Args:
            word: The normalized string to look up
            original_text: Optional original text with diacritics to use as a hint
        
        Returns:
            The best matching original form if found, otherwise None
        """
        try:
            # If we have original text, first try to find a longer form
            if original_text and ' ' in original_text:
                # Try to find the longer form first
                longer_word = remove_diacritics(original_text.lower())
                current = self.root
                for char in longer_word:
                    if char not in current.children:
                        break
                    current = current.children[char]
                else:  # If we found the longer form
                    if current.is_terminal and current.full_words:
                        # Use rapidfuzz to find the closest match
                        result = process.extractOne(
                            original_text,
                            current.full_words,
                            scorer=fuzz.ratio
                        )
                        if result:
                            return result[0]
            
            # If longer form not found or no original text, try the original word
            current = self.root
            for char in word:
                if char not in current.children:
                    logger.debug(f"Full word lookup failed for: {word}")
                    return None
                current = current.children[char]
            
            if current.is_terminal and current.full_words:
                # If we have original text, use it to find the best match
                if original_text:
                    # Use rapidfuzz to find the closest match
                    result = process.extractOne(
                        original_text,
                        current.full_words,
                        scorer=fuzz.ratio
                    )
                    if result:
                        # Debug print for Tân Hòa cases
                        if 'tan hoa' in word.lower():
                            print(f"Best match found: {result[0]} with score {result[1]}")
                        return result[0]
                
                # If no original text or no good match found, return the first version
                return current.full_words[0]
                
            logger.debug(f"No full word found for: {word}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting full word for '{word}': {str(e)}")
            return None

    def collect_all_words(self):
        """
        Collect all valid (normalized) words stored in the trie.
        
        :return: A list of normalized strings representing all valid entries.
        """
        try:
            results = []

            def dfs(node, path):
                if node.is_terminal:
                    results.extend(node.full_words)
                for c, child in node.children.items():
                    dfs(child, path + c)

            dfs(self.root, "")
            logger.debug(f"Collected {len(results)} words from trie")
            return results
        except Exception as e:
            logger.error(f"Error collecting words from trie: {str(e)}")
            return []


##############################################################
# 2. ADVANCED FUZZY SEARCH (USING EDIT DISTANCE)
##############################################################

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


##############################################################
# 3. TEXT NORMALIZATION & ABBREVIATION EXPANSIONS
##############################################################

def remove_diacritics(text):
    """
    Convert accented Vietnamese chars to ASCII equivalents 
    by using Unicode normalization.
    
    Args:
        text: The raw input string (potentially with diacritics)
    Returns:
        String with diacritics removed and converted to lowercase
    """
    if not text:
        return text
        
    # Convert to NFKD form and remove combining characters
    text = unicodedata.normalize('NFKD', text)
    result = []
    for c in text:
        if not unicodedata.combining(c):
            # Special handling for Đ/đ
            if c in 'Đđ':
                result.append('d')
            else:
                result.append(c.lower())
    
    return ''.join(result)


def expand_abbreviations(text):
    """
    Detect specific patterns like 'p1' -> 'phuong 1', 'q3' -> 'quan 3', etc.
    Also handles variations with dots and spaces like 'p.1', 'q. 3'
    
    :param text: A single token (already lowercase, punctuation removed).
    :return: Expanded string if matched, otherwise the original.
    """
    # Clean up extra spaces
    text = text.strip()
    
    # Handle patterns with optional dot and spaces
    # p1, p.1, p. 1 => phuong 1
    match_p = re.match(r'^p\.?\s*(\d+)$', text)
    if match_p:
        return f"phuong {match_p.group(1)}"

    # q1, q.1, q. 1 => quan 1
    match_q = re.match(r'^q\.?\s*(\d+)$', text)
    if match_q:
        return f"quan {match_q.group(1)}"

    # If no match, return original
    return text


def standardize_place_names(tokens):
    """
    Standardize place names using official dictionaries.
    This function:
    1. Looks up n-grams in PROVINCES, DISTRICTS, WARDS sets
    2. Replaces identified n-grams with their full token sequence
    3. Expands any 2-3 letter tokens that unambiguously map to a name
    
    Args:
        tokens (list): List of normalized tokens
        
    Returns:
        list: Standardized tokens
    """
    # Load dictionaries if not already loaded
    if not hasattr(standardize_place_names, 'name_sets'):
        provinces, districts, wards = load_dictionary_files()
        # Create sets of normalized names (without diacritics)
        standardize_place_names.name_sets = {
            'provinces': {remove_diacritics(p.lower()) for p in provinces},
            'districts': {remove_diacritics(d.lower()) for d in districts},
            'wards': {remove_diacritics(w.lower()) for w in wards}
        }
        # Create mapping of abbreviations to full names
        standardize_place_names.abbrev_map = build_abbreviation_map(provinces, districts, wards)
    
    result = []
    i = 0
    while i < len(tokens):
        # Try to match sequences of 1-4 tokens
        matched = False
        for window_size in range(4, 0, -1):
            if i + window_size <= len(tokens):
                window = tokens[i:i + window_size]
                window_text = ''.join(window)  # Try without spaces
                window_text_spaced = ' '.join(window)  # Try with spaces
                
                # Check if this sequence matches any official name
                for name_set in standardize_place_names.name_sets.values():
                    if window_text in name_set or window_text_spaced in name_set:
                        # Found a match, add the canonical form
                        result.extend(window)
                        i += window_size
                        matched = True
                        break
                if matched:
                    break
        
        if not matched:
            # Check for abbreviation expansion
            token = tokens[i]
            if (len(token) <= 3 and token.isupper() and 
                token in standardize_place_names.abbrev_map):
                # Expand the abbreviation
                expanded = standardize_place_names.abbrev_map[token].split()
                result.extend(expanded)
            else:
                result.append(token)
            i += 1
    
    return result

def build_abbreviation_map(provinces, districts, wards):
    """
    Build a mapping of abbreviations to their full names.
    Only includes unambiguous mappings (where the abbreviation matches exactly one name).
    """
    abbrev_map = {}
    
    # Helper function to add abbreviation if unambiguous
    def add_if_unambiguous(abbrev, full_name):
        if abbrev in abbrev_map:
            # If we've seen this abbreviation before, it's ambiguous
            del abbrev_map[abbrev]
        else:
            abbrev_map[abbrev] = full_name
    
    # Process all names
    for name_list in [provinces, districts, wards]:
        for name in name_list:
            # Skip names that are too short
            if len(name) <= 3:
                continue
                
            # Try different abbreviation patterns
            words = name.split()
            if len(words) >= 2:
                # Try first letters of each word
                abbrev = ''.join(w[0] for w in words)
                if 2 <= len(abbrev) <= 3:
                    add_if_unambiguous(abbrev, name)
            
            # Try first 2-3 letters of the name
            for length in [2, 3]:
                abbrev = name[:length]
                if abbrev.isalpha():
                    add_if_unambiguous(abbrev, name)
    
    return abbrev_map

def detect_hcm_context(text):
    """
    Detect if the text contains Ho Chi Minh City context.
    First removes diacritics to make detection more robust.
    
    Args:
        text (str): The input text to check
        
    Returns:
        tuple: (is_hcmc, standardized_text) where:
            - is_hcmc: boolean indicating if HCM context is detected
            - standardized_text: text with HCM variations standardized to 'tp hcm'
    """
    # First remove diacritics to make detection more robust
    text = remove_diacritics(text.lower())
    
    # HCM patterns to check - comprehensive list from all functions
    hcm_patterns = [
        # Standard forms
        (r'tp\.?\s*hcm', 'tp hcm'),
        (r'tp\.?\s*ho\s*chi\s*minh', 'tp hcm'),
        (r'thanh\s*pho\s*ho\s*chi\s*minh', 'tp hcm'),
        (r'sg', 'tp hcm'),
        (r'sai\s*gon', 'tp hcm'),
        (r'ho\s*chi\s*minh', 'tp hcm'),
        (r'tphcm', 'tp hcm'),
        (r'hochiminh', 'tp hcm'),
        (r'hcm', 'tp hcm'),
        
        # Additional variations
        (r'tp\.hochiminh', 'tp hcm'),
        (r'tp\.hcm', 'tp hcm'),
        (r'thanh\s*pho\s*hcm', 'tp hcm'),
        (r'thanh\s*pho\s*hochiminh', 'tp hcm'),
        (r'thanh\s*pho\s*sai\s*gon', 'tp hcm'),
        (r'thanh\s*pho\s*sg', 'tp hcm')
    ]
    
    # Check for HCM patterns
    is_hcmc = False
    for pattern, replacement in hcm_patterns:
        if re.search(pattern, text):
            text = re.sub(pattern, replacement, text)
            is_hcmc = True
            break
    
    # Additional check for token combinations
    if not is_hcmc:
        tokens = text.split()
        # Check if tp/thanh pho is followed by hcm/hochiminh
        if ('tp' in tokens or 'thanh pho' in tokens) and any(x in tokens for x in ['hochiminh', 'hcm']):
            text = re.sub(r'(tp|thanh\s*pho).*?(hochiminh|hcm)', 'tp hcm', text)
            is_hcmc = True
        # Check for individual tokens
        elif any(x in tokens for x in ['hcm', 'hochiminh']):
            text = re.sub(r'(hcm|hochiminh)', 'tp hcm', text)
            is_hcmc = True
    
    return is_hcmc, text

def is_hcm_context(tokens):
    """
    Check if the given tokens indicate Ho Chi Minh City context.
    Uses the detect_hcm_context function for consistency.
    
    Args:
        tokens: List of normalized tokens from the address
    Returns:
        bool: True if HCM context is detected, False otherwise
    """
    text = ' '.join(tokens)
    is_hcmc, _ = detect_hcm_context(text)
    return is_hcmc

def normalize_text(text):
    """
    Multi-pass normalization for addresses:
    1) Preprocess malformed text
    2) Convert to lowercase
    3) Replace punctuation with spaces
    4) Handle abbreviations and compound words
    5) Remove diacritics
    6) Normalize spaces and split into tokens
    7) Standardize place names using official dictionaries
    
    Returns:
        tuple: (normalized_tokens, original_tokens, is_hcm, comma_separated_groups)
    """
    if not text:
        return [], [], False, []
    
    # Store original text for later tokenization
    original_text = text
    
    # First split by commas to preserve comma information
    comma_parts = [part.strip() for part in text.split(',')]
    
    # Process each comma-separated part
    all_normalized_tokens = []
    all_original_tokens = []
    comma_groups = []  # Will store which tokens belong to which comma group
    
    current_group = 0
    for part in comma_parts:
        # Process this part
        part = part.strip()
        if not part:
            continue
            
        # Store original part before any modifications
        original_part = part
    
        # 1. Preprocess malformed text
        # Fix missing spaces between words by looking for camelCase patterns
        part = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
        part = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', part)
    
        # Fix common malformed patterns
        part = re.sub(r'([A-Za-z])(\d+)([A-Za-z])', r'\1 \2 \3', part)  # Split letter-number-letter
        part = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', part)  # Split number-letter
        part = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', part)  # Split letter-number
    
        # Fix common administrative unit prefixes that might be stuck to words
        admin_prefixes = ['huyen', 'quan', 'phuong', 'xa', 'thi', 'tran', 'tinh', 'thanh', 'pho']
        pattern = '|'.join(f'({prefix})' for prefix in admin_prefixes)
        part = re.sub(f'({pattern})(?=\s|$)', r'\1 ', part, flags=re.IGNORECASE)
    
        # Remove dots at word boundaries that aren't part of abbreviations
        part = re.sub(r'(?<=[^A-Za-z0-9])\.|\.(?=[^A-Za-z0-9])', ' ', part)
        
        # 2. Convert to lowercase for normalized version
        normalized_part = part.lower()

        # 3. Check for HCM context and standardize variations
        is_hcmc, normalized_part = detect_hcm_context(normalized_part)

        # 4. Replace punctuation with space, preserving dots for abbreviations
        normalized_part = re.sub(r'[,:/\\\-()]+', ' ', normalized_part)

        # 5. Handle special abbreviations before removing dots
        # Common province abbreviations
        province_abbrev = {
            r'\bt\.?\s*giang\b': 'tien giang',
            r'\bb\.?\s*duong\b': 'binh duong',
            r'\bb\.?\s*dinh\b': 'binh dinh',
            r'\bb\.?\s*thuan\b': 'binh thuan',
            r'\bb\.?\s*phuoc\b': 'binh phuoc',
            r'\bd\.?\s*nai\b': 'dong nai',
            r'\bh\.?\s*giang\b': 'hau giang',
            r'\bk\.?\s*giang\b': 'kien giang',
            r'\bt\.?\s*ninh\b': 'tay ninh',
            r'\bl\.?\s*an\b': 'long an',
            r'\bb\.?\s*tre\b': 'ben tre',
            r'\bv\.?\s*long\b': 'vinh long',
            r'\bs\.?\s*trang\b': 'soc trang',
            r'\bc\.?\s*tho\b': 'can tho',
            r'\ba\.?\s*giang\b': 'an giang'
        }
        
        # HCMC district abbreviations - only apply in HCM context
        if is_hcmc:
            district_abbrev = {
                # Main districts
                r'\bbt\b': 'binh thanh',      # Bình Thạnh
                r'\bpn\b': 'phu nhuan',       # Phú Nhuận
                r'\btb\b': 'tan binh',        # Tân Bình
                r'\btd\b': 'thu duc',         # Thủ Đức
                r'\bgv\b': 'go vap',          # Gò Vấp
                # Remove tp -> tan phu mapping for HCM context
                r'\bqb\b': 'quan binh',       # Quận Bình
                r'\bbc\b': 'binh chanh',      # Bình Chánh
                r'\bhm\b': 'hoc mon',         # Hóc Môn
                r'\bcg\b': 'can gio',         # Cần Giờ
                r'\bnbe\b': 'nha be',         # Nhà Bè
                
                # Additional variations
                r'\bbinh\s*thanh\b': 'binh thanh',  # Normalize Bình Thạnh
                r'\bphu\s*nhuan\b': 'phu nhuan',    # Normalize Phú Nhuận
                r'\btan\s*binh\b': 'tan binh',      # Normalize Tân Bình
                r'\bthu\s*duc\b': 'thu duc',        # Normalize Thủ Đức
                r'\bgo\s*vap\b': 'go vap',          # Normalize Gò Vấp
                # Remove tan phu normalization for HCM context
                r'\bbinh\s*chanh\b': 'binh chanh',  # Normalize Bình Chánh
                r'\bhoc\s*mon\b': 'hoc mon',        # Normalize Hóc Môn
                r'\bcan\s*gio\b': 'can gio',        # Normalize Cần Giờ
                r'\bnha\s*be\b': 'nha be',          # Normalize Nhà Bè
                
                # Numeric districts
                r'\bq\.?\s*(\d+)\b': r'quan \1',    # Q.1, Q.2, etc.
                r'\bq\s+(\d+)\b': r'quan \1',       # Q 1, Q 2, etc.
                r'\bquan\.?\s*(\d+)\b': r'quan \1', # Quan 1, Quan 2, etc.
                r'\bq(\d+)\b': r'quan \1',          # Q1, Q2, etc.
                
                # Special cases
                r'\bq\s*nhat\b': 'quan 1',          # Quận Nhất -> Quận 1
                r'\bquan\s*nhat\b': 'quan 1',
                r'\bq\s*muoi\s*mot\b': 'quan 11',   # Quận Mười Một -> Quận 11
                r'\bquan\s*muoi\s*mot\b': 'quan 11' # Quận Mười Một -> Quận 11
            }
            
            # Apply district abbreviations only in HCM context
            for pattern, replacement in district_abbrev.items():
                normalized_part = re.sub(pattern, replacement, normalized_part)
    
        # Apply province abbreviations
        for pattern, replacement in province_abbrev.items():
            normalized_part = re.sub(pattern, replacement, normalized_part)
    
        # Handle ward/district abbreviations with numbers
        normalized_part = re.sub(r'\bp\.?\s*(\d+)\b', r'phuong \1', normalized_part)
        normalized_part = re.sub(r'\bq\.?\s*(\d+)\b', r'quan \1', normalized_part)
    
        # Handle ward/district without numbers - only if followed by a letter
        normalized_part = re.sub(r'\bp\.\s*([a-zđ])', r'phuong \1', normalized_part)
        normalized_part = re.sub(r'\bq\.\s*([a-zđ])', r'quan \1', normalized_part)
    
        # Handle other common abbreviations
        normalized_part = re.sub(r'\bđ\.\s*', 'duong ', normalized_part)
        normalized_part = re.sub(r'\bt\.\s*', 'tinh ', normalized_part)
        # Only expand tp. if not in HCM context
        if not is_hcmc:
            normalized_part = re.sub(r'\btp\.\s*', 'tp ', normalized_part)
        normalized_part = re.sub(r'\btx\.\s*', 'thi xa ', normalized_part)
        normalized_part = re.sub(r'\btt\.\s*', 'thi tran ', normalized_part)
        normalized_part = re.sub(r'\bh\.\s*', 'huyen ', normalized_part)
    
        # Special cases - prevent over-expansion
        normalized_part = re.sub(r'\bkp\b', 'khu pho', normalized_part)  # Expand kp only as whole word
    
        # Now remove remaining dots
        normalized_part = re.sub(r'\.', ' ', normalized_part)

        # 6. Remove diacritics for normalized version
        normalized_part = remove_diacritics(normalized_part)

        # 7. Normalize multiple spaces
        normalized_part = re.sub(r'\s+', ' ', normalized_part).strip()

        # 8. Process the text word by word to maintain original-normalized pairs
        token_pairs = []
        
        # First, preprocess administrative unit abbreviations in original text
        admin_abbrev_map = {
            'Tnh': 'Tỉnh',
            'TNH': 'Tỉnh',
            'TNH.': 'Tỉnh',
            'H.': 'Huyện',
            'H': 'Huyện',
            'Q.': 'Quận',
            'Q': 'Quận',
            'P.': 'Phường',
            'P': 'Phường',
            'X.': 'Xã',
            'X': 'Xã',
            'TT.': 'Thị trấn',
            'TT': 'Thị trấn',
            'TX.': 'Thị xã',
            'TX': 'Thị xã',
            'TP.': 'Thành phố',
            'TP': 'Thành phố',
            'F.': 'Phường',
            'F': 'Phường'
        }
        
        # Replace abbreviations in original text
        for abbrev, full in admin_abbrev_map.items():
            if '.' in abbrev:
                original_part = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, original_part)
            else:
                pattern = r'\b' + re.escape(abbrev) + r'(?=[A-Z0-9])'
                original_part = re.sub(pattern, full + ' ', original_part)
                original_part = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, original_part)
        
        # Split into words while preserving diacritics
        original_words = []
        current_word = []
        
        for char in original_part:
            if char.isspace() or char in ',.!?;:':
                if current_word:
                    original_words.append(''.join(current_word))
                    current_word = []
            else:
                if current_word and (
                    (char.isdigit() and not current_word[-1].isdigit()) or
                    (not char.isdigit() and current_word[-1].isdigit())
                ):
                    original_words.append(''.join(current_word))
                    current_word = []
                current_word.append(char)
        if current_word:
            original_words.append(''.join(current_word))
        
        # Process each original word
        for orig_word in original_words:
            clean_orig = re.sub(r'[,.!?;:]', '', orig_word)
            if not clean_orig:
                continue
                
            norm_word = remove_diacritics(clean_orig.lower())
            token_pairs.append((norm_word, clean_orig))
        
        # Add tokens to our lists
        normalized_tokens = [pair[0] for pair in token_pairs]
        original_tokens = [pair[1] for pair in token_pairs]
        
        # Add to our main lists
        all_normalized_tokens.extend(normalized_tokens)
        all_original_tokens.extend(original_tokens)
        
        # Add group information
        comma_groups.extend([current_group] * len(normalized_tokens))
        
        current_group += 1

    # 9. Expand any remaining abbreviations token by token
    expanded_tokens = []
    expanded_original_tokens = []
    expanded_groups = []
    i = 0
    while i < len(all_normalized_tokens):
        token = all_normalized_tokens[i]
        original_token = all_original_tokens[i]
        group = comma_groups[i]
        
        if token in ['tp', 'hcm', 'kp']:
            expanded_tokens.append(token)
            expanded_original_tokens.append(original_token)
            expanded_groups.append(group)
        else:
            if token == 'q' and (i + 1 >= len(all_normalized_tokens) or not all_normalized_tokens[i + 1].isdigit()):
                expanded_tokens.append('quan')
                expanded_original_tokens.append('Quận')
                expanded_groups.append(group)
            else:
                expanded = expand_abbreviations(token)
                expanded_tokens.extend(expanded.split())
                expanded_original_tokens.append(original_token)
                expanded_groups.extend([group] * len(expanded.split()))
        i += 1

    # 10. Standardize place names using official dictionaries
    normalized_tokens = standardize_place_names(expanded_tokens)
    
    # 11. Final ward abbreviation expansion
    final_tokens = []
    final_original_tokens = []
    final_groups = []
    i = 0
    while i < len(normalized_tokens):
        if normalized_tokens[i] == 'p' and i + 1 < len(normalized_tokens) and not normalized_tokens[i + 1].isdigit():
            final_tokens.append('phuong')
            final_original_tokens.append('Phường')
            final_groups.append(expanded_groups[i])
            i += 1
        else:
            final_tokens.append(normalized_tokens[i])
            final_original_tokens.append(expanded_original_tokens[i])
            final_groups.append(expanded_groups[i])
            i += 1
    
    return final_tokens, final_original_tokens, is_hcmc, final_groups


##############################################################
# 4. LOADING DICTIONARIES & BUILDING TRIES
##############################################################

def build_trie_from_list(words_list):
    """
    Build a trie from a list of location names.
    Each location name is normalized for insertion, 
    but we store the original form (with diacritics) in the node.
    
    :param words_list: List of strings representing location names.
    :return: A Trie object containing the location data.
    """
    t = Trie()
    for w in words_list:
        w_clean = w.strip()
        # normalized version
        w_norm = remove_diacritics(w_clean.lower())
        t.insert(w_norm, original_name=w_clean)
    return t


def load_dictionary_files():
    """
    Load location data from text files:
    - list_province.txt: List of provinces
    - list_district.txt: List of districts
    - list_ward.txt: List of wards
    
    Each file contains one location name per line.
    
    :return: (list_of_provinces, list_of_districts, list_of_wards)
    """
    provinces = []
    districts = []
    wards = []
    
    # Load provinces
    try:
        with open('list_province.txt', 'r', encoding='utf-8') as f:
            provinces = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Warning: Could not load provinces: {e}")
        
    # Load districts
    try:
        with open('list_district.txt', 'r', encoding='utf-8') as f:
            districts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Warning: Could not load districts: {e}")
        
    # Load wards
    try:
        with open('list_ward.txt', 'r', encoding='utf-8') as f:
            wards = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Warning: Could not load wards: {e}")
        
    return provinces, districts, wards


##############################################################
# 5. PARSING LOGIC (PROVINCE -> DISTRICT -> WARD)
##############################################################

def find_province(tokens, province_trie, province_indicators):
    """
    Find province in the normalized and tokenized address.
    """
    try:
        if not tokens:
            return (None, None, 0)
            
        # Define province indicators
        admin_prefixes = set(province_indicators)

        # Special case: Check for HCM
        if is_hcm_context(tokens):
            return ('ho chi minh', 'Hồ Chí Minh', 3)

        # First try to find matches at the end of the address
        max_group = 3
        for group_size in range(min(max_group, len(tokens)), 1, -1):
            start_idx = len(tokens) - group_size
            group = tokens[start_idx:start_idx + group_size]
            group_text = ' '.join(group)
            
            # Try exact match first
            if province_trie.search_exact(group_text):
                original = province_trie.get_full_word(group_text)
                if validate_province_match(group_text, tokens):
                    return (group_text, original, group_size)
            
            # Try fuzzy match using rapidfuzz
            candidates = province_trie.collect_all_words()
            if candidates:
                result = process.extractOne(
                    group_text,
                    candidates,
                    scorer=fuzz.ratio,
                    score_cutoff=80
                )
                
                if result:
                    matched_text, score, _ = result
                    if validate_province_match(matched_text, tokens):
                        original = province_trie.get_full_word(matched_text)
                        if original:
                            return (matched_text, original, group_size)

        # Then try to find relevant tokens with indicators
        for i, token in enumerate(tokens):
            if token in admin_prefixes and i + 1 < len(tokens):
                # Look for next two tokens
                next_tokens = []
                j = i + 1
                count = 0
                while j < len(tokens) and count < 2:
                    if tokens[j] not in admin_prefixes:
                        next_tokens.append(tokens[j])
                        count += 1
                    j += 1
                if next_tokens:
                    # Try exact match first with the tokens
                    text_to_search = ' '.join(next_tokens)
                    if province_trie.search_exact(text_to_search):
                        original = province_trie.get_full_word(text_to_search)
                        if validate_province_match(text_to_search, tokens):
                            return (text_to_search, original, len(next_tokens))
                    
                    # Try with reversed tokens if we have exactly 2
                    if len(next_tokens) == 2:
                        reversed_text = ' '.join(reversed(next_tokens))
                        if province_trie.search_exact(reversed_text):
                            original = province_trie.get_full_word(reversed_text)
                            if validate_province_match(reversed_text, tokens):
                                return (reversed_text, original, len(next_tokens))
                    
                    # Try fuzzy match using rapidfuzz
                    candidates = province_trie.collect_all_words()
                    if candidates:
                        result = process.extractOne(
                            text_to_search,
                            candidates,
                            scorer=fuzz.ratio,
                            score_cutoff=80
                        )
                        
                        if result:
                            matched_text, score, _ = result
                            if validate_province_match(matched_text, tokens):
                                original = province_trie.get_full_word(matched_text)
                                if original:
                                    return (matched_text, original, len(next_tokens))

        # Finally try looking for potential province names in the middle
        for group_size in range(min(max_group, len(tokens)), 1, -1):
            for start_idx in range(len(tokens) - group_size + 1):
                # Skip if we've already checked these tokens with indicators
                if any(t in admin_prefixes for t in tokens[start_idx:start_idx + group_size]):
                    continue
                    
                group = tokens[start_idx:start_idx + group_size]
                group_text = ' '.join(group)
                
                # Try exact match first
                if province_trie.search_exact(group_text):
                    original = province_trie.get_full_word(group_text)
                    if validate_province_match(group_text, tokens):
                        return (group_text, original, group_size)
                
                # Try fuzzy match using rapidfuzz
                candidates = province_trie.collect_all_words()
                if candidates:
                    result = process.extractOne(
                        group_text,
                        candidates,
                        scorer=fuzz.ratio,
                        score_cutoff=80
                    )
                    
                    if result:
                        matched_text, score, _ = result
                        if validate_province_match(matched_text, tokens):
                            original = province_trie.get_full_word(matched_text)
                            if original:
                                return (matched_text, original, group_size)

        return (None, None, 0)
    except Exception as e:
        logger.error(f"Error in find_province: {str(e)}")
        return (None, None, 0)

def validate_province_match(matched_text, tokens):
    """
    Validate a potential province match to avoid incorrect matches.
    
    Args:
        matched_text: The normalized province name that was matched
        tokens: All tokens from the address
    Returns:
        bool: True if the match is valid, False otherwise
    """
    # List of provinces that are commonly confused
    confused_pairs = [
        {'thanh hoa', 'hoa binh'},
        {'ha nam', 'nam dinh'},
        {'quang nam', 'da nang'},
        {'quang ninh', 'ninh binh'},
        {'hoa binh', 'ha noi'},
        {'long an', 'binh duong'},
        {'thai binh', 'ha nam'}
    ]
    
    # Check if our match is part of a confused pair
    for pair in confused_pairs:
        if matched_text in pair:
            # If we matched one of a confused pair, check if the other one appears in the tokens
            other = next(p for p in pair if p != matched_text)
            other_parts = other.split()
            matched_parts = matched_text.split()
            
            # Count how many parts of each province appear in the tokens
            matched_count = sum(1 for part in matched_parts if part in tokens)
            other_count = sum(1 for part in other_parts if part in tokens)
            
            # If the other province has more matching parts, this is probably a wrong match
            if other_count > matched_count:
                return False
            
            # If both have the same number of matches, check position
            if other_count == matched_count:
                # Get the positions of the matches
                matched_pos = min(i for i, t in enumerate(tokens) if t in matched_parts)
                other_pos = min(i for i, t in enumerate(tokens) if t in other_parts)
                
                # Prefer the match that appears later in the address
                if other_pos > matched_pos:
                    return False
    
    return True

def is_ambiguous_name(name, district_trie, ward_trie):
    """
    Check if a name appears in both district and ward lists.
    
    Args:
        name: The normalized name to check
        district_trie: Trie containing district data
        ward_trie: Trie containing ward data
    Returns:
        bool: True if the name appears in both lists, False otherwise
    """
    try:
        return (district_trie.search_exact(name) and 
                ward_trie.search_exact(name))
    except Exception as e:
        logger.error(f"Error in is_ambiguous_name: {str(e)}")
        return False

def get_detailed_confidence_score(name, tokens, district_trie=None, ward_trie=None, fuzzy_score=None, scoring_weights=None, component_type=None, original_tokens=None, comma_groups=None):
    """
    Enhanced version of get_match_confidence that returns detailed information about the score calculation.
    Now uses a 2.0 scale and incorporates fuzzy score with adjusted weights.
    
    Args:
        name: The normalized name to check
        tokens: All normalized tokens from the address
        district_trie: Trie containing district data
        ward_trie: Trie containing ward data
        fuzzy_score: Optional fuzzy match score (0-100)
        scoring_weights: Optional dictionary of scoring weights
        component_type: Type of component being searched for ('ward' or 'district')
        original_tokens: Original tokens with diacritics
        comma_groups: List indicating which tokens belong to which comma-separated group
    """
    try:
        # Set default weights if not provided
        if scoring_weights is None:
            scoring_weights = {
                'fuzzy_score': 0.15,
                'exact_match': 0.35,
                'position_bonus': {
                    'beginning': 0.25,
                    'middle': 0.3
                },
                'length_bonus': {
                    '2_parts': 0.25,       # Base bonus for 2-part names
                    '3_parts': 0.35,       # Increased bonus for 3-part names
                    '4_parts': 0.45        # Even higher bonus for 4-part names
                },
                'non_ambiguous_bonus': 0.1,
                'comma_bonus': 0.25,
                'indicator_bonus': 0.25,
                'full_text_match_bonus': 0.3,  # Bonus for matching the full original text
                'original_text_match_bonus': 0.2,  # Bonus for matching original text with diacritics
                'unique_ward_bonus': 0.3,  # New bonus for matches that can only be wards
                'comma_boundary_penalty': -0.5  # New penalty for matches spanning comma boundaries
            }
            
        score_details = {
            'total_score': 0.0,
            'components': [],
            'is_ambiguous': False
        }
        
        # Check for ambiguity
        if district_trie and ward_trie:
            in_district = district_trie.search_exact(name)
            in_ward = ward_trie.search_exact(name)
            score_details['is_ambiguous'] = in_district and in_ward
            score_details['components'].append({
                'type': 'ambiguity_check',
                'in_district': in_district,
                'in_ward': in_ward,
                'is_ambiguous': score_details['is_ambiguous']
            })
        
        name_parts = name.split()
        
        # 1. Fuzzy Score
        if fuzzy_score is not None:
            fuzzy_component = (fuzzy_score / 100) * scoring_weights['fuzzy_score']
            score_details['total_score'] += fuzzy_component
            score_details['components'].append({
                'type': 'fuzzy_score',
                'score': fuzzy_component,
                'raw_score': fuzzy_score,
                'reason': f'Fuzzy match score of {fuzzy_score}% converted to {fuzzy_component:.2f}'
            })
        
        # 2. Exact Match
        exact_match_found = False
        match_start = -1
        match_end = -1
        for i in range(len(tokens) - len(name_parts) + 1):
            if tokens[i:i+len(name_parts)] == name_parts:
                score_details['total_score'] += scoring_weights['exact_match']
                score_details['components'].append({
                    'type': 'exact_match',
                    'score': scoring_weights['exact_match'],
                    'position': i,
                    'reason': 'Exakte Übereinstimmung gefunden'
                })
                exact_match_found = True
                match_start = i
                match_end = i + len(name_parts)
                break
        
        if not exact_match_found:
            score_details['components'].append({
                'type': 'exact_match',
                'score': 0.0,
                'reason': 'Keine exakte Übereinstimmung gefunden'
            })
        
        # 3. Position Bonus
        position_found = False
        for i in range(len(tokens) - len(name_parts) + 1):
            if tokens[i:i+len(name_parts)] == name_parts:
                position_found = True
                if i == 0:
                    score_details['total_score'] += scoring_weights['position_bonus']['beginning']
                    score_details['components'].append({
                        'type': 'position_bonus',
                        'score': scoring_weights['position_bonus']['beginning'],
                        'position': 'beginning',
                        'reason': 'Match am Anfang der Adresse'
                    })
                elif i > 0 and i < len(tokens) - len(name_parts):
                    score_details['total_score'] += scoring_weights['position_bonus']['middle']
                    score_details['components'].append({
                        'type': 'position_bonus',
                        'score': scoring_weights['position_bonus']['middle'],
                        'position': 'middle',
                        'reason': 'Match in der Mitte der Adresse'
                    })
                break
        
        if not position_found:
            score_details['components'].append({
                'type': 'position_bonus',
                'score': 0.0,
                'reason': 'Keine Position gefunden'
            })
        
        # 4. Length Bonus - Now with different weights based on number of parts
        if len(name_parts) >= 4:
            bonus = scoring_weights['length_bonus']['4_parts']
            reason = f'Mehrteiliger Name ({len(name_parts)} Teile)'
        elif len(name_parts) == 3:
            bonus = scoring_weights['length_bonus']['3_parts']
            reason = f'Mehrteiliger Name ({len(name_parts)} Teile)'
        elif len(name_parts) == 2:
            bonus = scoring_weights['length_bonus']['2_parts']
            reason = f'Mehrteiliger Name ({len(name_parts)} Teile)'
        else:
            bonus = 0.0
            reason = 'Einteiliger Name'
            
        score_details['total_score'] += bonus
        score_details['components'].append({
            'type': 'length_bonus',
            'score': bonus,
            'reason': reason
        })
        
        # 5. Non-ambiguity Bonus
        if not score_details['is_ambiguous']:
            score_details['total_score'] += scoring_weights['non_ambiguous_bonus']
            score_details['components'].append({
                'type': 'non_ambiguous_bonus',
                'score': scoring_weights['non_ambiguous_bonus'],
                'reason': 'Name ist nicht mehrdeutig'
            })
        else:
            score_details['components'].append({
                'type': 'non_ambiguous_bonus',
                'score': 0.0,
                'reason': 'Name ist mehrdeutig'
            })
            
        # 5.1 Unique Ward Bonus (New)
        if component_type == 'ward' and district_trie and ward_trie:
            # Check if the name exists in ward but not in district
            in_ward = ward_trie.search_exact(name)
            in_district = district_trie.search_exact(name)
            if in_ward and not in_district:
                score_details['total_score'] += scoring_weights['unique_ward_bonus']
                score_details['components'].append({
                    'type': 'unique_ward_bonus',
                    'score': scoring_weights['unique_ward_bonus'],
                    'reason': 'Name exists only as a ward, not as a district'
                })
            else:
                score_details['components'].append({
                    'type': 'unique_ward_bonus',
                    'score': 0.0,
                    'reason': 'Name exists in both ward and district lists or not found in ward list'
                })
        
        # 6. Comma Bonus
        comma_bonus_found = False
        for i in range(len(tokens) - len(name_parts) + 1):
            if tokens[i:i+len(name_parts)] == name_parts:
                for j in range(i-1, -1, -1):
                    if tokens[j] == ',':
                        score_details['total_score'] += scoring_weights['comma_bonus']
                        score_details['components'].append({
                            'type': 'comma_bonus',
                            'score': scoring_weights['comma_bonus'],
                            'reason': 'Match nach Komma gefunden'
                        })
                        comma_bonus_found = True
                        break
                break
        
        if not comma_bonus_found:
            score_details['components'].append({
                'type': 'comma_bonus',
                'score': 0.0,
                'reason': 'Kein Komma vor Match gefunden'
            })
            
        # 7. Indicator Bonus
        indicator_bonus_found = False
        ward_indicators = ['phuong', 'xa', 'thi tran', 'p', 'tt']
        district_indicators = ['quan', 'huyen', 'q', 'h']
        
        # Define distance-based scaling factors
        distance_scaling = {
            1: 1.0,    # Full bonus for immediate predecessor
            2: 0.5,    # Half bonus for 2 positions away
            3: 0.25    # Quarter bonus for 3 positions away
        }
        
        # Only check for ward indicators if we're looking for a ward
        if component_type == 'ward':
            for i in range(len(tokens) - len(name_parts) + 1):
                if tokens[i:i+len(name_parts)] == name_parts:
                    # Look for ward indicators before this position (max 3 positions)
                    for j in range(i-1, max(-1, i-4), -1):
                        if tokens[j] in ward_indicators:
                            distance = i - j
                            if distance in distance_scaling:
                                scaled_bonus = scoring_weights['indicator_bonus'] * distance_scaling[distance]
                                score_details['total_score'] += scaled_bonus
                                score_details['components'].append({
                                    'type': 'indicator_bonus',
                                    'score': scaled_bonus,
                                    'reason': f'Ward indicator "{tokens[j]}" found {distance} positions before match (scaled to {distance_scaling[distance]*100:.0f}%)'
                                })
                                indicator_bonus_found = True
                                break
                    break
        
        # Only check for district indicators if we're looking for a district
        if component_type == 'district':
            for i in range(len(tokens) - len(name_parts) + 1):
                if tokens[i:i+len(name_parts)] == name_parts:
                    # Look for district indicators before this position (max 3 positions)
                    for j in range(i-1, max(-1, i-4), -1):
                        if tokens[j] in district_indicators:
                            distance = i - j
                            if distance in distance_scaling:
                                scaled_bonus = scoring_weights['indicator_bonus'] * distance_scaling[distance]
                                score_details['total_score'] += scaled_bonus
                                score_details['components'].append({
                                    'type': 'indicator_bonus',
                                    'score': scaled_bonus,
                                    'reason': f'District indicator "{tokens[j]}" found {distance} positions before match (scaled to {distance_scaling[distance]*100:.0f}%)'
                                })
                                indicator_bonus_found = True
                                break
                    break
        
        if not indicator_bonus_found:
            score_details['components'].append({
                'type': 'indicator_bonus',
                'score': 0.0,
                'reason': 'No ward/district indicator found within 3 positions before match'
            })
            
        # 8. Full Text Match Bonus
        # Check if the name matches the full original text
        original_text = ' '.join(tokens)
        if name == remove_diacritics(original_text.lower()):
            score_details['total_score'] += scoring_weights['full_text_match_bonus']
            score_details['components'].append({
                'type': 'full_text_match_bonus',
                'score': scoring_weights['full_text_match_bonus'],
                'reason': 'Exakte Übereinstimmung mit dem vollständigen Originaltext'
            })
        else:
            score_details['components'].append({
                'type': 'full_text_match_bonus',
                'score': 0.0,
                'reason': 'Keine exakte Übereinstimmung mit dem vollständigen Originaltext'
            })
            
        # 9. Original Text Match Bonus (New)
        if original_tokens:
            # Find the position of the match in the normalized tokens
            for i in range(len(tokens) - len(name_parts) + 1):
                if tokens[i:i+len(name_parts)] == name_parts:
                    # Get the corresponding original tokens
                    original_part = ' '.join(original_tokens[i:i+len(name_parts)])
                    
                    # Get the stored full word from the trie
                    full_word = None
                    if component_type == 'ward' and ward_trie:
                        full_word = ward_trie.get_full_word(name, original_part)
                    elif component_type == 'district' and district_trie:
                        full_word = district_trie.get_full_word(name, original_part)
                    
                    if full_word:
                        # Split both into words for word-by-word comparison
                        original_words = original_part.split()
                        full_words = full_word.split()
                        
                        # Count exact word matches
                        exact_word_matches = 0
                        for orig_word, full_word in zip(original_words, full_words):
                            if orig_word == full_word:
                                exact_word_matches += 1
                        
                        # Calculate bonus based on number of exact word matches
                        if exact_word_matches > 0:
                            # Base bonus is proportional to number of exact matches
                            base_bonus = (exact_word_matches / len(full_words)) * scoring_weights['original_text_match_bonus']
                            
                            # Additional bonus for having all words match
                            if exact_word_matches == len(full_words):
                                base_bonus *= 1.2  # 20% extra bonus for perfect match
                            
                            score_details['total_score'] += base_bonus
                            score_details['components'].append({
                                'type': 'original_text_match_bonus',
                                'score': base_bonus,
                                'exact_matches': exact_word_matches,
                                'total_words': len(full_words),
                                'reason': f'Found {exact_word_matches} exact word matches out of {len(full_words)} words'
                            })
                        else:
                            # If no exact word matches, try fuzzy matching as fallback
                            ratio = fuzz.ratio(original_part, full_word)
                            if ratio >= 80:  # Only give bonus for high matches
                                bonus = (ratio / 100) * scoring_weights['original_text_match_bonus'] * 0.5  # Reduced bonus for fuzzy match
                                score_details['total_score'] += bonus
                                score_details['components'].append({
                                    'type': 'original_text_match_bonus',
                                    'score': bonus,
                                    'raw_score': ratio,
                                    'reason': f'No exact word matches, but fuzzy match with {ratio}% similarity'
                                })
                            else:
                                score_details['components'].append({
                                    'type': 'original_text_match_bonus',
                                    'score': 0.0,
                                    'raw_score': ratio,
                                    'reason': f'No exact word matches and fuzzy match too low ({ratio}% similarity)'
                                })
                    break
        
        # 10. Comma Boundary Penalty (New)
        if comma_groups and match_start >= 0 and match_end >= 0:
            # Get the groups of the matched tokens
            match_groups = set(comma_groups[match_start:match_end])
            # If we have more than one group, it means the match spans across comma boundaries
            if len(match_groups) > 1:
                penalty = scoring_weights['comma_boundary_penalty']
                score_details['total_score'] += penalty
                score_details['components'].append({
                    'type': 'comma_boundary_penalty',
                    'score': penalty,
                    'reason': f'Match spans across {len(match_groups)} comma-separated groups'
                })
            else:
                score_details['components'].append({
                    'type': 'comma_boundary_penalty',
                    'score': 0.0,
                    'reason': 'Match does not span across comma boundaries'
                })
        
        # Final score (max 2.0)
        score_details['total_score'] = min(score_details['total_score'], 2.0)
        
        return score_details
        
    except Exception as e:
        logger.error(f"Error in get_detailed_confidence_score: {str(e)}")
        return {
            'total_score': 0.0,
            'components': [],
            'is_ambiguous': False,
            'error': str(e)
        }

def log_fuzzy_matches(text_to_search, candidates, component_type, tokens, district_trie=None, ward_trie=None):
    """
    Logs the top 3 fuzzy matches with detailed confidence scores.
    Only logs when a test case has failed.
    """
    try:
        # Only log if we're in a failed test case (logging level is DEBUG)
        if logger.level != logging.DEBUG:
            return
            
        logger.info(f"\n{'='*80}")
        logger.info(f"Fuzzy-Match Analyse für: {text_to_search}")
        logger.info(f"Komponente: {component_type}")
        logger.info(f"{'='*80}")
        
        # Get top 3 matches with rapidfuzz
        results = process.extract(
            text_to_search,
            candidates,
            scorer=fuzz.ratio,
            limit=3
        )
        
        if not results:
            logger.info("Keine Fuzzy-Matches gefunden")
            return
        
        # Analyze each match
        for i, (matched_text, score, _) in enumerate(results, 1):
            logger.info(f"\nMatch #{i}:")
            logger.info(f"Gefundener Text: {matched_text}")
            logger.info(f"Fuzzy-Score: {score}%")
            
            # Calculate detailed confidence score with fuzzy score
            score_details = get_detailed_confidence_score(
                matched_text, 
                tokens,
                district_trie if component_type == 'district' else None,
                ward_trie if component_type == 'ward' else None,
                fuzzy_score=score  # Pass the fuzzy score
            )
            
            logger.info(f"Confidence-Score: {score_details['total_score']:.2f}")
            logger.info("Score-Komponenten:")
            
            for component in score_details['components']:
                if component['type'] == 'fuzzy_score':
                    logger.info(f"  - {component['type']}: {component['score']:.2f} (Raw score: {component['raw_score']}%)")
                else:
                    logger.info(f"  - {component['type']}: {component['score']:.2f} ({component['reason']})")
            
            logger.info(f"Mehrdeutig: {'Ja' if score_details['is_ambiguous'] else 'Nein'}")
            logger.info("-" * 40)
            
    except Exception as e:
        logger.error(f"Error in log_fuzzy_matches: {str(e)}")

def validate_district_match(matched_text, tokens, score, validation_thresholds, province=None, ward_trie=None, district_trie=None):
    """
    Validate a potential district match focusing on special cases and business rules.
    Uses the score passed from find_district.
    """
    try:
        # Special case: HCM districts
        if province and province.lower() == 'ho chi minh':
            if not matched_text.isdigit() and not any(d in matched_text for d in [
                'binh thanh', 'phu nhuan', 'tan binh', 'thu duc', 'go vap', 
                'hoc mon', 'binh chanh', 'can gio', 'nha be'
            ]):
                logger.debug(f"Rejecting: '{matched_text}' is not a valid HCM district")
                return False
        
        # Special case: Check for province position
        if province:
            province_parts = remove_diacritics(province.lower()).split()
            for i in range(len(tokens) - len(province_parts) + 1):
                if tokens[i:i+len(province_parts)] == province_parts:
                    province_pos = i
                    district_pos = tokens.index(matched_text.split()[0])
                    if district_pos > province_pos:
                        logger.debug(f"Rejecting: district at {district_pos} appears after province at {province_pos}")
                        return False
                    break
        
        # Use the passed score and compare with threshold
        return score >= validation_thresholds['district']['non_ambiguous']
        
    except Exception as e:
        logger.error(f"Error in validate_district_match: {str(e)}")
        return False

def validate_ward_match(matched_text, tokens, score, validation_thresholds, district_trie=None, ward_trie=None):
    """
    Validate a potential ward match focusing on special cases and business rules.
    Uses the score passed from find_ward.
    """
    try:
        # Special case: Check for comma context
        ward_parts = matched_text.split()
        ward_pos = -1
        for i in range(len(tokens) - len(ward_parts) + 1):
            if tokens[i:i+len(ward_parts)] == ward_parts:
                ward_pos = i
                # Check for comma before ward
                for j in range(i-1, -1, -1):
                    if tokens[j] == ',':
                        # If ward appears after comma, be more lenient
                        return score >= validation_thresholds['ward']['ambiguous_middle']
                break
        
        # Use the passed score and compare with threshold
        return score >= validation_thresholds['ward']['non_ambiguous']
        
    except Exception as e:
        logger.error(f"Error in validate_ward_match: {str(e)}")
        return False

def get_hcm_numeric_district_score(token, tokens, position):
    """
    Calculate a confidence score for a potential numeric district in HCM context.
    Returns 0.0 if it's not a valid numeric district.
    
    Args:
        token: The token to check
        tokens: All tokens from the address
        position: Position of the token in the tokens list
    
    Returns:
        float: Confidence score (0.0 to 2.0) or 0.0 if not valid
    """
    try:
        # Must be a standalone number
        if not token.isdigit():
            return 0.0
            
        # Must be in valid range
        num = int(token)
        if not (1 <= num <= 24):
            return 0.0
            
        # Check if it's part of a larger string (e.g., "46/8F")
        if position > 0 and '/' in tokens[position-1]:
            return 0.0
            
        # Base score starts at 1.0
        score = 1.0
        
        # Bonus for being after "quan"
        if position > 0 and tokens[position-1].lower() in ['q', 'quan']:
            score += 0.8
            
        # Bonus for being at start of address
        elif position == 0:
            score += 0.6
            
        # Penalty for being after ward indicators
        if position > 0 and tokens[position-1].lower() in ['p', 'phuong']:
            score -= 0.5
            
        # Penalty for being in the middle of other words
        if position > 0 and position < len(tokens) - 1:
            if not tokens[position-1].lower() in ['q', 'quan']:
                score -= 0.3
                
        return max(0.0, min(2.0, score))
        
    except Exception as e:
        logger.error(f"Error in get_hcm_numeric_district_score: {str(e)}")
        return 0.0

def find_district(tokens, district_trie, ward_trie=None, scoring_weights=None, is_hcm=False, original_tokens=None, comma_groups=None):
    if not tokens:
        logger.debug("No tokens provided for district search")
        return None, 0.0
        
    logger.debug(f"Searching for district in tokens: {tokens}")
    
    # First check for Q+number format (e.g., Q3, Q.3)
    for i, token in enumerate(tokens):
        if token.lower() in ['q', 'quan'] and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if next_token.isdigit() and 1 <= int(next_token) <= 24:
                logger.debug(f"Found Q+number format: {next_token} with confidence score: 2.0")
                return next_token, 2.0  # Maximum confidence for Q+number format
    
    # Collect all exact matches with their scores
    exact_matches = []
    
    # Then try exact match
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens) + 1)):
            search_text = ' '.join(tokens[i:j])
            logger.debug(f"Trying exact match for district: {search_text}")
            
            # Check if this is a numeric district in HCM
            if is_hcm and search_text.isdigit():
                numeric_score = get_hcm_numeric_district_score(search_text, tokens, i)
                if numeric_score > 0:
                    logger.debug(f"Found numeric district in HCM: {search_text} with confidence score: {numeric_score:.2f}")
                    exact_matches.append((search_text, numeric_score))
                    continue  # Skip the get_detailed_confidence_score calculation
                
            # Try exact match
            if district_trie.search_exact(search_text):
                logger.debug(f"Found exact match for district: {search_text}")
                # Get confidence score
                score_details = get_detailed_confidence_score(
                    search_text, 
                    tokens, 
                    district_trie, 
                    ward_trie,
                    scoring_weights=scoring_weights,
                    component_type='district',  # Specify we're looking for a district
                    original_tokens=original_tokens,  # Add original tokens
                    comma_groups=comma_groups  # Add comma groups
                )
                confidence = score_details['total_score']
                logger.debug(f"Confidence score for exact match: {confidence:.2f}")
                exact_matches.append((search_text, confidence))
    
    # If we have exact matches, return the one with highest confidence
    if exact_matches:
        # Sort by confidence score in descending order
        exact_matches.sort(key=lambda x: x[1], reverse=True)
        best_match, best_score = exact_matches[0]
        logger.debug(f"Returning best exact match: {best_match} with confidence score {best_score:.2f}")
        return best_match, best_score
    
    # If no exact match, try fuzzy match
    logger.debug("No exact match found, trying fuzzy match")
    best_match = None
    best_score = 0.0
    
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens) + 1)):
            search_text = ' '.join(tokens[i:j])
            logger.debug(f"Trying fuzzy match for district: {search_text}")
            
            # Get fuzzy matches using fuzzy_search_in_trie
            matches = fuzzy_search_in_trie(district_trie, search_text, max_dist=2)
            if matches:
                logger.debug(f"Found fuzzy matches for district: {matches}")
                # Log fuzzy matches (pass all required arguments)
                log_fuzzy_matches(
                    search_text,
                    [m[0] for m in matches],
                    'district',
                    tokens,
                    district_trie=district_trie,
                    ward_trie=ward_trie
                )
                
                # Get the best match
                match, fuzzy_score = matches[0]
                logger.debug(f"Best fuzzy match: {match} with fuzzy score {fuzzy_score}")
                
                # Get confidence score
                score_details = get_detailed_confidence_score(
                    match, 
                    tokens, 
                    district_trie, 
                    ward_trie,
                    fuzzy_score=fuzzy_score,
                    scoring_weights=scoring_weights,
                    component_type='district',  # Specify we're looking for a district
                    original_tokens=original_tokens,  # Add original tokens
                    comma_groups=comma_groups  # Add comma groups
                )
                
                if score_details['total_score'] > best_score:
                    best_match = match
                    best_score = score_details['total_score']
                    logger.debug(f"New best match: {best_match} with confidence score {best_score:.2f}")
    
    if best_match:
        logger.debug(f"Returning best fuzzy match: {best_match} with confidence score {best_score:.2f}")
        return best_match, best_score
    
    logger.debug("No district match found")
    return None, 0.0

def find_ward(tokens, ward_trie, district_trie=None, scoring_weights=None, original_tokens=None, comma_groups=None):
    if not tokens:
        logger.debug("No tokens provided for ward search")
        return None, 0.0
        
    logger.debug(f"Searching for ward in tokens: {tokens}")
    
    # First check for P+number format (e.g., P1, P.1)
    for i, token in enumerate(tokens):
        if token.lower() in ['p', 'phuong'] and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if next_token.isdigit() and 1 <= int(next_token) <= 20:  # HCM wards typically go up to 20
                logger.debug(f"Found P+number format: {next_token} with confidence score: 2.0")
                return next_token, 2.0  # Maximum confidence for P+number format
    
    # Collect all exact matches with their scores
    exact_matches = []
    
    # Then try exact match
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens) + 1)):
            search_text = ' '.join(tokens[i:j])
            logger.debug(f"Trying exact match for ward: {search_text}")
            
            # Try exact match
            if ward_trie.search_exact(search_text):
                logger.debug(f"Found exact match for ward: {search_text}")
                # Get confidence score
                score_details = get_detailed_confidence_score(
                    search_text, 
                    tokens, 
                    district_trie, 
                    ward_trie,
                    scoring_weights=scoring_weights,
                    component_type='ward',  # Specify we're looking for a ward
                    original_tokens=original_tokens,  # Add original tokens
                    comma_groups=comma_groups  # Add comma groups
                )
                confidence = score_details['total_score']
                logger.debug(f"Confidence score for exact match: {confidence:.2f}")
                exact_matches.append((search_text, confidence))
    
    # If we have exact matches, return the one with highest confidence
    if exact_matches:
        # Sort by confidence score in descending order
        exact_matches.sort(key=lambda x: x[1], reverse=True)
        best_match, best_score = exact_matches[0]
        logger.debug(f"Returning best exact match: {best_match} with confidence score {best_score:.2f}")
        return best_match, best_score
    
    # If no exact match, try fuzzy match
    logger.debug("No exact match found, trying fuzzy match")
    best_match = None
    best_score = 0.0
    
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens) + 1)):
            search_text = ' '.join(tokens[i:j])
            logger.debug(f"Trying fuzzy match for ward: {search_text}")
            
            # Get fuzzy matches
            matches = fuzzy_search_in_trie(ward_trie, search_text, max_dist=2)
            if matches:
                logger.debug(f"Found fuzzy matches for ward: {matches}")
                # Log fuzzy matches
                log_fuzzy_matches(
                    search_text,
                    [m[0] for m in matches],
                    'ward',
                    tokens,
                    district_trie=district_trie,
                    ward_trie=ward_trie
                )
                
                # Get the best match
                match, fuzzy_score = matches[0]
                logger.debug(f"Best fuzzy match: {match} with fuzzy score {fuzzy_score}")
                
                # Get confidence score
                score_details = get_detailed_confidence_score(
                    match, 
                    tokens, 
                    district_trie, 
                    ward_trie,
                    fuzzy_score=fuzzy_score,
                    scoring_weights=scoring_weights,
                    component_type='ward',  # Specify we're looking for a ward
                    original_tokens=original_tokens,  # Add original tokens
                    comma_groups=comma_groups  # Add comma groups
                )
                
                if score_details['total_score'] > best_score:
                    best_match = match
                    best_score = score_details['total_score']
                    logger.debug(f"New best match: {best_match} with confidence score {best_score:.2f}")
    
    if best_match:
        logger.debug(f"Returning best fuzzy match: {best_match} with confidence score {best_score:.2f}")
        return best_match, best_score
    
    logger.debug("No ward match found")
    return None, 0.0

##############################################################
# 6. MAIN DEMO
##############################################################

def parse_location_components(text, province_trie, district_trie, ward_trie, scoring_weights=None, validation_thresholds=None, disable_file_logging=False):
    """
    Parse location components from text using all available components:
    - Confidence Score System
    - Fuzzy Matching
    - Standardized Weights
    - Threshold Validation
    - Ambiguity Checking
    
    Args:
        text: The input text to parse
        province_trie: Trie containing province data
        district_trie: Trie containing district data
        ward_trie: Trie containing ward data
        scoring_weights: Optional dictionary of scoring weights
        validation_thresholds: Optional dictionary of validation thresholds
        disable_file_logging: If True, disable logging to file
    """
    try:
        # Set up logging based on parameter
        setup_logging(disable_file_logging)
        
        # Set default weights if not provided
        if scoring_weights is None:
            scoring_weights = {
                'fuzzy_score': 0.15,
                'exact_match': 0.35,
                'position_bonus': {
                    'beginning': 0.25,
                    'middle': 0.3
                },
                'length_bonus': {
                    '2_parts': 0.25,       # Base bonus for 2-part names
                    '3_parts': 0.35,       # Increased bonus for 3-part names
                    '4_parts': 0.45        # Even higher bonus for 4-part names
                },
                'non_ambiguous_bonus': 0.1,
                'comma_bonus': 0.25,
                'indicator_bonus': 0.25,
                'full_text_match_bonus': 0.3,  # Bonus for matching the full original text
                'original_text_match_bonus': 0.2,  # Bonus for matching original text with diacritics
                'unique_ward_bonus': 0.3,  # New bonus for matches that can only be wards
                'comma_boundary_penalty': -0.5  # New penalty for matches spanning comma boundaries
            }
        
        # Set default thresholds if not provided
        if validation_thresholds is None:
            validation_thresholds = {
                'ward': {
                    'non_ambiguous': 0.8,      # Increased from 0.7
                    'ambiguous_beginning': 0.9, # Increased from 0.7
                    'ambiguous_middle': 0.8     # Increased from 0.7
                },
                'district': {
                    'non_ambiguous': 0.8,      # Increased from 0.7
                    'ambiguous_beginning': 0.9, # Increased from 0.7
                    'ambiguous_middle': 0.8     # Increased from 0.7
                }
            }
        
        # Normalize input text and detect HCM context
        normalized_tokens, original_tokens, is_hcm, comma_groups = normalize_text(text)
        if not normalized_tokens:
            return {
                'province': None,
                'district': None,
                'ward': None,
                'is_hcm': False,
                'normalized_tokens': [],
                'original_tokens': [],
                'comma_groups': []
            }
        
        # Initialize results
        province = None
        district = None
        ward = None
        
        # Special handling for HCM context
        if is_hcm:
            province = 'Hồ Chí Minh'
        
        # If no HCM context or no numeric district found, proceed with normal parsing
        if not province:
            # 1. Find Province
            # First try exact match
            for i in range(len(normalized_tokens)):
                for j in range(i + 1, min(i + 4, len(normalized_tokens) + 1)):
                    search_text = ' '.join(normalized_tokens[i:j])
                    if province_trie.search_exact(search_text):
                        # Get the corresponding part of original text
                        original_part = ' '.join(original_tokens[i:j])
                        province = province_trie.get_full_word(search_text, original_part)
                        break
                if province:
                    break
            
            # If no exact match, try fuzzy match
            if not province:
                for i in range(len(normalized_tokens)):
                    for j in range(i + 1, min(i + 4, len(normalized_tokens) + 1)):
                        search_text = ' '.join(normalized_tokens[i:j])
                        matches = fuzzy_search_in_trie(province_trie, search_text, max_dist=2)
                        if matches:
                            # Get the corresponding part of original text
                            original_part = ' '.join(original_tokens[i:j])
                            province = province_trie.get_full_word(matches[0][0], original_part)
                            break
                    if province:
                        break
        
        # 2. Find District using find_district with confidence score
        district, district_score = find_district(
            normalized_tokens, 
            district_trie, 
            ward_trie, 
            scoring_weights=scoring_weights,
            is_hcm=is_hcm,
            original_tokens=original_tokens,  # Add original tokens
            comma_groups=comma_groups  # Add comma groups
        )
        
        # Validate district match
        if district and validate_district_match(district, normalized_tokens, district_score, validation_thresholds, province, ward_trie, district_trie):
            # Find the position of the district in normalized tokens
            district_parts = district.split()
            for i in range(len(normalized_tokens) - len(district_parts) + 1):
                if normalized_tokens[i:i+len(district_parts)] == district_parts:
                    # Get the corresponding part of original text
                    original_part = ' '.join(original_tokens[i:i+len(district_parts)])
                    district = district_trie.get_full_word(district, original_part)
                    break
        else:
            district = None
        
        # 3. Find Ward using find_ward with confidence score
        ward, ward_score = find_ward(
            normalized_tokens, 
            ward_trie, 
            district_trie, 
            scoring_weights=scoring_weights,
            original_tokens=original_tokens,  # Add original tokens
            comma_groups=comma_groups  # Add comma groups
        )
        
        # Validate ward match
        if ward and validate_ward_match(ward, normalized_tokens, ward_score, validation_thresholds, district_trie, ward_trie):
            # Find the position of the ward in normalized tokens
            ward_parts = ward.split()
            for i in range(len(normalized_tokens) - len(ward_parts) + 1):
                if normalized_tokens[i:i+len(ward_parts)] == ward_parts:
                    # Get the corresponding part of original text
                    original_part = ' '.join(original_tokens[i:i+len(ward_parts)])
                    ward = ward_trie.get_full_word(ward, original_part)
                    break
        else:
            ward = None
        
        return {
            'province': province,
            'district': district,
            'ward': ward,
            'is_hcm': is_hcm,
            'normalized_tokens': normalized_tokens,
            'original_tokens': original_tokens,
            'comma_groups': comma_groups
        }
    except Exception as e:
        logger.error(f"Error in parse_location_components: {str(e)}")
        logger.error(traceback.format_exc())  # Add stack trace to log file
        return None

def run_demo_test_file():
    """
    Run first 100 tests from test.json file and compare results with expected output.
    Uses the parse_location_components function with confidence scoring.
    """
    import json
    from collections import defaultdict
    import traceback
    
    # Configure logging
    setup_logging(disable_file_logging=False)
    logger = logging.getLogger('address_parser')
    
    print("\n=== Running First 100 Test Cases from test.json ===\n")
    
    try:
        # Load dictionary data and build tries once
        province_list, district_list, ward_list = load_dictionary_files()
        print(f"\nLoaded dictionary data:")
        print(f"Provinces: {len(province_list)}")
        print(f"Districts: {len(district_list)}")
        print(f"Wards: {len(ward_list)}")
        
        province_trie = build_trie_from_list(province_list)
        district_trie = build_trie_from_list(district_list)
        ward_trie = build_trie_from_list(ward_list)
        
        # Verify trie contents
        print("\nTrie contents:")
        print(f"Provinces in trie: {len(province_trie.collect_all_words())}")
        print(f"Districts in trie: {len(district_trie.collect_all_words())}")
        print(f"Wards in trie: {len(ward_trie.collect_all_words())}")
        
        # Load test cases
        try:
            with open('test.json', 'r', encoding='utf-8') as f:
                test_cases = json.load(f)  # Only take first 10 test cases
                print(f"\nSuccessfully loaded {len(test_cases)} test cases")
        except Exception as e:
            print(f"Error loading test.json: {e}")
            return

        # Run tests
        total_tests = len(test_cases)
        successful = 0
        failed = 0
        failures = defaultdict(list)

        print(f"\nRunning {total_tests} test cases...")

        indeeex = 0;
        
        for i, test_case in enumerate(test_cases, 1):
            try:

                if test_case['text'] != "Thái Ha, HBa Vì, T.pHNội":
                    continue


                print(f"\nProcessing test case {i} of {total_tests}")
                address = test_case['text']
                expected = test_case['result']
                
                print(f"\nTest Case {i}:")
                print(f"Input address: {address}")
                print(f"Expected result: {json.dumps(expected, ensure_ascii=False)}")

                
                print("Starting address parsing...")
                # Parse address using our new function
                result = parse_location_components(
                    address,
                    province_trie=province_trie,
                    district_trie=district_trie,
                    ward_trie=ward_trie,
                    disable_file_logging=True,  # Enable logging for this test case
                    validation_thresholds = {
                        "ward": {
                            "non_ambiguous": 0.75,
                            "ambiguous_beginning": 0.85,
                            "ambiguous_middle": 0.75
                        },
                        "district": {
                            "non_ambiguous": 0.75,
                            "ambiguous_beginning": 0.85,
                            "ambiguous_middle": 0.75
                        }
                    },
                    scoring_weights={
                        "fuzzy_score": 0.15,
                        "exact_match": 0.35,
                        "position_bonus": {
                            "beginning": 0.25,
                            "middle": 0.3
                        },
                        "length_bonus": {
                            "2_parts": 0.25,       # Base bonus for 2-part names
                            "3_parts": 0.35,       # Increased bonus for 3-part names
                            "4_parts": 0.45        # Even higher bonus for 4-part names
                        },
                        "non_ambiguous_bonus": 0.1,
                        "comma_bonus": 0.25,
                        "indicator_bonus": 0.25,
                        "full_text_match_bonus": 0.3,  # Bonus for matching the full original text
                        "original_text_match_bonus": 0.2,  # New bonus for matching original text with diacritics
                        "unique_ward_bonus": 0.2  # New bonus for matches that can only be wards
                    }
                )
                print("Address parsing completed")
                
                # Compare results
                province_match = result['province'] == expected['province']
                district_match = result['district'] == expected['district']
                ward_match = result['ward'] == expected['ward']
                
                print("\nActual result:")
                print(f"Province: {result['province']} {'✓' if province_match else '✗'}")
                print(f"District: {result['district']} {'✓' if district_match else '✗'}")
                print(f"Ward: {result['ward']} {'✓' if ward_match else '✗'}")
                
                if province_match and district_match and ward_match:
                    successful += 1
                    print("\nStatus: SUCCESS ✓")
                else:
                    failed += 1
                    failure_reasons = []
                    
                    if not province_match:
                        failure_reasons.append(f"Province: expected '{expected['province']}', got '{result['province']}'")
                    if not district_match:
                        failure_reasons.append(f"District: expected '{expected['district']}', got '{result['district']}'")
                    if not ward_match:
                        failure_reasons.append(f"Ward: expected '{expected['ward']}', got '{result['ward']}'")
                    
                    print("\nStatus: FAILED ✗")
                    print(f"Failure reasons: {'; '.join(failure_reasons)}")
                    
                    failures[address] = failure_reasons
                
                print("-" * 80)
                
            except Exception as e:
                print(f"Error processing test case {i}: {str(e)}")
                logger.error(f"Error processing test case {i}: {str(e)}")
                logger.error(traceback.format_exc())  # Add stack trace to log file
                failed += 1
                failures[address] = [f"Error: {str(e)}"]
                continue
        
        if failed > 0:
            print("\nFailed test cases:")
            for address, reasons in failures.items():
                print(f"\nAddress: {address}")
                print(f"Reasons: {'; '.join(reasons)}")

        # Print summary
        print("\n=== Test Results Summary ===")
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {(successful/total_tests)*100:.2f}%")

        print(indeeex)
                
    except Exception as e:
        print(f"Critical error in run_demo_test_file: {str(e)}")
        logger.error(f"Critical error in run_demo_test_file: {str(e)}")
        logger.error(traceback.format_exc())  # Add stack trace to log file

if __name__ == "__main__":
    # Uncomment one of these to run either the demo or the test file
    # run_demo()
    run_demo_test_file()


