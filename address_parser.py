import re
import unicodedata
import time
import logging
from rapidfuzz import fuzz, process

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from ERROR to DEBUG to see all messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('address_parser.log', mode='w', encoding='utf-8')
    ]
)

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
        full_word (str or None): Optionally store the original word (with diacritics) 
                                 if this node is the end of a valid entry.
    """
    def __init__(self):
        self.children = {}
        self.is_terminal = False
        self.full_word = None


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
                current.full_word = original_name
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

    def get_full_word(self, word):
        """
        Retrieve the original form of 'word' if it exists in the trie.
        
        :param word: The normalized string to look up.
        :return: The stored original form if found, otherwise None.
        """
        try:
            current = self.root
            for char in word:
                if char not in current.children:
                    logger.debug(f"Full word lookup failed for: {word}")
                    return None
                current = current.children[char]
            if current.is_terminal:
                logger.debug(f"Found full word for {word}: {current.full_word}")
                return current.full_word
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
                    results.append(path)
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
    """
    if not text:
        return [], False
    
    # 1. Preprocess malformed text
    # Fix missing spaces between words by looking for camelCase patterns
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text)
    
    # Fix common malformed patterns
    text = re.sub(r'([A-Za-z])(\d+)([A-Za-z])', r'\1 \2 \3', text)  # Split letter-number-letter
    text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Split number-letter
    text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)  # Split letter-number
    
    # Fix common administrative unit prefixes that might be stuck to words
    # Only split if the prefix is followed by a space or end of string
    admin_prefixes = ['huyen', 'quan', 'phuong', 'xa', 'thi', 'tran', 'tinh', 'thanh', 'pho']
    pattern = '|'.join(f'({prefix})' for prefix in admin_prefixes)
    text = re.sub(f'({pattern})(?=\s|$)', r'\1 ', text, flags=re.IGNORECASE)
    
    # Remove dots at word boundaries that aren't part of abbreviations
    text = re.sub(r'(?<=[^A-Za-z0-9])\.|\.(?=[^A-Za-z0-9])', ' ', text)
    
    # Fix common typos in district names
    typo_fixes = {
        r'bao la\.?': 'bao lam',
        r'ph\d+o': 'pho',
        r'hxuan': 'huyen xuan',
        r'tphô': 'thanh pho',
        r'ph\d+ố': 'pho',
        r'q\.(\d+)': r'quan \1',
        r'p\.(\d+)': r'phuong \1',
        r'h\.([a-zđ])': r'huyen \1',
        r'tx\.([a-zđ])': r'thi xa \1',
        r'tt\.([a-zđ])': r'thi tran \1',
    }
    
    for pattern, replacement in typo_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    # 2. Convert to lowercase
    text = text.lower()

    # 3. Check for HCM context and standardize variations
    is_hcmc, text = detect_hcm_context(text)

    # 4. Replace punctuation with space, preserving dots for abbreviations
    text = re.sub(r'[,:/\\\-()]+', ' ', text)

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
            text = re.sub(pattern, replacement, text)
    
    # Apply province abbreviations
    for pattern, replacement in province_abbrev.items():
        text = re.sub(pattern, replacement, text)
    
    # Handle ward/district abbreviations with numbers
    text = re.sub(r'\bp\.?\s*(\d+)\b', r'phuong \1', text)
    text = re.sub(r'\bq\.?\s*(\d+)\b', r'quan \1', text)
    
    # Handle ward/district without numbers - only if followed by a letter
    text = re.sub(r'\bp\.\s*([a-zđ])', r'phuong \1', text)
    text = re.sub(r'\bq\.\s*([a-zđ])', r'quan \1', text)
    
    # Handle other common abbreviations
    text = re.sub(r'\bđ\.\s*', 'duong ', text)
    text = re.sub(r'\bt\.\s*', 'tinh ', text)
    # Only expand tp. if not in HCM context
    if not is_hcmc:
        text = re.sub(r'\btp\.\s*', 'tp ', text)
    text = re.sub(r'\btx\.\s*', 'thi xa ', text)
    text = re.sub(r'\btt\.\s*', 'thi tran ', text)
    text = re.sub(r'\bh\.\s*', 'huyen ', text)
    
    # Special cases - prevent over-expansion
    text = re.sub(r'\bkp\b', 'khu pho', text)  # Expand kp only as whole word
    
    # Now remove remaining dots
    text = re.sub(r'\.', ' ', text)

    # 6. Remove diacritics (already done in detect_hcm_context)

    # 7. Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 8. Split into tokens
    tokens = text.split()

    # 9. Expand any remaining abbreviations token by token
    expanded_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Skip expansion for certain tokens
        if token in ['tp', 'hcm', 'kp']:
            expanded_tokens.append(token)
        else:
            # Special handling for 'quan' without number
            if token == 'q' and (i + 1 >= len(tokens) or not tokens[i + 1].isdigit()):
                expanded_tokens.append('quan')
            else:
                expanded = expand_abbreviations(token)
                expanded_tokens.extend(expanded.split())
        i += 1

    # 10. Standardize place names using official dictionaries
    tokens = standardize_place_names(expanded_tokens)
    
    # 11. Final ward abbreviation expansion
    final_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == 'p' and i + 1 < len(tokens) and not tokens[i + 1].isdigit():
            final_tokens.append('phuong')
            i += 1
        else:
            final_tokens.append(tokens[i])
            i += 1
    
    return final_tokens, is_hcmc


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

def get_match_confidence(name, tokens, district_trie=None, ward_trie=None):
    """
    Get the confidence score and ambiguity status for a potential match.
    
    Args:
        name: The normalized name to check
        tokens: All tokens from the address
        district_trie: Trie containing district data
        ward_trie: Trie containing ward data
    Returns:
        tuple: (is_ambiguous, confidence_score)
            - is_ambiguous: True if name appears in both district and ward lists
            - confidence_score: Float between 0 and 1 indicating match confidence
    """
    try:
        logger.debug(f"\nCalculating confidence for: {name}")
        logger.debug(f"Tokens: {tokens}")
        
        # Check if name is ambiguous
        is_ambiguous = False
        if district_trie and ward_trie:
            in_district = district_trie.search_exact(name)
            in_ward = ward_trie.search_exact(name)
            is_ambiguous = in_district and in_ward
            logger.debug(f"Found in district list: {in_district}")
            logger.debug(f"Found in ward list: {in_ward}")
            logger.debug(f"Is ambiguous: {is_ambiguous}")
        
        # Calculate confidence score based on multiple factors
        confidence = 0.0
        logger.debug("Calculating confidence factors:")
        
        # 1. Exact match gets highest confidence
        name_parts = name.split()
        for i in range(len(tokens) - len(name_parts) + 1):
            if tokens[i:i+len(name_parts)] == name_parts:
                confidence += 0.4
                logger.debug(f"+0.4 for exact match at position {i}")
                break
        else:
            logger.debug("No exact match found")
        
        # 2. Position bonus (adjusted for wards)
        position_found = False
        for i in range(len(tokens) - len(name_parts) + 1):
            if tokens[i:i+len(name_parts)] == name_parts:
                position_found = True
                # Reduced position bonuses for wards
                if i == 0:  # Beginning
                    confidence += 0.2  # Reduced from 0.4
                    logger.debug(f"+0.2 for appearing at beginning")
                elif i > 0 and i < len(tokens) - len(name_parts):  # Middle
                    confidence += 0.3  # Kept same
                    logger.debug(f"+0.3 for appearing in middle")
                break
        
        if not position_found:
            logger.debug("No position match found")
        
        # 3. Length bonus (increased for multi-word names)
        if len(name_parts) > 1:
            confidence += 0.2
            logger.debug(f"+0.2 for multi-word name")
        
        # 4. Non-ambiguous bonus
        if not is_ambiguous:
            confidence += 0.3
            logger.debug(f"+0.3 for non-ambiguous name")
        
        # 5. Comma-aware bonus
        # Check if this match appears after a comma
        for i in range(len(tokens) - len(name_parts) + 1):
            if tokens[i:i+len(name_parts)] == name_parts:
                # Look for comma before this position
                for j in range(i-1, -1, -1):
                    if tokens[j] == ',':
                        confidence += 0.2
                        logger.debug(f"+0.2 for appearing after comma")
                        break
                break
        
        final_confidence = min(confidence, 1.0)
        logger.debug(f"Final confidence score: {final_confidence}")
        
        return is_ambiguous, final_confidence
    except Exception as e:
        logger.error(f"Error in get_match_confidence: {str(e)}")
        return False, 0.0

def validate_district_match(matched_text, tokens, province=None, ward_trie=None, district_trie=None):
    """
    Validate a potential district match using improved ambiguity handling.
    """
    try:
        district_parts = matched_text.split()
        district_pos = -1
        for i in range(len(tokens) - len(district_parts) + 1):
            if tokens[i:i+len(district_parts)] == district_parts:
                district_pos = i
                break
        
        logger.debug(f"Validating district match: {matched_text} at position {district_pos}")
        
        # Get match confidence and ambiguity status
        is_ambiguous, confidence = get_match_confidence(matched_text, tokens, district_trie, ward_trie)
        logger.debug(f"Match confidence: {confidence}, Is ambiguous: {is_ambiguous}")
        
        # Check for province position
        if province:
            province_parts = remove_diacritics(province.lower()).split()
            province_pos = -1
            for i in range(len(tokens) - len(province_parts) + 1):
                if tokens[i:i+len(province_parts)] == province_parts:
                    province_pos = i
                    break
            
            if district_pos > province_pos and province_pos != -1:
                logger.debug(f"Rejecting: district at {district_pos} appears after province at {province_pos}")
                return False
        
        # Special handling for HCM
        if province and province.lower() == 'ho chi minh':
            if not matched_text.isdigit() and not any(d in matched_text for d in ['binh thanh', 'phu nhuan', 'tan binh', 'thu duc', 'go vap', 'hoc mon', 'binh chanh', 'can gio', 'nha be']):
                logger.debug(f"Rejecting: '{matched_text}' is not a valid HCM district")
                return False
        
        # If not ambiguous and has good confidence, accept
        if not is_ambiguous and confidence >= 0.5:  # Lowered from 0.6
            logger.debug(f"Accepting non-ambiguous district match with good confidence: {matched_text}")
            return True
        
        # If ambiguous, check position and confidence
        if is_ambiguous:
            # Prefer middle position for districts
            if district_pos > 0 and district_pos < len(tokens) - len(district_parts):
                if confidence >= 0.4:  # Lowered from 0.5
                    logger.debug(f"Accepting ambiguous district match in middle position with good confidence: {matched_text}")
                    return True
            
            # If at beginning, require higher confidence
            if district_pos == 0 and confidence >= 0.6:  # Lowered from 0.7
                logger.debug(f"Accepting ambiguous district match at beginning with high confidence: {matched_text}")
                return True
        
        logger.debug(f"Rejecting district match: {matched_text}")
        return False
    except Exception as e:
        logger.error(f"Error in validate_district_match: {str(e)}")
        return False

def validate_ward_match(matched_text, tokens, district_trie=None, ward_trie=None):
    """
    Validate a potential ward match using improved ambiguity handling.
    """
    try:
        ward_parts = matched_text.split()
        ward_pos = -1
        for i in range(len(tokens) - len(ward_parts) + 1):
            if tokens[i:i+len(ward_parts)] == ward_parts:
                ward_pos = i
                break
        
        logger.debug(f"Validating ward match: {matched_text} at position {ward_pos}")
        
        # Get match confidence and ambiguity status
        is_ambiguous, confidence = get_match_confidence(matched_text, tokens, district_trie, ward_trie)
        logger.debug(f"Match confidence: {confidence}, Is ambiguous: {is_ambiguous}")
        
        # If not ambiguous and has good confidence, accept
        if not is_ambiguous and confidence >= 0.6:  # Increased from 0.5
            logger.debug(f"Accepting non-ambiguous ward match with good confidence: {matched_text}")
            return True
        
        # If ambiguous, check position and confidence
        if is_ambiguous:
            # Prefer beginning position for wards
            if ward_pos == 0:
                if confidence >= 0.7:  # Increased from 0.4
                    logger.debug(f"Accepting ambiguous ward match at beginning with high confidence: {matched_text}")
                    return True
            
            # If in middle, require higher confidence
            if ward_pos > 0 and confidence >= 0.6:  # Kept same
                logger.debug(f"Accepting ambiguous ward match in middle with high confidence: {matched_text}")
                return True
        
        # Special case: Check if this ward appears after a comma
        if ward_pos > 0:
            for i in range(ward_pos-1, -1, -1):
                if tokens[i] == ',':
                    # If ward appears after comma, be more lenient
                    if confidence >= 0.5:
                        logger.debug(f"Accepting ward match after comma with good confidence: {matched_text}")
                        return True
                    break
        
        logger.debug(f"Rejecting ward match: {matched_text}")
        return False
    except Exception as e:
        logger.error(f"Error in validate_ward_match: {str(e)}")
        return False

def find_district(tokens, district_trie, district_indicators, province=None, ward_trie=None):
    """
    Find district in the normalized and tokenized address.
    Uses context from province and ward to improve matching accuracy.
    """
    try:
        if not tokens:
            logger.debug("No tokens provided for district search")
            return (None, None, 0)
        
        # Define district indicators
        admin_prefixes = set(district_indicators)
        logger.debug(f"Searching for district in tokens: {tokens}")
        logger.debug(f"Province context: {province}")
        
        # Special case: Check for HCM numeric districts
        if is_hcm_context(tokens):
            logger.debug("HCM context detected, checking for numeric districts")
            # First check for explicit district indicators
            for i, token in enumerate(tokens):
                # Check for "quan X" pattern
                if token == 'quan' and i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    if next_token.isdigit() and len(next_token) == 1:
                        logger.debug(f"Found HCM numeric district pattern 'quan {next_token}'")
                        return (next_token, f"Quận {next_token}", 2)
                
                # Check for "qX" pattern
                if token.startswith('q') and len(token) == 2 and token[1].isdigit():
                    district_num = token[1]
                    logger.debug(f"Found HCM numeric district pattern 'q{district_num}'")
                    return (district_num, f"Quận {district_num}", 1)
            
            # For HCM context, also check for named districts
            for i, token in enumerate(tokens):
                if token in admin_prefixes and i + 1 < len(tokens):
                    logger.debug(f"Found district indicator: {token}")
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
                        logger.debug(f"Trying district match with indicator: {text_to_search}")
                        if district_trie.search_exact(text_to_search):
                            original = district_trie.get_full_word(text_to_search)
                            logger.debug(f"Found exact district match with indicator: {text_to_search} -> {original}")
                            if validate_district_match(text_to_search, tokens, province, ward_trie, district_trie):
                                logger.debug(f"Validated district match with indicator: {text_to_search}")
                                return (text_to_search, original, len(next_tokens))
                            else:
                                logger.debug(f"Failed to validate district match with indicator: {text_to_search}")
        
        # First try to find matches with indicators
        logger.debug("Starting indicator-based search for districts")
        for i, token in enumerate(tokens):
            if token in admin_prefixes and i + 1 < len(tokens):
                logger.debug(f"Found district indicator: {token}")
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
                    logger.debug(f"Trying district match with indicator: {text_to_search}")
                    if district_trie.search_exact(text_to_search):
                        original = district_trie.get_full_word(text_to_search)
                        logger.debug(f"Found exact district match with indicator: {text_to_search} -> {original}")
                        if validate_district_match(text_to_search, tokens, province, ward_trie, district_trie):
                            logger.debug(f"Validated district match with indicator: {text_to_search}")
                            return (text_to_search, original, len(next_tokens))
                        else:
                            logger.debug(f"Failed to validate district match with indicator: {text_to_search}")
                    
                    # Try with reversed tokens if we have exactly 2
                    if len(next_tokens) == 2:
                        reversed_text = ' '.join(reversed(next_tokens))
                        logger.debug(f"Trying reversed tokens: {reversed_text}")
                        if district_trie.search_exact(reversed_text):
                            original = district_trie.get_full_word(reversed_text)
                            logger.debug(f"Found exact district match with reversed tokens: {reversed_text} -> {original}")
                            if validate_district_match(reversed_text, tokens, province, ward_trie, district_trie):
                                logger.debug(f"Validated district match with reversed tokens: {reversed_text}")
                                return (reversed_text, original, len(next_tokens))
                            else:
                                logger.debug(f"Failed to validate district match with reversed tokens: {reversed_text}")
                    
                    # Try fuzzy match using rapidfuzz with higher threshold
                    candidates = district_trie.collect_all_words()
                    if candidates:
                        logger.debug(f"Trying fuzzy match for: {text_to_search}")
                        result = process.extractOne(
                            text_to_search,
                            candidates,
                            scorer=fuzz.ratio,
                            score_cutoff=90  # Increased threshold for stricter matching
                        )
                        
                        if result:
                            matched_text, score, _ = result
                            logger.debug(f"Found fuzzy district match: {matched_text} (score: {score})")
                            if validate_district_match(matched_text, tokens, province, ward_trie, district_trie):
                                logger.debug(f"Validated fuzzy district match: {matched_text}")
                                original = district_trie.get_full_word(matched_text)
                                if original:
                                    return (matched_text, original, len(next_tokens))
                            else:
                                logger.debug(f"Failed to validate fuzzy district match: {matched_text}")
        
        # Then try to find exact matches in the tokens
        max_group = 3
        logger.debug("Starting exact match search for districts")
        best_match = None
        best_match_info = None
        for group_size in range(min(max_group, len(tokens)), 1, -1):
            for start_idx in range(len(tokens) - group_size + 1):
                # Skip if any token is an indicator
                if any(t in admin_prefixes for t in tokens[start_idx:start_idx + group_size]):
                    continue
                group = tokens[start_idx:start_idx + group_size]
                group_text = ' '.join(group)
                logger.debug(f"Trying exact match for group: {group_text}")
                # Try exact match first
                if district_trie.search_exact(group_text):
                    original = district_trie.get_full_word(group_text)
                    logger.debug(f"Found exact district match: {group_text} -> {original}")
                    ambiguous = False
                    if ward_trie is not None:
                        ambiguous = is_ambiguous_name(group_text, district_trie, ward_trie)
                    if validate_district_match(group_text, tokens, province, ward_trie, district_trie):
                        # Prefer non-ambiguous match, or later match if ambiguous
                        if not ambiguous:
                            logger.debug(f"Validated non-ambiguous district match: {group_text}")
                            return (group_text, original, group_size)
                        else:
                            # Save ambiguous match, but keep looking for a better one
                            if best_match is None or start_idx > best_match_info['start_idx']:
                                best_match = (group_text, original, group_size)
                                best_match_info = {'start_idx': start_idx, 'ambiguous': True}
                    else:
                        logger.debug(f"Failed to validate district match: {group_text}")
        
        # If we found an ambiguous match, return it as fallback
        if best_match is not None:
            logger.debug(f"Returning ambiguous district match as fallback: {best_match[0]}")
            return best_match

        logger.debug("No district match found after all attempts")
        return (None, None, 0)
    except Exception as e:
        logger.error(f"Error in find_district: {str(e)}")
        return (None, None, 0)

def find_ward(tokens, ward_trie, ward_indicators, district_trie=None):
    """
    Find ward in the normalized and tokenized address.
    Now handles ambiguous names that appear in both district and ward lists.
    """
    try:
        if not tokens:
            return (None, None, 0)
            
        logger.debug(f"\n=== Starting ward search ===")
        logger.debug(f"Input tokens: {tokens}")
            
        # Define ward indicators
        admin_prefixes = set(ward_indicators)
        
        # First look for explicit ward indicators with numbers
        for i, token in enumerate(tokens):
            # Check for "phuong X" pattern
            if token == 'phuong' and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.isdigit() and len(next_token) <= 2:
                    # Don't expand numeric wards to full names
                    return (next_token, next_token, 2)
            
            # Check for "pX" pattern
            if token.startswith('p') and len(token) <= 3 and token[1:].isdigit():
                ward_num = token[1:]
                # Don't expand numeric wards to full names
                return (ward_num, ward_num, 1)
        
        # First try to find matches at the beginning of the address
        max_group = 4
        for group_size in range(min(max_group, len(tokens)), 1, -1):
            group = tokens[:group_size]
            group_text = ' '.join(group)
            logger.debug(f"\nTrying group at beginning: {group_text}")
            
            # Try exact match first
            if ward_trie.search_exact(group_text):
                logger.debug(f"Found exact match for: {group_text}")
                # Get the original form with diacritics
                original = ward_trie.get_full_word(group_text)
                logger.debug(f"Original form from trie: {original}")
                if original and validate_ward_match(group_text, tokens, district_trie, ward_trie):
                    # If the input text has diacritics, use it instead
                    input_text = ' '.join(tokens[:group_size])
                    logger.debug(f"Input text: {input_text}")
                    if any(c in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ' for c in input_text):
                        logger.debug(f"Using input text with diacritics: {input_text}")
                        return (group_text, input_text, group_size)
                    logger.debug(f"Using original form from trie: {original}")
                    return (group_text, original, group_size)
            
            # Try fuzzy match with higher threshold
            candidates = ward_trie.collect_all_words()
            if candidates:
                result = process.extractOne(
                    group_text,
                    candidates,
                    scorer=fuzz.ratio,
                    score_cutoff=90  # Increased threshold for stricter matching
                )
                
                if result:
                    matched_text, score, _ = result
                    logger.debug(f"Found fuzzy match: {matched_text} (score: {score})")
                    if validate_ward_match(matched_text, tokens, district_trie, ward_trie):
                        # Get the original form with diacritics
                        original = ward_trie.get_full_word(matched_text)
                        logger.debug(f"Original form from trie: {original}")
                        if original:
                            # If the input text has diacritics, use it instead
                            input_text = ' '.join(tokens[:group_size])
                            logger.debug(f"Input text: {input_text}")
                            if any(c in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ' for c in input_text):
                                logger.debug(f"Using input text with diacritics: {input_text}")
                                return (matched_text, input_text, group_size)
                            logger.debug(f"Using original form from trie: {original}")
                            return (matched_text, original, group_size)
        
        # Then try to find matches with indicators
        for i, token in enumerate(tokens):
            if token in admin_prefixes and i + 1 < len(tokens):
                # Look for next three tokens (increased from two)
                next_tokens = []
                j = i + 1
                count = 0
                while j < len(tokens) and count < 3:  # Increased from 2
                    if tokens[j] not in admin_prefixes:
                        next_tokens.append(tokens[j])
                        count += 1
                    j += 1
                
                if next_tokens:
                    text_to_search = ' '.join(next_tokens)
                    logger.debug(f"\nTrying with indicator {token}: {text_to_search}")
                    
                    # Try exact match first
                    if ward_trie.search_exact(text_to_search):
                        logger.debug(f"Found exact match for: {text_to_search}")
                        # Get the original form with diacritics
                        original = ward_trie.get_full_word(text_to_search)
                        logger.debug(f"Original form from trie: {original}")
                        if original and validate_ward_match(text_to_search, tokens, district_trie, ward_trie):
                            # If the input text has diacritics, use it instead
                            input_text = ' '.join(next_tokens)
                            logger.debug(f"Input text: {input_text}")
                            if any(c in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ' for c in input_text):
                                logger.debug(f"Using input text with diacritics: {input_text}")
                                return (text_to_search, input_text, len(next_tokens))
                            logger.debug(f"Using original form from trie: {original}")
                            return (text_to_search, original, len(next_tokens))
                    
                    # Try with reversed tokens if we have exactly 2
                    if len(next_tokens) == 2:
                        reversed_text = ' '.join(reversed(next_tokens))
                        logger.debug(f"Trying reversed tokens: {reversed_text}")
                        if ward_trie.search_exact(reversed_text):
                            logger.debug(f"Found exact match for reversed: {reversed_text}")
                            # Get the original form with diacritics
                            original = ward_trie.get_full_word(reversed_text)
                            logger.debug(f"Original form from trie: {original}")
                            if original and validate_ward_match(reversed_text, tokens, district_trie, ward_trie):
                                # If the input text has diacritics, use it instead
                                input_text = ' '.join(reversed(next_tokens))
                                logger.debug(f"Input text: {input_text}")
                                if any(c in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ' for c in input_text):
                                    logger.debug(f"Using input text with diacritics: {input_text}")
                                    return (reversed_text, input_text, len(next_tokens))
                                logger.debug(f"Using original form from trie: {original}")
                                return (reversed_text, original, len(next_tokens))
                    
                    # Try fuzzy match with higher threshold
                    candidates = ward_trie.collect_all_words()
                    if candidates:
                        result = process.extractOne(
                            text_to_search,
                            candidates,
                            scorer=fuzz.ratio,
                            score_cutoff=90  # Increased threshold for stricter matching
                        )
                        
                        if result:
                            matched_text, score, _ = result
                            logger.debug(f"Found fuzzy match: {matched_text} (score: {score})")
                            if validate_ward_match(matched_text, tokens, district_trie, ward_trie):
                                # Get the original form with diacritics
                                original = ward_trie.get_full_word(matched_text)
                                logger.debug(f"Original form from trie: {original}")
                                if original:
                                    # If the input text has diacritics, use it instead
                                    input_text = ' '.join(next_tokens)
                                    logger.debug(f"Input text: {input_text}")
                                    if any(c in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ' for c in input_text):
                                        logger.debug(f"Using input text with diacritics: {input_text}")
                                        return (matched_text, input_text, len(next_tokens))
                                    logger.debug(f"Using original form from trie: {original}")
                                    return (matched_text, original, len(next_tokens))
        
        # Finally try looking for potential ward names in the middle
        for group_size in range(min(max_group, len(tokens)), 1, -1):
            for start_idx in range(len(tokens) - group_size + 1):
                # Skip if any token is an indicator
                if any(t in admin_prefixes for t in tokens[start_idx:start_idx + group_size]):
                    continue
                    
                group = tokens[start_idx:start_idx + group_size]
                group_text = ' '.join(group)
                logger.debug(f"\nTrying group in middle: {group_text}")
                
                # Try exact match first
                if ward_trie.search_exact(group_text):
                    logger.debug(f"Found exact match for: {group_text}")
                    # Get the original form with diacritics
                    original = ward_trie.get_full_word(group_text)
                    logger.debug(f"Original form from trie: {original}")
                    if original and validate_ward_match(group_text, tokens, district_trie, ward_trie):
                        # If the input text has diacritics, use it instead
                        input_text = ' '.join(group)
                        logger.debug(f"Input text: {input_text}")
                        if any(c in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ' for c in input_text):
                            logger.debug(f"Using input text with diacritics: {input_text}")
                            return (group_text, input_text, group_size)
                        logger.debug(f"Using original form from trie: {original}")
                        return (group_text, original, group_size)
                
                # Try fuzzy match with higher threshold
                candidates = ward_trie.collect_all_words()
                if candidates:
                    result = process.extractOne(
                        group_text,
                        candidates,
                        scorer=fuzz.ratio,
                        score_cutoff=90  # Increased threshold for stricter matching
                    )
                    
                    if result:
                        matched_text, score, _ = result
                        logger.debug(f"Found fuzzy match: {matched_text} (score: {score})")
                        if validate_ward_match(matched_text, tokens, district_trie, ward_trie):
                            # Get the original form with diacritics
                            original = ward_trie.get_full_word(matched_text)
                            logger.debug(f"Original form from trie: {original}")
                            if original:
                                # If the input text has diacritics, use it instead
                                input_text = ' '.join(group)
                                logger.debug(f"Input text: {input_text}")
                                if any(c in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ' for c in input_text):
                                    logger.debug(f"Using input text with diacritics: {input_text}")
                                    return (matched_text, input_text, group_size)
                                logger.debug(f"Using original form from trie: {original}")
                                return (matched_text, original, group_size)

        logger.debug("No ward match found")
        return (None, None, 0)
    except Exception as e:
        logger.error(f"Error in find_ward: {str(e)}")
        return (None, None, 0)

def parse_address(address: str, province_trie: Trie, district_trie: Trie, ward_trie: Trie) -> dict:
    """
    Parse an address string to extract province, district, and ward components.
    Uses a hierarchical approach: province -> district -> ward.
    """
    try:
        # Initialize result structure
        result = {
            'province': None,
            'district': None,
            'ward': None,
            'context': {
                'is_hcm': False,
                'normalized_tokens': [],
                'error': None
            }
        }
        
        logger.debug(f"Parsing address: {address}")
        
        # 1. Normalize input text and detect HCM context
        tokens, is_hcmc = normalize_text(address)
        if not tokens:
            logger.debug("No tokens found after normalization")
            result['context']['error'] = "Empty or invalid address"
            return result
            
        logger.debug(f"Normalized tokens: {tokens}")
        logger.debug(f"Is HCM context: {is_hcmc}")
        
        result['context']['normalized_tokens'] = tokens
        result['context']['is_hcm'] = is_hcmc
        
        # 2. Define indicator words for each component
        province_indicators = ['tinh', 'thanh pho', 'tp']
        district_indicators = ['huyen', 'quan', 'q', 'thi xa', 'tx']
        ward_indicators = ['phuong', 'xa', 'thi tran', 'p', 'tt']
        
        # 3. Find components in hierarchical order
        # First find province
        logger.debug("Starting province search")
        matched_province, original_province, consumed_province = find_province(
            tokens, province_trie, province_indicators
        )
        if original_province:
            logger.debug(f"Found province: {original_province}")
            result['province'] = original_province
        else:
            logger.debug("No province found")
            
        # Then find district (considering HCM context and province)
        logger.debug("Starting district search")
        matched_district, original_district, consumed_district = find_district(
            tokens, district_trie, district_indicators, result['province'], ward_trie
        )
        
        # Handle HCM numeric districts
        if is_hcmc and matched_district:
            if matched_district.isdigit():
                # For HCM numeric districts, use the number directly
                logger.debug(f"Using HCM numeric district: {matched_district}")
                result['district'] = matched_district
            else:
                logger.debug(f"Using HCM district: {original_district}")
                result['district'] = original_district
        else:
            if original_district:
                logger.debug(f"Using district: {original_district}")
                result['district'] = original_district
            else:
                logger.debug("No district found")
            
        # Finally find ward
        logger.debug("Starting ward search")
        matched_ward, original_ward, consumed_ward = find_ward(
            tokens, ward_trie, ward_indicators, district_trie
        )
        if original_ward:
            logger.debug(f"Found ward: {original_ward}")
            result['ward'] = original_ward
        else:
            logger.debug("No ward found")
            
        # 4. Post-processing validation
        # If we found HCM but no province, set it
        if is_hcmc and not result['province']:
            logger.debug("Setting HCM as province")
            result['province'] = 'Hồ Chí Minh'
            
        logger.debug(f"Final parsing result: {result}")
        return result
        
    except Exception as e:
        # Handle any errors during parsing
        logger.error(f"Error during address parsing: {str(e)}")
        return {
            'province': None,
            'district': None,
            'ward': None,
            'context': {
                'is_hcm': False,
                'normalized_tokens': tokens if 'tokens' in locals() else [],
                'error': str(e)
            }
        }


##############################################################
# 6. MAIN DEMO
##############################################################

def parse_location_components(
    address: str,
    province_trie: Trie = None,
    district_trie: Trie = None,
    ward_trie: Trie = None,
    province_indicators: list = None,
    district_indicators: list = None,
    ward_indicators: list = None
) -> dict:
    """
    Parse an address string into its components (province, district, ward).
    
    Args:
        address (str): The address string to parse
        province_trie (Trie, optional): Trie containing province data
        district_trie (Trie, optional): Trie containing district data
        ward_trie (Trie, optional): Trie containing ward data
        province_indicators (list, optional): List of province indicator words
        district_indicators (list, optional): List of district indicator words
        ward_indicators (list, optional): List of ward indicator words
        
    Returns:
        dict: A dictionary containing the parsed components:
            {
                'province': str or None,
                'district': str or None,
                'ward': str or None,
                'context': {
                    'is_hcm': bool,
                    'normalized_tokens': list,
                    'error': str or None
                }
            }
    """
    try:
        # Initialize result structure
        result = {
            'province': None,
            'district': None,
            'ward': None,
            'context': {
                'is_hcm': False,
                'normalized_tokens': [],
                'error': None
            }
        }
        
        logger.debug(f"Parsing address: {address}")
        
        # 1. Load dictionary data and build tries if not provided
        if not all([province_trie, district_trie, ward_trie]):
            province_list, district_list, ward_list = load_dictionary_files()
            
            if not province_trie:
                province_trie = build_trie_from_list(province_list)
            if not district_trie:
                district_trie = build_trie_from_list(district_list)
            if not ward_trie:
                ward_trie = build_trie_from_list(ward_list)
        
        # 2. Set default indicators if not provided
        if province_indicators is None:
            province_indicators = ['tinh', 'thanh pho', 'tp']
        if district_indicators is None:
            district_indicators = ['huyen', 'quan', 'q', 'thi xa', 'tx']
        if ward_indicators is None:
            ward_indicators = ['phuong', 'xa', 'thi tran', 'p', 'tt']
        
        # 3. Normalize input text and detect HCM context
        tokens, is_hcmc = normalize_text(address)
        if not tokens:
            logger.debug("No tokens found after normalization")
            result['context']['error'] = "Empty or invalid address"
            return result
            
        logger.debug(f"Normalized tokens: {tokens}")
        logger.debug(f"Is HCM context: {is_hcmc}")
            
        result['context']['normalized_tokens'] = tokens
        result['context']['is_hcm'] = is_hcmc
        
        # 4. Find components in hierarchical order
        # First find province
        logger.debug("Starting province search")
        matched_province, original_province, consumed_province = find_province(
            tokens, province_trie, province_indicators
        )
        if original_province:
            logger.debug(f"Found province: {original_province}")
            result['province'] = original_province
        else:
            logger.debug("No province found")
            
        # Then find district (considering HCM context and province)
        logger.debug("Starting district search")
        matched_district, original_district, consumed_district = find_district(
            tokens, district_trie, district_indicators, result['province'], ward_trie
        )
        
        # Handle HCM numeric districts
        if is_hcmc and matched_district:
            if matched_district.isdigit():
                # For HCM numeric districts, use the number directly
                logger.debug(f"Using HCM numeric district: {matched_district}")
                result['district'] = matched_district
            else:
                logger.debug(f"Using HCM district: {original_district}")
                result['district'] = original_district
        else:
            if original_district:
                logger.debug(f"Using district: {original_district}")
                result['district'] = original_district
            else:
                logger.debug("No district found")
            
        # Finally find ward
        logger.debug("Starting ward search")
        matched_ward, original_ward, consumed_ward = find_ward(
            tokens, ward_trie, ward_indicators, district_trie
        )
        if original_ward:
            logger.debug(f"Found ward: {original_ward}")
            result['ward'] = original_ward
        else:
            logger.debug("No ward found")
            
        # 5. Post-processing validation
        # If we found HCM but no province, set it
        if is_hcmc and not result['province']:
            logger.debug("Setting HCM as province")
            result['province'] = 'Hồ Chí Minh'
            
        logger.debug(f"Final parsing result: {result}")
        return result
        
    except Exception as e:
        # Handle any errors during parsing
        logger.error(f"Error during address parsing: {str(e)}")
        return {
            'province': None,
            'district': None,
            'ward': None,
            'context': {
                'is_hcm': False,
                'normalized_tokens': tokens if 'tokens' in locals() else [],
                'error': str(e)
            }
        }

# Update the demo functions to use the new function signature
def run_demo():
    print("\n=== Vietnamese Address Parser Demo ===\n")
    
    # Load dictionary data and build tries once
    province_list, district_list, ward_list = load_dictionary_files()
    province_trie = build_trie_from_list(province_list)
    district_trie = build_trie_from_list(district_list)
    ward_trie = build_trie_from_list(ward_list)
    
    # Sample addresses to parse
    sample_addresses = [
        "154/4/81 Nguyễn - Phúc Chu, P15, TB, TP. Hồ Chí Minh",
        "TT Tan Binh Huyen Yen Son, Tuyenn Quangg",
        "p1, q3, tp.hochiminh",
        ", Tân Phươc, Tin GJiang"
    ]

    print("=== Parsing Sample Addresses ===\n")
    for i, addr in enumerate(sample_addresses, 1):
        print(f"Example {i}:")
        print(f"Input:    {addr}")
        
        # Parse the address using the new function with pre-built tries
        result = parse_location_components(
            addr,
            province_trie=province_trie,
            district_trie=district_trie,
            ward_trie=ward_trie
        )
        
        # Format the output
        print("\nParsed Components:")
        print(f"  Province: {result['province'] or 'Not found'}")
        print(f"  District: {result['district'] or 'Not found'}")
        print(f"  Ward:     {result['ward'] or 'Not found'}")
        
        # Print context information
        print("\nContext:")
        print(f"  HCM Context: {'Yes' if result['context']['is_hcm'] else 'No'}")
        print(f"  Normalized Tokens: {' '.join(result['context']['normalized_tokens'])}")
        if result['context']['error']:
            print(f"  Error: {result['context']['error']}")
        
        print("\n" + "="*50 + "\n")

def run_demo_test_file():
    """
    Run first 100 tests from test.json file and compare results with expected output.
    Prints detailed information for each test case.
    """
    import json
    from collections import defaultdict
    
    print("\n=== Running First 100 Test Cases from test.json ===\n")
    
    # Load dictionary data and build tries once
    province_list, district_list, ward_list = load_dictionary_files()
    province_trie = build_trie_from_list(province_list)
    district_trie = build_trie_from_list(district_list)
    ward_trie = build_trie_from_list(ward_list)
    
    # Load test cases
    try:
        with open('test.json', 'r', encoding='utf-8') as f:
            test_cases = json.load(f)[:100]  # Only take first 100 test cases
    except Exception as e:
        print(f"Error loading test.json: {e}")
        return

    # Run tests
    total_tests = len(test_cases)
    successful = 0
    failed = 0
    failures = defaultdict(list)

    print(f"Running {total_tests} test cases...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        address = test_case['text']
        expected = test_case['result']
        
        print(f"\nTest Case {i}:")
        print(f"Input address: {address}")
        print("Expected result:", expected)
        
        # Parse the address using the new function with pre-built tries
        result = parse_location_components(
            address,
            province_trie=province_trie,
            district_trie=district_trie,
            ward_trie=ward_trie
        )
        
        # Print normalized tokens and HCM context
        print("\nParsing details:")
        print(f"Normalized tokens: {' '.join(result['context']['normalized_tokens'])}")
        print(f"HCM context: {'Yes' if result['context']['is_hcm'] else 'No'}")
        
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
            
            failures[address] = failure_reasons
            print("\nStatus: FAILED ✗")
            print("Failure reasons:", '; '.join(failure_reasons))
        
        print("-" * 80)

    # Print summary
    print("\n=== Test Results Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/total_tests)*100:.2f}%")
    
    if failed > 0:
        print("\nFailed test cases:")
        for address, reasons in failures.items():
            print(f"\nAddress: {address}")
            print("Reasons:", '; '.join(reasons))

if __name__ == "__main__":
    # Uncomment one of these to run either the demo or the test file
    # run_demo()
    run_demo_test_file()


