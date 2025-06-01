import re
import unicodedata
from vietnamese_address_classification.src.utils.abbreviation_expander import expand_abbreviations, build_abbreviation_map
from vietnamese_address_classification.src.data.data_loader import load_dictionary_files

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