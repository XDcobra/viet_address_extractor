import re
import unicodedata
import time
from rapidfuzz import fuzz, process

from vietnamese_address_classification.src.core.trie import build_trie_from_list
from vietnamese_address_classification.src.core.validator import validate_province_match, validate_district_match, validate_ward_match
from vietnamese_address_classification.src.utils.normalization import normalize_text
from vietnamese_address_classification.src.utils.ambiguousness import resolve_ambiguous_candidates
from vietnamese_address_classification.src.utils.fuzzy import fuzzy_search_in_trie
from vietnamese_address_classification.src.core.scoring import get_detailed_confidence_score, get_hcm_numeric_district_score
from vietnamese_address_classification.src.utils.normalization import is_hcm_context
from vietnamese_address_classification.src.data.data_loader import load_dictionary_files

##############################################################
# 1. TRIE STRUCTURE & HELPER METHODS
##############################################################




##############################################################
# 2. ADVANCED FUZZY SEARCH (USING EDIT DISTANCE)
##############################################################




##############################################################
# 3. TEXT NORMALIZATION & ABBREVIATION EXPANSIONS
##############################################################




##############################################################
# 4. LOADING DICTIONARIES & BUILDING TRIES
##############################################################




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
        print(f"Error in find_province: {str(e)}")
        return (None, None, 0)



def find_district(tokens, district_trie, ward_trie=None, scoring_weights=None, is_hcm=False, original_tokens=None, comma_groups=None):
    if not tokens:
        return [], []
        
    # First check for Q+number format (e.g., Q3, Q.3)
    for i, token in enumerate(tokens):
        if token.lower() in ['q', 'quan'] and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if next_token.isdigit() and 1 <= int(next_token) <= 24:
                return [(next_token, 2.0)]  # Return as single match with max confidence
    
    # Collect all exact matches with their scores
    exact_matches = []
    
    # Then try exact match
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens) + 1)):
            search_text = ' '.join(tokens[i:j])
            
            # Check if this is a numeric district in HCM
            if is_hcm and search_text.isdigit():
                numeric_score = get_hcm_numeric_district_score(search_text, tokens, i)
                if numeric_score > 0:
                    exact_matches.append((search_text, numeric_score))
                    continue  # Skip the get_detailed_confidence_score calculation
                
            # Try exact match
            if district_trie.search_exact(search_text):
                # Get confidence score
                score_details = get_detailed_confidence_score(
                    search_text, 
                    tokens, 
                    district_trie, 
                    ward_trie,
                    scoring_weights=scoring_weights,
                    component_type='district',
                    original_tokens=original_tokens,
                    comma_groups=comma_groups
                )
                confidence = score_details['total_score']
                exact_matches.append((search_text, confidence))
    
    # If we have exact matches, sort them by confidence
    if exact_matches:
        exact_matches.sort(key=lambda x: x[1], reverse=True)
        # Return top 3 exact matches
        return exact_matches[:3]
    
    # If no exact match, try fuzzy match
    fuzzy_matches = []
    
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens) + 1)):
            search_text = ' '.join(tokens[i:j])
            
            # Get fuzzy matches using fuzzy_search_in_trie
            matches = fuzzy_search_in_trie(district_trie, search_text, max_dist=2)
            if matches:
                # Process each match
                for match, fuzzy_score in matches:
                    # Get confidence score
                    score_details = get_detailed_confidence_score(
                        match, 
                        tokens, 
                        district_trie, 
                        ward_trie,
                        fuzzy_score=fuzzy_score,
                        scoring_weights=scoring_weights,
                        component_type='district',
                        original_tokens=original_tokens,
                        comma_groups=comma_groups
                    )
                    
                    fuzzy_matches.append((match, score_details['total_score']))
    
    # Sort fuzzy matches by confidence and return top 3
    fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
    return fuzzy_matches[:3] if fuzzy_matches else []

def find_ward(tokens, ward_trie, district_trie=None, scoring_weights=None, original_tokens=None, comma_groups=None):
    if not tokens:
        return [], []
    
    # First check for P+number format (e.g., P1, P.1)
    for i, token in enumerate(tokens):
        if token.lower() in ['p', 'phuong'] and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if next_token.isdigit() and 1 <= int(next_token) <= 20:  # HCM wards typically go up to 20
                return [(next_token, 2.0)]  # Return as single match with max confidence
    
    # Collect all exact matches with their scores
    exact_matches = []
    
    # Then try exact match
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens) + 1)):
            search_text = ' '.join(tokens[i:j])
            
            # Try exact match
            if ward_trie.search_exact(search_text):
                # Get confidence score
                score_details = get_detailed_confidence_score(
                    search_text, 
                    tokens, 
                    district_trie, 
                    ward_trie,
                    scoring_weights=scoring_weights,
                    component_type='ward',
                    original_tokens=original_tokens,
                    comma_groups=comma_groups
                )
                confidence = score_details['total_score']
                exact_matches.append((search_text, confidence))
    
    # If we have exact matches, sort them by confidence
    if exact_matches:
        exact_matches.sort(key=lambda x: x[1], reverse=True)
        # Return top 3 exact matches
        return exact_matches[:3]
    
    # If no exact match, try fuzzy match
    fuzzy_matches = []
    
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens) + 1)):
            search_text = ' '.join(tokens[i:j])
            
            # Get fuzzy matches
            matches = fuzzy_search_in_trie(ward_trie, search_text, max_dist=2)
            if matches:
                # Process each match
                for match, fuzzy_score in matches:
                    # Get confidence score
                    score_details = get_detailed_confidence_score(
                        match, 
                        tokens, 
                        district_trie, 
                        ward_trie,
                        fuzzy_score=fuzzy_score,
                        scoring_weights=scoring_weights,
                        component_type='ward',
                        original_tokens=original_tokens,
                        comma_groups=comma_groups
                    )
                    
                    fuzzy_matches.append((match, score_details['total_score']))
    
    # Sort fuzzy matches by confidence and return top 3
    fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
    return fuzzy_matches[:3] if fuzzy_matches else []

##############################################################
# 6. MAIN DEMO
##############################################################



def parse_location_components(text, province_list=None, district_list=None, ward_list=None, scoring_weights=None, validation_thresholds=None, disable_file_logging=False):
    """
    Parse location components from text using all available components:
    - Confidence Score System
    - Fuzzy Matching
    - Standardized Weights
    - Threshold Validation
    - Ambiguity Checking
    
    Args:
        text: The input text to parse
        province_list: List containing province data
        district_trie: List containing district data
        ward_trie: List containing ward data
        scoring_weights: Optional dictionary of scoring weights
        validation_thresholds: Optional dictionary of validation thresholds
        disable_file_logging: If True, disable logging to file
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

        # Set Default Tries if not provided
        province_list_loaded, district_list_loaded, ward_list_loaded = load_dictionary_files()
        if province_list is None:
            province_list = province_list_loaded
        if district_list is None:
            district_list = district_list_loaded
        if ward_list is None:
            ward_list = ward_list_loaded

        # Load tries
        province_trie = build_trie_from_list(province_list)
        district_trie = build_trie_from_list(district_list)
        ward_trie = build_trie_from_list(ward_list)
        
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
                'comma_groups': [],
                'district_candidates': [],
                'ward_candidates': []
            }
        
        # Initialize results
        province = None
        district = None
        ward = None
        district_candidates = []
        ward_candidates = []
        
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
        district_matches = find_district(
            normalized_tokens, 
            district_trie, 
            ward_trie, 
            scoring_weights=scoring_weights,
            is_hcm=is_hcm,
            original_tokens=original_tokens,
            comma_groups=comma_groups
        )
        
        # Store all district candidates
        district_candidates = []
        for match, score in district_matches:
            if validate_district_match(match, normalized_tokens, score, validation_thresholds, province, ward_trie, district_trie):
                # Find the position of the district in normalized tokens
                district_parts = match.split()
                for i in range(len(normalized_tokens) - len(district_parts) + 1):
                    if normalized_tokens[i:i+len(district_parts)] == district_parts:
                        # Get the corresponding part of original text
                        original_part = ' '.join(original_tokens[i:i+len(district_parts)])
                        original_district = district_trie.get_full_word(match, original_part)
                        if original_district:
                            district_candidates.append((original_district, score))
                        break
        
        # 3. Find Ward using find_ward with confidence score
        ward_matches = find_ward(
            normalized_tokens, 
            ward_trie, 
            district_trie, 
            scoring_weights=scoring_weights,
            original_tokens=original_tokens,
            comma_groups=comma_groups
        )
        
        # Store all ward candidates
        ward_candidates = []
        for match, score in ward_matches:
            if validate_ward_match(match, normalized_tokens, score, validation_thresholds, district_trie, ward_trie):
                # Find the position of the ward in normalized tokens
                ward_parts = match.split()
                for i in range(len(normalized_tokens) - len(ward_parts) + 1):
                    if normalized_tokens[i:i+len(ward_parts)] == ward_parts:
                        # Get the corresponding part of original text
                        original_part = ' '.join(original_tokens[i:i+len(ward_parts)])
                        original_ward = ward_trie.get_full_word(match, original_part)
                        if original_ward:
                            ward_candidates.append((original_ward, score))
                        break
        
        # After finding district and ward candidates, try to resolve ambiguity
        if is_hcm:
            # For HCM, just use the best matches since district and ward names are unique
            district = district_candidates[0][0] if district_candidates else None
            ward = ward_candidates[0][0] if ward_candidates else None
            was_resolved = True
        else:
            # For non-HCM addresses, try to resolve ambiguity
            resolved_district, resolved_ward, was_resolved = resolve_ambiguous_candidates(
                district_candidates,
                ward_candidates,
                district_trie,
                ward_trie,
                province,  # Pass the province to the resolution function
                text  # Pass the original input text
            )
            
            # Use resolved values if available
            if was_resolved:
                district = resolved_district
                ward = resolved_ward
            else:
                # Use best matches if no resolution was possible
                if district_candidates:
                    district = district_candidates[0][0]
                if ward_candidates:
                    ward = ward_candidates[0][0]

        return {
            'province': province,
            'district': district,
            'ward': ward,
            'is_hcm': is_hcm,
            'normalized_tokens': normalized_tokens,
            'original_tokens': original_tokens,
            'comma_groups': comma_groups,
            'district_candidates': district_candidates,
            'ward_candidates': ward_candidates,
            'was_resolved': was_resolved
        }
    except Exception as e:
        print(f"Error in parse_location_components: {str(e)}")
        return None


