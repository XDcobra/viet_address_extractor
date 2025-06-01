from vietnamese_address_classification.src.utils.normalization import remove_diacritics

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
                return False
        
        # Special case: Check for province position
        if province:
            province_parts = remove_diacritics(province.lower()).split()
            for i in range(len(tokens) - len(province_parts) + 1):
                if tokens[i:i+len(province_parts)] == province_parts:
                    province_pos = i
                    district_pos = tokens.index(matched_text.split()[0])
                    if district_pos > province_pos:
                        return False
                    break
        
        # Use the passed score and compare with threshold
        return score >= validation_thresholds['district']['non_ambiguous']
        
    except Exception as e:
        print(f"Error in validate_district_match: {str(e)}")
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
        print(f"Error in validate_ward_match: {str(e)}")
        return False