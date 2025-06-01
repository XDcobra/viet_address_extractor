from rapidfuzz import fuzz, process
from vietnamese_address_classification.src.utils.normalization import remove_diacritics

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
        print(f"Error in get_detailed_confidence_score: {str(e)}")
        return {
            'total_score': 0.0,
            'components': [],
            'is_ambiguous': False,
            'error': str(e)
        }



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
        print(f"Error in get_hcm_numeric_district_score: {str(e)}")
        return 0.0