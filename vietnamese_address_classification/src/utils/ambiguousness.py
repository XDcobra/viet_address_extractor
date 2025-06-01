from vietnamese_address_classification.src.utils.normalization import remove_diacritics

def resolve_ambiguous_candidates(district_candidates, ward_candidates, district_trie, ward_trie, province=None, original_text=None):
    try:
        if not district_candidates or not ward_candidates:
            return None, None, False
            
        # First, check if we have any unique candidates
        unique_districts = []
        unique_wards = []
        ambiguous_names = set()
        
        # Normalize province name for comparison if provided
        norm_province = remove_diacritics(province.lower()) if province else None
        
        # Get normalized district candidates for comparison
        norm_district_candidates = [remove_diacritics(d.lower()) for d, _ in district_candidates]
        
        # Check district candidates
        for district, score in district_candidates:
            norm_district = remove_diacritics(district.lower())
            
            # Check if this district exists in ward list
            if ward_trie.search_exact(norm_district):
                # If we have province context, check if the district is unique within that province
                if norm_province:
                    # Get all districts in this province
                    province_districts = []
                    for d in district_trie.collect_all_words():
                        if norm_province in remove_diacritics(d.lower()):
                            province_districts.append(d)
                    
                    # Check if this district is unique within the province
                    if len([d for d in province_districts if remove_diacritics(d.lower()) == norm_district]) == 1:
                        unique_districts.append((district, score))
                        continue
                
                ambiguous_names.add(norm_district)
            else:
                unique_districts.append((district, score))
                
        # Check ward candidates with context validation
        filtered_ward_candidates = []
        for ward, score in ward_candidates:
            norm_ward = remove_diacritics(ward.lower())
            ward_parts = norm_ward.split()
            
            # Check if any part of the ward name appears in district candidates
            is_part_of_district = False
            for norm_district in norm_district_candidates:
                # Check each ward part against the district name
                for part in ward_parts:
                    if part in norm_district:
                        # Find positions in original text
                        norm_original_text = remove_diacritics(original_text.lower())
                        ward_start = norm_original_text.find(norm_ward)
                        district_start = norm_original_text.find(norm_district)
                        
                        if ward_start != -1 and district_start != -1:
                            # Get the text between ward and district
                            if ward_start < district_start:
                                text_between = original_text[ward_start + len(ward):district_start]
                            else:
                                text_between = original_text[district_start + len(norm_district):ward_start]
                            
                            # Check if there's a comma between them
                            has_comma_between = ',' in text_between
                            
                            if not has_comma_between:
                                is_part_of_district = True
                                # If the ward name is part of a district name and not separated by comma, heavily reduce its score
                                score *= 0.50  # Reduce the score by 50%
                                break
                if is_part_of_district:
                    break
            
            # Check if this ward exists in district list
            if district_trie.search_exact(norm_ward):
                # If we have province context, check if the ward is unique within that province
                if norm_province:
                    # Get all wards in this province
                    province_wards = []
                    for w in ward_trie.collect_all_words():
                        if norm_province in remove_diacritics(w.lower()):
                            province_wards.append(w)
                    
                    # Check if this ward is unique within the province
                    if len([w for w in province_wards if remove_diacritics(w.lower()) == norm_ward]) == 1:
                        filtered_ward_candidates.append((ward, score))
                        continue
                
                ambiguous_names.add(norm_ward)
            else:
                # Add to filtered candidates with appropriate score
                filtered_ward_candidates.append((ward, score))
        
        # Sort filtered ward candidates by score
        filtered_ward_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # If we have unique candidates, use them
        if unique_districts or filtered_ward_candidates:
            resolved_district = None
            resolved_ward = None
            
            # If we have a unique district, use it
            if unique_districts:
                resolved_district = unique_districts[0][0]  # Take the first (best) unique district
                # Find the best ward that's not the same as the district
                for ward, score in filtered_ward_candidates:
                    if remove_diacritics(ward.lower()) != remove_diacritics(resolved_district.lower()):
                        resolved_ward = ward
                        break
            
            # If we have a unique ward, use it
            if filtered_ward_candidates:
                resolved_ward = filtered_ward_candidates[0][0]  # Take the first (best) unique ward
                # Find the best district that's not the same as the ward
                for district, score in district_candidates:
                    if remove_diacritics(district.lower()) != remove_diacritics(resolved_ward.lower()):
                        resolved_district = district
                        break
            
            # If we still don't have both, use the best candidates
            if not resolved_district and district_candidates:
                resolved_district = district_candidates[0][0]
            if not resolved_ward and filtered_ward_candidates:
                resolved_ward = filtered_ward_candidates[0][0]
                
            return resolved_district, resolved_ward, True
        
        # If no unique candidates, try to use province context to break ties
        if norm_province:
            # Get all districts and wards in this province
            province_districts = []
            province_wards = []
            
            for d in district_trie.collect_all_words():
                if norm_province in remove_diacritics(d.lower()):
                    province_districts.append(d)
                    
            for w in ward_trie.collect_all_words():
                if norm_province in remove_diacritics(w.lower()):
                    province_wards.append(w)
            
            # Check if any of our candidates are more common in one list than the other
            for district, score in district_candidates:
                norm_district = remove_diacritics(district.lower())
                district_count = len([d for d in province_districts if remove_diacritics(d.lower()) == norm_district])
                ward_count = len([w for w in province_wards if remove_diacritics(w.lower()) == norm_district])
                
                if district_count > ward_count:
                    # This name is more common as a district in this province
                    return district, filtered_ward_candidates[0][0] if filtered_ward_candidates else None, True
            
            for ward, score in filtered_ward_candidates:
                norm_ward = remove_diacritics(ward.lower())
                district_count = len([d for d in province_districts if remove_diacritics(d.lower()) == norm_ward])
                ward_count = len([w for w in province_wards if remove_diacritics(w.lower()) == norm_ward])
                
                if ward_count > district_count:
                    # This name is more common as a ward in this province
                    return district_candidates[0][0] if district_candidates else None, ward, True
        
        # If no resolution possible, return the best matches if available
        if district_candidates and filtered_ward_candidates:
            return district_candidates[0][0], filtered_ward_candidates[0][0], False
        elif district_candidates:
            return district_candidates[0][0], None, False
        elif filtered_ward_candidates:
            return None, filtered_ward_candidates[0][0], False
        else:
            return None, None, False
        
    except Exception as e:
        print(f"Error in resolve_ambiguous_candidates: {str(e)}")
        return None, None, False