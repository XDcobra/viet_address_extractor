from vietnamese_address_classification.classifier import parse_location_components

def example():
    case = {
        "text": "357/28,Ng-T- Thuật,P1,Q3,TP.HồChíMinh.",
        "result": {
        "province": "Hồ Chí Minh",
        "district": "3",
        "ward": "1"
        }
    }
    
    result = parse_location_components(
        text=case["text"], 
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
            "unique_ward_bonus": 0.2,  # New bonus for matches that can only be wards,
            "comma_boundary_penalty": -0.5
        })
    print(result["province"])
    print(result["district"])
    print(result["ward"])

if __name__ == "__main__":
    example()