"""
Examples demonstrating the scoring system of the Vietnamese Address Parser.
"""

from vietnamese_address_classification import parse_location_components

def main():
    
    # Example addresses with different scoring scenarios
    addresses = [
        # Example 1: Clear component separation with commas
        {
            "address": "123 Đường Nguyễn Văn Cừ, Phường 4, Quận 5, TP HCM",
            "description": "Clear component separation with commas"
        },
        # Example 2: Ambiguous ward name at beginning
        {
            "address": "Bến Nghé, Quận 1, TP HCM",
            "description": "Ambiguous ward name at beginning"
        },
        # Example 3: Multi-part district name
        {
            "address": "Phường 4, Quận Gò Vấp, TP HCM",
            "description": "Multi-part district name"
        },
        # Example 4: No commas, relying on position
        {
            "address": "Phường 4 Quận 5 TP HCM",
            "description": "No commas, relying on position"
        },
        # Example 5: Abbreviated components
        {
            "address": "P.4 Q.5 TPHCM",
            "description": "Abbreviated components"
        }
    ]
    
    print("Scoring System Examples\n")
    print("-" * 80)
    
    for example in addresses:
        print(f"\nExample: {example['description']}")
        print(f"Input address: {example['address']}")
        
        result = parse_location_components(
            example['address'],
            validation_thresholds={
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
                    "2_parts": 0.25,
                    "3_parts": 0.35,
                    "4_parts": 0.45
                },
                "non_ambiguous_bonus": 0.1,
                "comma_bonus": 0.25,
                "indicator_bonus": 0.25,
                "full_text_match_bonus": 0.3,
                "original_text_match_bonus": 0.2,
                "unique_ward_bonus": 0.2,
                "comma_boundary_penalty": -0.5
            }
        )
        
        print("Parsed components:")
        print(f"  Province: {result['province']}")
        print(f"  District: {result['district']}")
        print(f"  Ward: {result['ward']}")
        print("-" * 80)

if __name__ == "__main__":
    main() 