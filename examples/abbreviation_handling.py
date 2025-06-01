"""
Examples demonstrating the abbreviation handling capabilities of the Vietnamese Address Parser.
"""

from vietnamese_address_classification import parse_location_components
from vietnamese_address_classification.src.utils.normalization import normalize_text
from vietnamese_address_classification.src.utils.abbreviation_expander import expand_abbreviations

def main():
    
    # Example addresses with abbreviations
    addresses = [
        # Example 1: Common abbreviations
        {
            "address": "123 Đường Lê Lợi, P. Bến Thành, Q.1, TP.HCM",
            "description": "Common abbreviations (P., Q., TP.)"
        },
        # Example 2: Numbered ward/district
        {
            "address": "45 Nguyễn Huệ, P.4, Q.5, TPHCM",
            "description": "Numbered ward/district"
        },
        # Example 3: Mixed abbreviations
        {
            "address": "67 Lê Duẩn, P.Bến Nghé, Q.1, TP Hồ Chí Minh",
            "description": "Mixed abbreviations"
        },
        # Example 4: Multiple abbreviations
        {
            "address": "89 Trần Hưng Đạo, P.Cầu Ông Lãnh, Q.1, TPHCM",
            "description": "Multiple abbreviations"
        },
        # Example 5: Abbreviated province
        {
            "address": "321 Hai Bà Trưng, P.Tân Định, Q.1, HCM",
            "description": "Abbreviated province"
        }
    ]
    
    print("Abbreviation Handling Examples\n")
    print("-" * 80)
    
    # First, demonstrate abbreviation expansion
    print("\nAbbreviation Expansion Examples:")
    print("-" * 40)
    test_tokens = ["P.4", "Q.5", "TP.HCM", "P.Bến Nghé", "Q.1"]
    for token in test_tokens:
        expanded = expand_abbreviations(token)
        print(f"{token:15} → {expanded}")
    
    print("\nAddress Parsing Examples:")
    print("-" * 80)
    
    for example in addresses:
        print(f"\nExample: {example['description']}")
        print(f"Input address: {example['address']}")
        
        # First show the normalized text
        normalized_tokens, is_hcmc = normalize_text(example['address'])
        print(f"Normalized tokens: {normalized_tokens}")
        
        # Then show the parsed result
        result = parse_location_components(
            example['address']
        )
        
        print("Parsed components:")
        print(f"  Province: {result['province']}")
        print(f"  District: {result['district']}")
        print(f"  Ward: {result['ward']}")
        print("-" * 80)

if __name__ == "__main__":
    main() 