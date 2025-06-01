"""
Examples demonstrating the diacritic handling capabilities of the Vietnamese Address Parser.
"""

from vietnamese_address_classification import parse_location_components
from vietnamese_address_classification.src.utils.normalization import normalize_text, remove_diacritics


def main():
    
    # Example addresses with diacritics
    addresses = [
        # Example 1: Full diacritics
        {
            "address": "123 Đường Nguyễn Văn Cừ, Phường 4, Quận 5, TP HCM",
            "description": "Full diacritics"
        },
        # Example 2: Mixed diacritics
        {
            "address": "45 Le Duan, Phường Bến Nghé, Quận 1, TP HCM",
            "description": "Mixed diacritics"
        },
        # Example 3: No diacritics
        {
            "address": "67 Duong so 3, Phuong 7, Quan 3, TP Ho Chi Minh",
            "description": "No diacritics"
        },
        # Example 4: Complex diacritics
        {
            "address": "89 Trần Hưng Đạo, Phường Cầu Ông Lãnh, Quận 1, TPHCM",
            "description": "Complex diacritics"
        },
        # Example 5: Special characters
        {
            "address": "321 Hai Bà Trưng, Phường Tân Định, Quận 1, HCM",
            "description": "Special characters"
        }
    ]
    
    print("Diacritic Handling Examples\n")
    print("-" * 80)
    
    # First, demonstrate diacritic removal
    print("\nDiacritic Removal Examples:")
    print("-" * 40)
    test_strings = [
        "Nguyễn Văn Cừ",
        "Lê Duẩn",
        "Bến Nghé",
        "Cầu Ông Lãnh",
        "Hai Bà Trưng"
    ]
    for text in test_strings:
        removed = remove_diacritics(text)
        print(f"{text:20} → {removed}")
    
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