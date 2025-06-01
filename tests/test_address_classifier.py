import pytest

from vietnamese_address_classification.src.core.trie import Trie
from vietnamese_address_classification.src.utils.normalization import normalize_text
from vietnamese_address_classification.classifier import parse_location_components


def test_trie():
    trie = Trie()
    trie.insert("test")
    assert trie.search_exact("test") == True
    assert trie.search_exact("test1") == False
    
def test_normalize_text():
    """Test the text normalization function with various inputs."""
    test_cases = [
        {
            "input": "123 Đường Nguyễn Văn Cừ, Phường 4, Quận 5, TP HCM",
            "expected_tokens": ['123', 'duong', 'nguyen', 'van', 'cu', 'phuong', '4', 'quan', '5', 'thanh', 'pho', 'huyen', 'cm'],
            "expected_is_hcmc": True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case["input"]
        expected_tokens = test_case["expected_tokens"]
        expected_is_hcmc = test_case["expected_is_hcmc"]
        
        final_tokens, final_original_tokens, is_hcmc, final_groups = normalize_text(input_text)
        print(final_tokens)
        print(expected_tokens)


        # Check tokens
        assert final_tokens == expected_tokens
            
        # Check HCM flag
        assert is_hcmc == expected_is_hcmc

def test_classifier():
    """
    Run first 100 tests from test.json file and compare results with expected output.
    Uses the parse_location_components function with confidence scoring.
    """
    import json
    from collections import defaultdict
    import traceback
    
    print("\n=== Running First 100 Test Cases from test.json ===\n")
    
    try:
        
        # Load test cases
        test_cases = [
            {
                "text": "TT Tân Bình Huyện Yên Sơn, Tuyên Quang",
                "result": {
                    "province": "Tuyên Quang",
                    "district": "Yên Sơn",
                    "ward": "Tân Bình"
                }
            }
        ]

        # Run tests
        total_tests = len(test_cases)
        successful = 0
        failed = 0
        failures = defaultdict(list)

        print(f"\nRunning {total_tests} test cases...")

        indeeex = 0;
        
        for i, test_case in enumerate(test_cases, 1):
            try:

                print(f"\nProcessing test case {i} of {total_tests}")
                address = test_case['text']
                expected = test_case['result']
                
                print(f"\nTest Case {i}:")
                print(f"Input address: {address}")
                print(f"Expected result: {json.dumps(expected, ensure_ascii=False)}")

                
                print("Starting address parsing...")
                # Parse address using our new function
                result = parse_location_components(
                    address,
                    disable_file_logging=True,  # Enable logging for this test case
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
                    }
                )
                print("Address parsing completed")
                
                # Compare results
                province_match = result['province'] == expected['province']
                district_match = result['district'] == expected['district']
                ward_match = result['ward'] == expected['ward']
                
                print("\nActual result:")
                print(f"Province: {result['province']} {'✓' if province_match else '✗'}")
                print(f"District: {result['district']} {'✓' if district_match else '✗'}")
                print(f"Ward: {result['ward']} {'✓' if ward_match else '✗'}")
                
                assert province_match and district_match and ward_match

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
                    
                    print("\nStatus: FAILED ✗")
                    print(f"Failure reasons: {'; '.join(failure_reasons)}")
                    
                    failures[address] = failure_reasons
                
                print("-" * 80)
                
            except Exception as e:
                print(f"Error processing test case {i}: {str(e)}")
                failed += 1
                failures[address] = [f"Error: {str(e)}"]
                continue
        
        if failed > 0:
            print("\nFailed test cases:")
            for address, reasons in failures.items():
                print(f"\nAddress: {address}")
                print(f"Reasons: {'; '.join(reasons)}")

        # Print summary
        print("\n=== Test Results Summary ===")
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {(successful/total_tests)*100:.2f}%")

        print(indeeex)
                
    except Exception as e:
        print(f"Critical error in run_demo_test_file: {str(e)}")