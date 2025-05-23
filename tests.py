import re
import json
from address_parser import normalize_text, expand_abbreviations, remove_diacritics

def load_test_cases():
    """Load test cases from test.json file"""
    try:
        with open('test.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading test cases: {str(e)}")
        return None

def load_norm_abb_test_cases():
    """Load test cases from test_norm_abb.json file"""
    try:
        with open('test_norm_abb.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading normalization and abbreviation test cases: {str(e)}")
        return None

def test_remove_diacritics():
    """Test diacritic removal with real test cases from test.json"""
    print("\nTesting remove_diacritics:")
    
    # Load test cases
    test_cases = load_test_cases()
    if not test_cases:
        print("❌ Could not load test cases")
        return False
        
    total_tests = 0
    passed_tests = 0
    
    # Run tests
    print("\nRunning tests:")
    for i, test in enumerate(test_cases, 1):
        input_text = test["text"]
        
        # Test name is first 40 chars of input
        test_name = f"Test {i}: {input_text[:40]}{'...' if len(input_text) > 40 else ''}"
        result_str = f"{test_name:<60}"
        
        # Remove diacritics and compare
        result = remove_diacritics(input_text)
        
        # For comparison, we need to:
        # 1. Convert both strings to lowercase
        # 2. Remove punctuation from both
        # 3. Normalize spaces in both
        def normalize_for_comparison(s):
            # Remove punctuation except digits
            s = re.sub(r'[^\w\s\d]', ' ', s)
            # Normalize spaces
            s = re.sub(r'\s+', ' ', s).strip()
            return s.lower()
            
        result_normalized = normalize_for_comparison(result)
        input_normalized = normalize_for_comparison(input_text)
        
        # The only difference between result and input should be the diacritics
        # All other aspects (spaces, punctuation) should be preserved
        total_tests += 1
        
        # Check if the normalized versions are different (they should be if input had diacritics)
        # and if the result doesn't have any diacritics
        has_diacritics = lambda s: bool(re.search(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', s))
        
        if not has_diacritics(result) and (has_diacritics(input_text) or result_normalized == input_normalized):
            result_str += "✅"
            passed_tests += 1
        else:
            result_str += "❌"
            print(result_str)
            print(f"{'Input:':<12}{input_text}")
            print(f"{'Got:':<12}{result}")
            continue
            
        print(result_str)
    
    # Print summary
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    if passed_tests == total_tests:
        print("✅ All remove_diacritics tests passed!")
    else:
        print("❌ Some remove_diacritics tests failed")
    
    return passed_tests == total_tests

def test_expand_abbreviations():
    """Test abbreviation expansion with real test cases from test.json"""
    print("\nTesting expand_abbreviations:")
    
    # Load test cases
    test_cases = load_test_cases()
    if not test_cases:
        print("❌ Could not load test cases")
        return False
        
    total_tests = 0
    passed_tests = 0
    
    # Run tests
    print("\nRunning tests:")
    for i, test in enumerate(test_cases, 1):
        input_text = test["text"]
        # Get normalized text from test case
        normalized = normalize_text(input_text)
        if not normalized:
            continue
            
        # Test name is first 40 chars of input
        test_name = f"Test {i}: {input_text[:40]}{'...' if len(input_text) > 40 else ''}"
        result_str = f"{test_name:<60}"
        
        # Test each token for abbreviation expansion
        total_tests += 1
        all_passed = True
        for token in input_text.lower().split():
            result = expand_abbreviations(token)
            # Check if expansion matches expected patterns
            if token.startswith(('p', 'q')) and token[1:].isdigit():
                expected = 'phuong ' + token[1:] if token[0] == 'p' else 'quan ' + token[1:]
                if result != expected:
                    all_passed = False
                    print(f"Token expansion failed: {token} -> {result} (expected {expected})")
                    
        if all_passed:
            result_str += "✅"
            passed_tests += 1
        else:
            result_str += "❌"
            
        print(result_str)
    
    # Print summary
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    if passed_tests == total_tests:
        print("✅ All expand_abbreviations tests passed!")
    else:
        print("❌ Some expand_abbreviations tests failed")
    
    return passed_tests == total_tests

def test_normalize_text():
    """Test the text normalization function with various inputs."""
    test_cases = [
        {
            "input": "123 Đường Nguyễn Văn Cừ, Phường 4, Quận 5, TP HCM",
            "expected_tokens": ['123', 'duong', 'nguyen', 'van', 'cu', 'phuong', '4', 'quan', '5', 'tp', 'hcm'],
            "expected_is_hcmc": True
        },
        {
            "input": "45 Lê Duẩn, P. Bến Nghé, Quận 1, Thành phố Hồ Chí Minh",
            "expected_tokens": ['45', 'le', 'duan', 'phuong', 'ben', 'nghe', 'quan', '1', 'tp', 'hcm'],
            "expected_is_hcmc": True
        },
        {
            "input": "67 Đường số 3, Phường 7, Quận 3, TP. Hồ Chí Minh",
            "expected_tokens": ['67', 'duong', 'so', '3', 'phuong', '7', 'quan', '3', 'tp', 'hcm'],
            "expected_is_hcmc": True
        },
        {
            "input": "89 Trần Hưng Đạo, Phường Cầu Ông Lãnh, Quận 1, TPHCM",
            "expected_tokens": ['89', 'tran', 'hung', 'dao', 'phuong', 'cau', 'ong', 'lanh', 'quan', '1', 'tp', 'hcm'],
            "expected_is_hcmc": True
        },
        {
            "input": "321 Hai Bà Trưng, Phường Tân Định, Quận 1, HCM",
            "expected_tokens": ['321', 'hai', 'ba', 'trung', 'phuong', 'tan', 'dinh', 'quan', '1', 'tp', 'hcm'],
            "expected_is_hcmc": True
        },
        {
            "input": "123 Đường Lê Lợi, Phường Bến Thành, Quận 1, Hà Nội",
            "expected_tokens": ['123', 'duong', 'le', 'loi', 'phuong', 'ben', 'thanh', 'quan', '1', 'ha', 'noi'],
            "expected_is_hcmc": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case["input"]
        expected_tokens = test_case["expected_tokens"]
        expected_is_hcmc = test_case["expected_is_hcmc"]
        
        result_tokens, result_is_hcmc = normalize_text(input_text)
        
        # Check tokens
        if result_tokens != expected_tokens:
            print(f"❌ Test {i} failed - Token mismatch:")
            print(f"  Input: {input_text}")
            print(f"  Expected tokens: {expected_tokens}")
            print(f"  Got tokens: {result_tokens}")
            return False
            
        # Check HCM flag
        if result_is_hcmc != expected_is_hcmc:
            print(f"❌ Test {i} failed - HCM flag mismatch:")
            print(f"  Input: {input_text}")
            print(f"  Expected is_hcmc: {expected_is_hcmc}")
            print(f"  Got is_hcmc: {result_is_hcmc}")
            return False
    
    print("✅ All normalization tests passed!")
    return True

def run_all_tests():
    """Run all test functions"""
    tests = [
        test_remove_diacritics,
        test_expand_abbreviations,
        test_normalize_text
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    test_remove_diacritics()