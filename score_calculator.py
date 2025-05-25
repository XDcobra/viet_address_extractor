import json
import logging
from typing import List, Dict, Tuple
from address_parser import (
    load_dictionary_files,
    build_trie_from_list,
    parse_location_components
)
from itertools import product
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('score_optimization.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('score_optimizer')

def generate_weight_combinations():
    """
    Generate combinations of weights for scoring.
    Now includes indicator bonus for ward/district indicators.
    """
    # Define ranges for each weight component - adjusted to sum to ~2.0
    weight_ranges = {
        'fuzzy_score': [0.15, 0.2, 0.25],  # 3 values around 0.2
        'exact_match': [0.3, 0.35, 0.4],   # 3 values around 0.35
        'position_bonus': {
            'beginning': [0.15, 0.2, 0.25],  # 3 values around 0.2
            'middle': [0.2, 0.25, 0.3]      # 3 values around 0.25
        },
        'length_bonus': [0.15, 0.2, 0.25],  # 3 values around 0.2
        'non_ambiguous_bonus': [0.1, 0.15, 0.2],  # 3 values around 0.15
        'comma_bonus': [0.15, 0.2, 0.25],   # 3 values around 0.2
        'indicator_bonus': [0.15, 0.2, 0.25]  # 3 values around 0.2
    }
    
    combinations = []
    
    # Generate combinations ensuring they sum to 2.0
    for fuzzy in weight_ranges['fuzzy_score']:
        for exact in weight_ranges['exact_match']:
            for beg_pos in weight_ranges['position_bonus']['beginning']:
                for mid_pos in weight_ranges['position_bonus']['middle']:
                    for length in weight_ranges['length_bonus']:
                        for non_amb in weight_ranges['non_ambiguous_bonus']:
                            for comma in weight_ranges['comma_bonus']:
                                for indicator in weight_ranges['indicator_bonus']:
                                    # Calculate total weight
                                    total = (fuzzy + exact + beg_pos + mid_pos + 
                                           length + non_amb + comma + indicator)
                                    
                                    # Only keep combinations that sum to approximately 2.0
                                    if 1.9 <= total <= 2.1:  # Slightly wider range
                                        combinations.append({
                                            'fuzzy_score': fuzzy,
                                            'exact_match': exact,
                                            'position_bonus': {
                                                'beginning': beg_pos,
                                                'middle': mid_pos
                                            },
                                            'length_bonus': length,
                                            'non_ambiguous_bonus': non_amb,
                                            'comma_bonus': comma,
                                            'indicator_bonus': indicator
                                        })
    
    # If no combinations found, add our current best known combination
    if not combinations:
        combinations.append({
            'fuzzy_score': 0.3,
            'exact_match': 0.6,
            'position_bonus': {
                'beginning': 0.3,
                'middle': 0.4
            },
            'length_bonus': 0.4,
            'non_ambiguous_bonus': 0.2,
            'comma_bonus': 0.3,
            'indicator_bonus': 0.4
        })
    
    return combinations

def generate_threshold_combinations():
    """
    Generate combinations of thresholds for validation.
    Adjusted for 2.0 scale with more focused values.
    """
    # Define ranges for thresholds (all on 2.0 scale) - focused on promising ranges
    threshold_ranges = {
        'ward': {
            'non_ambiguous': [0.75, 0.8, 0.85],      # 3 values around 0.8
            'ambiguous_beginning': [0.85, 0.9, 0.95], # 3 values around 0.9
            'ambiguous_middle': [0.75, 0.8, 0.85]     # 3 values around 0.8
        },
        'district': {
            'non_ambiguous': [0.75, 0.8, 0.85],      # 3 values around 0.8
            'ambiguous_beginning': [0.85, 0.9, 0.95], # 3 values around 0.9
            'ambiguous_middle': [0.75, 0.8, 0.85]     # 3 values around 0.8
        }
    }
    
    combinations = []
    
    # Generate all combinations
    for ward_non_amb in threshold_ranges['ward']['non_ambiguous']:
        for ward_amb_beg in threshold_ranges['ward']['ambiguous_beginning']:
            for ward_amb_mid in threshold_ranges['ward']['ambiguous_middle']:
                for dist_non_amb in threshold_ranges['district']['non_ambiguous']:
                    for dist_amb_beg in threshold_ranges['district']['ambiguous_beginning']:
                        for dist_amb_mid in threshold_ranges['district']['ambiguous_middle']:
                            # Add validation to ensure thresholds make sense
                            if (ward_non_amb <= ward_amb_beg and  # Non-ambiguous should be lower than ambiguous
                                ward_amb_mid <= ward_amb_beg and  # Middle should be lower than beginning
                                dist_non_amb <= dist_amb_beg and  # Same for district
                                dist_amb_mid <= dist_amb_beg):
                                combinations.append({
                                    'ward': {
                                        'non_ambiguous': ward_non_amb,
                                        'ambiguous_beginning': ward_amb_beg,
                                        'ambiguous_middle': ward_amb_mid
                                    },
                                    'district': {
                                        'non_ambiguous': dist_non_amb,
                                        'ambiguous_beginning': dist_amb_beg,
                                        'ambiguous_middle': dist_amb_mid
                                    }
                                })
    
    # If no combinations found, add our current best known combination
    if not combinations:
        combinations.append({
            'ward': {
                'non_ambiguous': 0.8,
                'ambiguous_beginning': 0.9,
                'ambiguous_middle': 0.8
            },
            'district': {
                'non_ambiguous': 0.8,
                'ambiguous_beginning': 0.9,
                'ambiguous_middle': 0.8
            }
        })
    
    return combinations

def evaluate_combination(weights, thresholds, test_cases):
    """
    Evaluate a combination of weights and thresholds on test cases.
    Now includes detailed logging of which combinations work best for specific cases.
    """
    try:
        # Load dictionary data and build tries
        province_list, district_list, ward_list = load_dictionary_files()
        province_trie = build_trie_from_list(province_list)
        district_trie = build_trie_from_list(district_list)
        ward_trie = build_trie_from_list(ward_list)
        
        total_score = 0.0
        total_tests = len(test_cases)
        case_results = []  # Store results for each test case
        
        for test_case in test_cases:
            address = test_case['text']
            expected = test_case['result']
            
            # Parse address using parse_location_components with file logging disabled
            result = parse_location_components(
                address,
                province_trie=province_trie,
                district_trie=district_trie,
                ward_trie=ward_trie,
                scoring_weights=weights,
                validation_thresholds=thresholds,
                disable_file_logging=True  # Disable file logging
            )
            
            # Calculate score for each component (2.0 points each)
            province_score = 2.0 if result['province'] == expected['province'] else 0.0
            district_score = 2.0 if result['district'] == expected['district'] else 0.0
            ward_score = 2.0 if result['ward'] == expected['ward'] else 0.0
            
            # Store detailed results for this case
            case_results.append({
                'address': address,
                'expected': expected,
                'result': result,
                'scores': {
                    'province': province_score,
                    'district': district_score,
                    'ward': ward_score
                }
            })
            
            # Add scores for this test case
            total_score += province_score + district_score + ward_score
        
        # Calculate final score (normalized to 0-1)
        final_score = total_score / (total_tests * 6.0)  # 6.0 = 2.0 points per component * 3 components
        
        # Return both the score and detailed results
        return final_score, case_results
        
    except Exception as e:
        logger.error(f"Error in evaluate_combination: {str(e)}")
        return 0.0, []  # Return 0 score and empty results on error

def evaluate_combination_wrapper(args):
    """
    Wrapper function for evaluate_combination to be used with ThreadPoolExecutor.
    """
    test_cases, weights, thresholds = args
    return evaluate_combination(weights, thresholds, test_cases)

def save_optimization_result(weights, thresholds, success_rate, case_results):
    """
    Save optimization results to a JSON file.
    Appends new results to existing ones if the file exists.
    Excludes case_results to keep the file size manageable.
    """
    try:
        # Try to load existing results
        try:
            with open('optimization_results.json', 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
        except (FileNotFoundError, json.JSONDecodeError):
            existing_results = []
        
        # Create new result entry (without case_results)
        new_result = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'success_rate': success_rate,
            'weights': weights,
            'thresholds': thresholds
        }
        
        # Add new result to list
        existing_results.append(new_result)
        
        # Sort results by success rate in descending order
        existing_results.sort(key=lambda x: x['success_rate'], reverse=True)
        
        # Save updated results
        with open('optimization_results.json', 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Error saving optimization results: {str(e)}")

def find_optimal_weights(test_file='test.json', max_tests=10):
    """
    Find the optimal combination of weights and thresholds using multithreading.
    Now includes detailed logging of which combinations work best for specific cases.
    """
    try:
        # Load test cases
        with open(test_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)[:max_tests]
        
        # Generate weight and threshold combinations
        weight_combinations = generate_weight_combinations()
        threshold_combinations = generate_threshold_combinations()
        
        total_combinations = len(weight_combinations) * len(threshold_combinations)
        print(f"\nTesting {total_combinations} combinations...")
        
        best_success_rate = 0
        best_weights = None
        best_thresholds = None
        best_case_results = None
        
        # Prepare arguments for parallel processing
        args_list = []
        for weights in weight_combinations:
            for thresholds in threshold_combinations:
                args_list.append((
                    test_cases,
                    weights,
                    thresholds
                ))
        
        # Determine number of threads (use CPU count - 1 to leave one core free)
        num_threads = max(1, multiprocessing.cpu_count() - 1)
        
        # Process combinations in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(evaluate_combination_wrapper, args): args 
                for args in args_list
            }
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_args), 1):
                args = future_to_args[future]
                weights, thresholds = args[1], args[2]
                
                try:
                    success_rate, case_results = future.result()
                    
                    # Only print current combination and its success rate
                    print(f"\rTesting combination {i}/{total_combinations} - Success rate: {success_rate:.2%}", end="", flush=True)
                    
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_weights = weights
                        best_thresholds = thresholds
                        best_case_results = case_results
                        
                        print(f"\n\nNew best combination found! Success rate: {best_success_rate:.2%}")
                        
                        # Save the new best combination immediately
                        save_optimization_result(
                            best_weights,
                            best_thresholds,
                            best_success_rate,
                            best_case_results
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing combination {i}: {str(e)}")
        
        # If no best combination found, use our current best known values
        if best_success_rate == 0:
            best_weights = {
                'fuzzy_score': 0.3,
                'exact_match': 0.6,
                'position_bonus': {
                    'beginning': 0.3,
                    'middle': 0.4
                },
                'length_bonus': 0.4,
                'non_ambiguous_bonus': 0.2,
                'comma_bonus': 0.3,
                'indicator_bonus': 0.4
            }
            best_thresholds = {
                'ward': {
                    'non_ambiguous': 0.8,
                    'ambiguous_beginning': 0.9,
                    'ambiguous_middle': 0.8
                },
                'district': {
                    'non_ambiguous': 0.8,
                    'ambiguous_beginning': 0.9,
                    'ambiguous_middle': 0.8
                }
            }
            best_success_rate = 0.0
            best_case_results = []
            
            # Save the default values
            save_optimization_result(
                best_weights,
                best_thresholds,
                best_success_rate,
                best_case_results
            )
        
        return best_weights, best_thresholds, best_success_rate, best_case_results
        
    except Exception as e:
        logger.error(f"Critical error in find_optimal_weights: {str(e)}")
        return None, None, 0.0, []

if __name__ == '__main__':
    best_weights, best_thresholds, best_success_rate, best_case_results = find_optimal_weights(max_tests=100)
    
    print("\n=== Optimization Results ===")
    print(f"Best success rate: {best_success_rate:.2%}")
    print("\nBest weights:")
    print(json.dumps(best_weights, indent=2))
    print("\nBest thresholds:")
    print(json.dumps(best_thresholds, indent=2)) 