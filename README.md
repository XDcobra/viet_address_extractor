# Vietnamese Address Parser

A robust Python package for parsing and classifying Vietnamese addresses, with a focus on accurate location component identification and scoring.

## Features

- Accurate parsing of Vietnamese addresses into province, district, and ward components
- Support for various address formats and abbreviations
- Advanced scoring system for component matching
- Diacritic handling and text normalization
- Abbreviation expansion (e.g., "P." → "Phường", "Q." → "Quận")
- Trie-based dictionary matching for efficient lookups

## Installation

```bash
pip install wheel
python setup.py sdist bdist_wheel
pip install dist/vietnamese_address_classification-0.1-py3-none-any.whl
```

Note: This pip package is not published to the pip network. So you need to close this package and install it locally

## Usage

```python
from vietnamese_address_classification import parse_location_components

case = {
        "text": "357/28,Ng-T- Thuật,P1,Q3,TP.HồChíMinh.",
        "result": {
        "province": "Hồ Chí Minh",
        "district": "3",
        "ward": "1"
        }
    }
    
    result = parse_location_components(case["text"])
    print(result["province"])
    print(result["district"])
    print(result["ward"])

print(result)
# Output: {
#     'province': 'Hồ Chí Minh',
#     'district': '3',
#     'ward': '1'
# }
```

## Scoring System

The parser uses a sophisticated scoring system to accurately identify address components. Here are the key scoring features:

### Validation Thresholds
- Ward matching:
  - Non-ambiguous: 0.75
  - Ambiguous beginning: 0.85
  - Ambiguous middle: 0.75
- District matching:
  - Non-ambiguous: 0.75
  - Ambiguous beginning: 0.85
  - Ambiguous middle: 0.75

### Scoring Weights
- Fuzzy matching: 15%
- Exact matches: 35%
- Position bonuses:
  - Beginning of text: 25%
  - Middle of text: 30%
- Length bonuses:
  - 2-part names: 25%
  - 3-part names: 35%
  - 4-part names: 45%
- Additional bonuses:
  - Non-ambiguous matches: 10%
  - Comma presence: 25%
  - Indicator words: 25%
  - Full text matches: 30%
  - Original text matches: 20%
  - Unique ward matches: 20%
- Penalties:
  - Comma boundary violations: -50%

## Examples

Check the `examples` folder for detailed usage examples:

1. `example_simple.py`: Basic address parsing examples
2. `scoring_examples.py`: Examples demonstrating the scoring system
3. `abbreviation_handling.py`: Examples of abbreviation expansion
4. `diacritic_handling.py`: Examples of diacritic removal and handling
5. `example_with_options.py`: Another simple example how to set manual scoring values
6. `example_with_own_list.py`: Example how to set custom lists for province, district and wards


## Testing

Run the test suite:

```bash
pytest address_parser/tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

