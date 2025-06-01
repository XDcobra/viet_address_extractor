

def load_dictionary_files():
    """
    Load location data from text files:
    - list_province.txt: List of provinces
    - list_district.txt: List of districts
    - list_ward.txt: List of wards
    
    Each file contains one location name per line.
    
    :return: (list_of_provinces, list_of_districts, list_of_wards)
    """
    provinces = []
    districts = []
    wards = []
    
    # Load provinces
    try:
        with open('vietnamese_address_classification/data/list_province.txt', 'r', encoding='utf-8') as f:
            provinces = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Warning: Could not load provinces: {e}")
        
    # Load districts
    try:
        with open('vietnamese_address_classification/data/list_district.txt', 'r', encoding='utf-8') as f:
            districts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Warning: Could not load districts: {e}")
        
    # Load wards
    try:
        with open('vietnamese_address_classification/data/list_ward.txt', 'r', encoding='utf-8') as f:
            wards = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Warning: Could not load wards: {e}")
        
    return provinces, districts, wards