import os
from address_parser import Trie, parse_location_components, load_dictionary_files, build_trie_from_list

# ✅ Normalization mappings
def normalize(name, mapping):
    name = name.lower().strip()
    return mapping.get(name, name)

same_province = {
    "hcm": "Hồ Chí Minh", "tp hcm": "Hồ Chí Minh", "tp. hcm": "Hồ Chí Minh",
    "sài gòn": "Hồ Chí Minh", "ho chi minh": "Hồ Chí Minh", "hồ chí minh": "Hồ Chí Minh",
    "hn": "Hà Nội", "tp hn": "Hà Nội", "tp. hn": "Hà Nội", "ha noi": "Hà Nội",
}

same_district = {
    "q1": "Quận 1", "quan 1": "Quận 1", "quận 1": "Quận 1",
    "q2": "Quận 2", "quan 2": "Quận 2", "quận 2": "Quận 2",
    "q3": "Quận 3", "quan 3": "Quận 3", "quận 3": "Quận 3",
    "thủ đức": "Thành phố Thủ Đức", "thu duc": "Thành phố Thủ Đức",
    "binh thanh": "Quận Bình Thạnh", "bình thạnh": "Quận Bình Thạnh",
    "go vap": "Quận Gò Vấp", "gò vấp": "Quận Gò Vấp",
    "tan binh": "Quận Tân Bình", "tân bình": "Quận Tân Bình",
}

same_ward = {
    "p1": "Phường 1", "phuong 1": "Phường 1", "phường 1": "Phường 1",
    "p2": "Phường 2", "phuong 2": "Phường 2", "phường 2": "Phường 2",
    "p3": "Phường 3", "phuong 3": "Phường 3", "phường 3": "Phường 3",
    "p.13": "Phường 13", "phuong 13": "Phường 13",
    "thảo điền": "Phường Thảo Điền", "thao dien": "Phường Thảo Điền",
    "tân định": "Phường Tân Định", "tan dinh": "Phường Tân Định",
}

class Solution:
    def __init__(self):
        # Initialize tries for each level
        self.province_trie = Trie()
        self.district_trie = Trie()
        self.ward_trie = Trie()

            # Load dictionary data and build tries once
        province_list, district_list, ward_list = load_dictionary_files()
        self.province_trie = build_trie_from_list(province_list)
        self.district_trie = build_trie_from_list(district_list)
        self.ward_trie = build_trie_from_list(ward_list)

    def process(self, s: str):
        # Get all matches
        result = parse_location_components(self,
                    s,
                    province_trie=self.province_trie,
                    district_trie=self.district_trie,
                    ward_trie=self.ward_trie,
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
                        "unique_ward_bonus": 0.2  # New bonus for matches that can only be wards
                    }
                )
        province = result["province"]
        district = result["district"]
        ward = result["ward"]

        if province == None:
           province = ""
        if district == None:
            district = ""
        if ward == None:
            ward = ""

        return {
            "province": province,
            "district": district,
            "ward": ward
        }
