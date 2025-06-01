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

    provinces = ["Hà Nội", "Đà Nẵng"]
    districts = [3]
    wards = [1]


    result = parse_location_components(
        text=case["text"], 
        district_list=districts, 
        province_list=provinces, 
        ward_list=wards)
    
    print(result)

if __name__ == "__main__":
    example()