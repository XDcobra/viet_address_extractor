from vietnamese_address_classification import parse_location_components

def example():
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

if __name__ == "__main__":
    example()