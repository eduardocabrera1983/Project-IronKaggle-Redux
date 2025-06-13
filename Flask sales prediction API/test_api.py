import requests
import json

def test_api(base_url="http://localhost:5000"):
    """Test the Flask API with various scenarios"""
    
    print(f"Testing API at: {base_url}")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    print("\n" + "-" * 30 + "\n")
    
    # Test 2: Valid prediction request (exact format from user)
    print("2. Testing valid prediction...")
    valid_payload = {
        "store_ID": 49,
        "day_of_week": 4,
        "date": "26/06/2014",
        "nb_customers_on_day": 1254,
        "open": 1,
        "promotion": 0,
        "state_holiday": "0",
        "school_holiday": 1
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=valid_payload)
        print(f"Prediction status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Valid prediction failed: {e}")
    
    print("\n" + "-" * 30 + "\n")
    
    # Test 3: Alternative date format
    print("3. Testing alternative date format...")
    alt_date_payload = {
        "store_ID": "25",  # String instead of int
        "day_of_week": "3",  # String instead of int
        "date": "2014-06-26",  # Different date format
        "nb_customers_on_day": "800",  # String instead of int
        "open": "1",
        "promotion": "0",
        "state_holiday": "0",
        "school_holiday": "0"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=alt_date_payload)
        print(f"Alt format status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Alt format test failed: {e}")
    
    print("\n" + "-" * 30 + "\n")
    
    # Test 4: Invalid input (missing field)
    print("4. Testing invalid input (missing field)...")
    invalid_payload = {
        "store_ID": 49,
        "day_of_week": 4,
        # Missing "date" field
        "nb_customers_on_day": 1254,
        "open": 1,
        "promotion": 0,
        "state_holiday": "0",
        "school_holiday": 1
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=invalid_payload)
        print(f"Invalid input status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Invalid input test failed: {e}")
    
    print("\n" + "-" * 30 + "\n")
    
    # Test 5: Invalid data types
    print("5. Testing invalid data types...")
    invalid_type_payload = {
        "store_ID": "not_a_number",
        "day_of_week": 4,
        "date": "26/06/2014",
        "nb_customers_on_day": 1254,
        "open": 1,
        "promotion": 0,
        "state_holiday": "0",
        "school_holiday": 1
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=invalid_type_payload)
        print(f"Invalid type status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Invalid type test failed: {e}")

if __name__ == "__main__":
    # Test locally
    print("Testing local API (make sure Flask app is running)...")
    test_api("http://localhost:5000")
    
    # Uncomment below to test on AWS (replace with your EC2 public IP)
    # print("\n" + "=" * 60 + "\n")
    # print("Testing AWS API...")
    # test_api("http://YOUR_EC2_PUBLIC_IP:5000")