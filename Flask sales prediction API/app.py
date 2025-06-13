import pickle
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to store the model
model = None

def load_model():
    """Load the XGBoost model from pickle file"""
    global model
    try:
        with open('XGBRegressor.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def parse_date(date_str):
    """Parse date string in various formats and extract features"""
    try:
        # Try different date formats
        date_formats = [
            "%d/%m/%Y",  # 26/06/2014
            "%Y-%m-%d",  # 2014-06-26
            "%m/%d/%Y",  # 06/26/2014
            "%d-%m-%Y",  # 26-06-2014
        ]
        
        parsed_date = None
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(str(date_str), fmt)
                break
            except ValueError:
                continue
        
        if parsed_date is None:
            raise ValueError(f"Could not parse date: {date_str}")
        
        # Extract features that might be useful
        return {
            'year': parsed_date.year,
            'month': parsed_date.month,
            'day': parsed_date.day,
            'weekday': parsed_date.weekday(),  # 0=Monday, 6=Sunday
        }
    except Exception as e:
        raise ValueError(f"Error parsing date '{date_str}': {str(e)}")

def validate_and_convert_input(data):
    """Validate and convert input data to the correct format"""
    try:
        # Required fields
        required_fields = [
            'store_ID', 'day_of_week', 'date', 'nb_customers_on_day',
            'open', 'promotion', 'state_holiday', 'school_holiday'
        ]
        
        # Check for missing required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Convert and validate each field
        processed_data = {}
        
        # store_ID - should be integer
        try:
            processed_data['store_ID'] = int(data['store_ID'])
        except (ValueError, TypeError):
            raise ValueError(f"store_ID must be a valid integer, got: {data['store_ID']}")
        
        # day_of_week - should be integer 0-6 or 1-7
        try:
            dow = int(data['day_of_week'])
            if dow < 0 or dow > 7:
                raise ValueError(f"day_of_week must be between 0-7, got: {dow}")
            processed_data['day_of_week'] = dow
        except (ValueError, TypeError):
            raise ValueError(f"day_of_week must be a valid integer, got: {data['day_of_week']}")
        
        # Parse date
        date_info = parse_date(data['date'])
        processed_data['date'] = data['date']  # Keep original for reference
        processed_data.update(date_info)
        
        # nb_customers_on_day - should be numeric
        try:
            processed_data['nb_customers_on_day'] = float(data['nb_customers_on_day'])
        except (ValueError, TypeError):
            raise ValueError(f"nb_customers_on_day must be numeric, got: {data['nb_customers_on_day']}")
        
        # open - should be 0 or 1
        try:
            open_val = int(data['open'])
            if open_val not in [0, 1]:
                raise ValueError(f"open must be 0 or 1, got: {open_val}")
            processed_data['open'] = open_val
        except (ValueError, TypeError):
            raise ValueError(f"open must be 0 or 1, got: {data['open']}")
        
        # promotion - should be 0 or 1
        try:
            promo_val = int(data['promotion'])
            if promo_val not in [0, 1]:
                raise ValueError(f"promotion must be 0 or 1, got: {promo_val}")
            processed_data['promotion'] = promo_val
        except (ValueError, TypeError):
            raise ValueError(f"promotion must be 0 or 1, got: {data['promotion']}")
        
        # state_holiday - convert to numeric (0 if "0", 1 otherwise)
        state_holiday = str(data['state_holiday']).strip()
        if state_holiday == "0":
            processed_data['state_holiday'] = 0
        else:
            processed_data['state_holiday'] = 1
        
        # school_holiday - should be 0 or 1
        try:
            school_val = int(data['school_holiday'])
            if school_val not in [0, 1]:
                raise ValueError(f"school_holiday must be 0 or 1, got: {school_val}")
            processed_data['school_holiday'] = school_val
        except (ValueError, TypeError):
            raise ValueError(f"school_holiday must be 0 or 1, got: {data['school_holiday']}")
        
        return processed_data
    
    except Exception as e:
        raise ValueError(f"Input validation error: {str(e)}")

def prepare_model_input(processed_data):
    """Prepare input data for the XGBoost model"""
    try:
        # Create feature array in the expected order for the model
        # Adjust this based on your actual model's feature order
        features = [
            processed_data['store_ID'],
            processed_data['day_of_week'],
            processed_data['nb_customers_on_day'],
            processed_data['open'],
            processed_data['promotion'],
            processed_data['state_holiday'],
            processed_data['school_holiday'],
            processed_data['year'],
            processed_data['month'],
            processed_data['day'],
            processed_data['weekday']
        ]
        
        # Convert to numpy array and reshape for single prediction
        X = np.array(features).reshape(1, -1)
        
        return X
    
    except Exception as e:
        raise ValueError(f"Error preparing model input: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.'
            }), 500
        
        # Get JSON data from request
        if not request.is_json:
            return jsonify({
                'error': 'Request must contain JSON data'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400
        
        # Validate and convert input
        processed_data = validate_and_convert_input(data)
        
        # Prepare input for model
        X = prepare_model_input(processed_data)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Return prediction
        return jsonify({
            'prediction': float(prediction),
            'input_processed': processed_data,
            'status': 'success'
        })
    
    except ValueError as e:
        return jsonify({
            'error': f'Input validation error: {str(e)}'
        }), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'XGBoost Regression API',
        'endpoints': {
            'POST /predict': 'Make predictions',
            'GET /health': 'Health check',
            'GET /': 'This documentation'
        },
        'example_request': {
            'store_ID': 49,
            'day_of_week': 4,
            'date': '26/06/2014',
            'nb_customers_on_day': 1254,
            'open': 1,
            'promotion': 0,
            'state_holiday': '0',
            'school_holiday': 1
        }
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Flask app...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)