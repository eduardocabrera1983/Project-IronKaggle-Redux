import pickle
from flask import Flask, request, jsonify
from dateutil import parser as date_parser
import pandas as pd

app = Flask(__name__)

# Load your trained model (and, if you have one, a preprocessing pipeline)
with open("model/XGBRegressor.pkl", "rb") as f:
    model = pickle.load(f)

EXPECTED_KEYS = {
    "store_ID": int,
    "day_of_week": int,
    "date": str,
    "nb_customers_on_day": int,
    "open": int,
    "promotion": int,
    "state_holiday": str,
    "school_holiday": int,
}

def parse_and_validate(data):
    parsed = {}
    # 1) Check for missing keys
    for key, expected_type in EXPECTED_KEYS.items():
        if key not in data:
            raise ValueError(f"Missing required field: '{key}'")
        value = data[key]
        # 2) If expected int but got string, try to convert
        if expected_type is int:
            try:
                parsed[key] = int(value)
            except (ValueError, TypeError):
                raise ValueError(f"Field '{key}' must be an integer (got {value!r})")
        elif key == "date":
            # 3) Try multiple date formats
            try:
                dt = date_parser.parse(value, dayfirst=True)
                parsed[key] = dt
            except (ValueError, TypeError):
                raise ValueError(f"Field 'date' must be a valid date string (got {value!r})")
        else:
            # string fields
            parsed[key] = str(value)
    return parsed

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        inputs = parse_and_validate(payload)
        # Convert to DataFrame (model expects tabular)
        df = pd.DataFrame([{
            **{k: inputs[k] for k in EXPECTED_KEYS if k != "date"},
            # if your model wants numeric date parts instead:
            "year": inputs["date"].year,
            "month": inputs["date"].month,
            "day": inputs["date"].day,
        }])
        preds = model.predict(df)
        return jsonify({"prediction": preds.tolist()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # catch‐all for unexpected issues
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
