import pandas as pd
import joblib
import numpy as np

def make_prediction(sample_data):
    # Load saved elements
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/random_forest_model.pkl')
    
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([sample_data])
    
    # For a simple script, we assume values are pre-mapped numericals or match training shapes
    # Scale features
    scaled_input = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)
    
    return prediction[0], max(probability[0])

if __name__ == "__main__":
    # Example raw numerical/encoded array layout matching your features
    # (Replace these dummy values with actual encoded row details from your data)
    dummy_patient = {
        "Age": 50, "Race": 0, "Marital Status": 1, "T Stage ": 1, "N Stage": 0,
        "6th Stage": 0, "differentiate": 1, "Grade": 2, "A Stage": 0, "Tumor Size": 4,
        "Estrogen Status": 1, "Progesterone Status": 1, "Regional Node Examined": 24,
        "Reginol Node Positive": 1, "Survival Months": 60
    }
    
    result, confidence = make_prediction(dummy_patient)
    print(f"Prediction Class Outcome: {result} with {confidence*100:.2f}% confidence.")
