import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def clean_and_preprocess(data_path):
    df = pd.read_csv(data_path)
    
    # Clean up column names just in case
    df.columns = df.columns.str.strip()
    
    # Categorical columns from your notebook
    categorical_cols = [
        'Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage', 
        'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status'
    ]
    
    # Encode categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # Encode target variable ('Status')
    if 'Status' in df.columns:
        df['Status'] = le.fit_transform(df['Status'])
        
    # Separate features and target
    X = df.drop(columns=['Status'])
    y = df['Status']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save the scaler for later use in predictions
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_scaled_df, y

if __name__ == "__main__":
    X, y = clean_and_preprocess('data/Breast_Cancer.csv')
    print("Data preprocessed successfully! Shapes:", X.shape, y.shape)
