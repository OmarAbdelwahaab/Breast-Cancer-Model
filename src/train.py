import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from preprocess import clean_and_preprocess

def train_model():
    # 1. Get preprocessed data
    X, y = clean_and_preprocess('data/Breast_Cancer.csv')
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Random Forest
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    predictions = model.predict(X_test)
    print("\n--- Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
    print(classification_report(y_test, predictions))
    
    # 5. Save the trained model
    joblib.dump(model, 'models/random_forest_model.pkl')
    print("Model saved to models/random_forest_model.pkl")

if __name__ == "__main__":
    train_model()
