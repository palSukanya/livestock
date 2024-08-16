import os
import joblib
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model
from src.predict import evaluate_model

# Load and preprocess data
X_train, X_test, y_train, y_test, le_animal, le_breed, le_product, le_disease = load_and_preprocess_data('data/livestock_dataset.csv')

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
accuracy = evaluate_model(model, X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Create 'models' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the trained model and label encoders
joblib.dump(model, 'models/livestock_disease_predictor.pkl')
joblib.dump(le_animal, 'models/label_encoder_animal.pkl')
joblib.dump(le_breed, 'models/label_encoder_breed.pkl')
joblib.dump(le_product, 'models/label_encoder_product.pkl')
joblib.dump(le_disease, 'models/label_encoder_disease.pkl')
