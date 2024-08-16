import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path='data/livestock_disease_dataset.csv'):
    # Load data
    df = pd.read_csv(file_path)
    
    # Clean up column names by stripping leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Drop any unnamed columns if they are empty
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Clean the Production column: extract the average of the range
    def extract_average_production(value):
        if isinstance(value, float):
            return value  # Already a float, return as is
        
        # Handle 'X to Y' format
        if ' to ' in value:
            lower, upper = value.split(' to ')
        # Handle 'X-Y' format (hyphen)
        elif '-' in value:
            parts = value.split('-')
            if len(parts) == 2:  # Only process if there are exactly two parts
                lower, upper = parts
            else:
                # Handle unexpected cases with multiple hyphens
                return None  # or handle according to your needs
        else:
            # Handle simple numbers or other formats
            try:
                return float(value.strip().split(' ')[0])
            except ValueError:
                return None  # or handle the error as needed
        
        try:
            lower = float(lower.strip())
            upper = float(upper.split(' ')[0].strip())  # Ignore units if present
            return (lower + upper) / 2
        except ValueError:
            return None  # Handle conversion errors gracefully

    df['Production'] = df['Production'].apply(extract_average_production)

    # Initialize label encoders
    le_animal = LabelEncoder()
    le_breed = LabelEncoder()
    le_product = LabelEncoder()
    le_disease = LabelEncoder()
    le_symptom1 = LabelEncoder()
    le_symptom2 = LabelEncoder()
    le_symptom3 = LabelEncoder()

    # Encode categorical features
    df['animal_encoded'] = le_animal.fit_transform(df['Animal'])
    df['breed_encoded'] = le_breed.fit_transform(df['Breed'])
    df['product_encoded'] = le_product.fit_transform(df['Products'])
    df['symptom1_encoded'] = le_symptom1.fit_transform(df['Symptom1'])
    df['symptom2_encoded'] = le_symptom2.fit_transform(df['Symptom2'])
    df['symptom3_encoded'] = le_symptom3.fit_transform(df['Symptom3'])
    df['disease_encoded'] = le_disease.fit_transform(df['Disease'])

    # Define features and target
    X = df[['animal_encoded', 'breed_encoded', 'product_encoded', 'Production', 
            'symptom1_encoded', 'symptom2_encoded', 'symptom3_encoded']]
    y = df['disease_encoded']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le_animal, le_breed, le_product, le_disease
