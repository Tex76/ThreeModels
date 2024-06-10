import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Load the dataset
data = pd.read_csv('Clean_Dataset.csv')

# Print column names to inspect
print("Column Names:", data.columns)

# Preprocess the data


def preprocess_data(data):
    # Encode categorical features
    label_encoder = LabelEncoder()
    categorical_features = ['airline', 'flight', 'source_city',
                            'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
    for feature in categorical_features:
        if feature in data.columns:
            data[feature] = label_encoder.fit_transform(data[feature])
        else:
            print(
                f"Warning: '{feature}' column not found. Skipping encoding for this feature.")

    # Drop missing values
    data.dropna(inplace=True)

    return data


data = preprocess_data(data)

# Split the data
X = data.drop('price', axis=1)
y = data['price']

# Keep the last 15 records for future prediction
future_data = data.iloc[-15:]
data = data.iloc[:-15]

# Split the remaining data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# Train and evaluate the models


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = mse
    return results


models = {
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

results = train_and_evaluate(models, X_train, y_train, X_test, y_test)

# Print the results
print("Model Performance (MSE):")
for model_name, mse in results.items():
    print(f"{model_name}: {mse}")

# Corrected K-fold cross-validation on training set
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    cv_results[name] = -cv_scores.mean()

# Print cross-validation results
print("\nK-Fold Cross-Validation (MSE):")
for model_name, mse in cv_results.items():
    print(f"{model_name}: {mse}")
