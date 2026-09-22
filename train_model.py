import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load Dataset from CSV File
print("Loading dataset from CSV file...")
df = pd.read_csv('housing.csv')

# 2. Feature Selection & Data Preprocessing
X = df[['Area_sqft', 'Bedrooms', 'Age_years']]
y = df['Price_Lakhs']

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Building & Training
print("Training the Random Forest Regressor model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 5. Save the Trained Model
joblib.dump(model, 'house_price_model.pkl')
print("\nModel successfully saved as 'house_price_model.pkl'!")