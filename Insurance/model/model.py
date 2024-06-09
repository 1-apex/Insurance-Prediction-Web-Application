import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv('Medicalpremium.csv')

# Ensure there are no null values
if df.isnull().sum().sum() != 0:
    raise ValueError("Dataset contains null values")

# Select features and target
X = df.drop('PremiumPrice', axis=1)
y = df['PremiumPrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Save the model and scaler to files
joblib.dump(model, 'insurance_premium_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

