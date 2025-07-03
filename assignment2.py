import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Function to Generate synthetic dataset (5000 records)
np.random.seed(42)
data = {
    'Speed_Limit': np.random.randint(30, 120, 5000),
    'Road_Condition': np.random.choice(['Dry', 'Wet', 'Icy'], 5000, p=[0.7, 0.2, 0.1]),
    'Weather': np.random.choice(['Clear', 'Rain', 'Fog', 'Snow'], 5000, p=[0.6, 0.2, 0.1, 0.1]),
    'Light_Condition': np.random.choice(['Daylight', 'Dusk', 'Dark'], 5000, p=[0.6, 0.2, 0.2]),
    'Vehicle_Age': np.random.randint(0, 30, 5000),
    'Severity': np.random.uniform(10, 100, 5000)  # Target variable
}

# Apply realistic patterns to severity
for i in range(5000):
    data['Severity'][i] += (
        data['Speed_Limit'][i] * 0.3 +
        (20 if data['Road_Condition'][i] == 'Icy' else 10 if data['Road_Condition'][i] == 'Wet' else 0) +
        (15 if data['Weather'][i] in ['Rain', 'Snow'] else 0) +
        (25 if data['Light_Condition'][i] == 'Dark' else 0) -
        data['Vehicle_Age'][i] * 0.5
    )
    
# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing: One-hot encode categorical variables
df = pd.get_dummies(df, columns=['Road_Condition', 'Weather', 'Light_Condition'])

# Split data
X = df.drop('Severity', axis=1)
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Model RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, 'accident_severity_model.pkl')

# Prediction example
hypothetical_data = pd.DataFrame({
    'Speed_Limit': [80],
    'Vehicle_Age': [5],
    'Road_Condition_Dry': [0],
    'Road_Condition_Icy': [1],
    'Road_Condition_Wet': [0],
    'Weather_Clear': [0],
    'Weather_Fog': [0],
    'Weather_Rain': [1],
    'Weather_Snow': [0],
    'Light_Condition_Dark': [1],
    'Light_Condition_Daylight': [0],
    'Light_Condition_Dusk': [0]
})

loaded_model = joblib.load('accident_severity_model.pkl')
predicted_severity = loaded_model.predict(hypothetical_data)[0]
print(f"\nPredicted Severity: {predicted_severity:.1f}/100")

# Feature importance visualization
coefficients = pd.Series(model.coef_, index=X.columns)
plt.figure(figsize=(10, 6))
coefficients.sort_values().plot(kind='barh')
plt.title('Feature Impact on Accident Severity')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('feature_importance.png')