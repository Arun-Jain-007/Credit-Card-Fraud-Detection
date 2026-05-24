import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
file_path = 'dataset/creditcard.csv'
df = pd.read_csv(file_path)

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Report
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print('Model and scaler saved successfully!')