# Customer Churn Prediction
# By Mthunzi Ntombela

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv('churn_data.csv')

# 2. Clean data - convert Yes/No to 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

# 3. Features and Target
X = df[['MonthlyCharges', 'Tenure', 'Contract']]
y = df['Churn']

# 4. Split - 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Predict & Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 7. Predict for a new customer
# Example: MonthlyCharges=80, Tenure=2 months, Month-to-month contract (0)
new_customer = [[80, 2, 0]]
result = model.predict(new_customer)
print(f"\nNew Customer Will Churn? {'Yes - Offer Discount!' if result[0]==1 else 'No - Safe'}")
