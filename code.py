# Import required libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
df = pd.read_csv("Churn.csv")

# Basic checks
print(df.head())
print("Null values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print("Target distribution:\n", df['Exited'].value_counts())

# Features & target
X = df.drop(columns=['Exited', 'CustomerId'])
y = df['Exited']

# Encode categorical variables
X['Gender'].replace({'Female': 0, 'Male': 1}, inplace=True)
X = pd.get_dummies(X, drop_first=True)

# Handle class imbalance with SMOTE
sm = SMOTE()
X, y = sm.fit_resample(X, y)
print("Balanced target distribution:\n", y.value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions & evaluation
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model and scaler
with open("NB_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")
