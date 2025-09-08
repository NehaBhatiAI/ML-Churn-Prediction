# Import required libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
X['Gender'] = X['Gender'].map({'Female': 0, 'Male': 1})
X = pd.get_dummies(X, drop_first=True)

# Save the feature names for consistency (important for later prediction)
feature_names = X.columns.tolist()

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)
print("Balanced target distribution:\n", y.value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions & evaluation
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and scaler
with open("NB_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature names
with open("features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model, scaler, and features saved successfully!")


