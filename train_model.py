import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Categorize scores: Low / Medium / High
def categorize(score):
    if score <= 50: return "Low"
    elif score <= 75: return "Medium"
    else: return "High"

df["Math"] = df["math score"].apply(categorize)
df["Reading"] = df["reading score"].apply(categorize)
df["Writing"] = df["writing score"].apply(categorize)

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Features and targets
X = df.drop(["math score","reading score","writing score","Math","Reading","Writing"], axis=1)
y = df[["Math","Reading","Writing"]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Multi-output Random Forest
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/student_perf_model.pkl")
print("✅ Multi-score model trained and saved successfully!")
