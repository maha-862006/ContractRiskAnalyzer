import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing.text_preprocessor import preprocess_sentence


# Load dataset
df = pd.read_excel(
    "data/contract_risk_statements_with_responsible_party.xlsx"
)

print(f"\nDataset Loaded: {len(df)} records")

# Preprocess
df["Clean_Text"] = df["Risk Statement"].apply(preprocess_sentence)

X = df["Clean_Text"]
y = df["Responsible Party"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")

# Build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report\n")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, predictions))

# Save model
joblib.dump(model, "saved_models/party_model.pkl")

print("\nParty Model Saved Successfully!")