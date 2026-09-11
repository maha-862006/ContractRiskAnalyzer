import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier

from preprocessing.text_preprocessor import preprocess_sentence


# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_excel(
    "data/contract_risk_statements_with_responsible_party.xlsx"
)

print(f"\nDataset Loaded: {len(df)} records")


# ----------------------------
# Preprocess Text
# ----------------------------
df["Clean_Text"] = df["Risk Statement"].apply(
    preprocess_sentence
)

print("Text Preprocessed")


# ----------------------------
# Features and Labels
# ----------------------------
X = df["Clean_Text"]
y = df["Category"]


# ----------------------------
# Encode Labels
# ----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nCategories:")
for number, category in enumerate(label_encoder.classes_):
    print(f"{number}: {category}")


# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")


# ----------------------------
# Build Pipeline
# ----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=False
    ))
])


# ----------------------------
# Train
# ----------------------------
model.fit(X_train, y_train)

print("\nTraining Complete")


# ----------------------------
# Evaluate
# ----------------------------
predictions = model.predict(X_test)

# CatBoost may return shape (n, 1)
predictions = predictions.ravel().astype(int)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print(f"Accuracy: {accuracy:.2%}")
print("==============================")


print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    )
)


print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ----------------------------
# Probability Test
# ----------------------------
probabilities = model.predict_proba(X_test)

print("\nProbability Test Successful")
print(f"Probability shape: {probabilities.shape}")


# ----------------------------
# Save Model
# ----------------------------
joblib.dump(
    model,
    "saved_models/catboost_category_model.pkl"
)

joblib.dump(
    label_encoder,
    "saved_models/catboost_label_encoder.pkl"
)

print("\nCatBoost Model Saved Successfully!")