import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing.text_preprocessor import preprocess_sentence


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_excel(
    "data/contract_risk_statements_with_responsible_party.xlsx"
)

print(f"\nDataset Loaded: {len(df)} records")


# -------------------------------------------------
# Preprocess
# -------------------------------------------------
df["Clean_Text"] = df["Risk Statement"].apply(
    preprocess_sentence
)


# -------------------------------------------------
# Same Test Split Used For All Models
# -------------------------------------------------
X = df["Clean_Text"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")


# -------------------------------------------------
# Load Models
# -------------------------------------------------
models = {

    "Logistic Regression":
        joblib.load(
            "saved_models/category_model.pkl"
        ),

    "Support Vector Machine":
        joblib.load(
            "saved_models/svm_category_model.pkl"
        ),

    "Random Forest":
        joblib.load(
            "saved_models/random_forest_category_model.pkl"
        ),

    "Naive Bayes":
        joblib.load(
            "saved_models/naive_bayes_category_model.pkl"
        ),

    "Decision Tree":
        joblib.load(
            "saved_models/decision_tree_category_model.pkl"
        ),

    "XGBoost":
        joblib.load(
            "saved_models/xgboost_category_model.pkl"
        ),

    "LightGBM":
        joblib.load(
            "saved_models/lightgbm_category_model.pkl"
        ),

    "CatBoost":
        joblib.load(
            "saved_models/catboost_category_model.pkl"
        ),

    "MLP / DNN":
        joblib.load(
            "saved_models/mlp_category_model.pkl"
        )
}


# -------------------------------------------------
# Label Encoders
# -------------------------------------------------
xgboost_encoder = joblib.load(
    "saved_models/xgboost_label_encoder.pkl"
)

lightgbm_encoder = joblib.load(
    "saved_models/lightgbm_label_encoder.pkl"
)

catboost_encoder = joblib.load(
    "saved_models/catboost_label_encoder.pkl"
)


# -------------------------------------------------
# Evaluate Models
# -------------------------------------------------
results = []


for model_name, model in models.items():

    print(f"\nEvaluating: {model_name}")

    predictions = model.predict(X_test)

    # Convert encoded predictions back to category names
    if model_name == "XGBoost":

        predictions = xgboost_encoder.inverse_transform(
            predictions.astype(int)
        )

    elif model_name == "LightGBM":

        predictions = lightgbm_encoder.inverse_transform(
            predictions.astype(int)
        )

    elif model_name == "CatBoost":

        predictions = catboost_encoder.inverse_transform(
            predictions.astype(int)
        )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })


# -------------------------------------------------
# Create Comparison Table
# -------------------------------------------------
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)


# -------------------------------------------------
# Display Results
# -------------------------------------------------
print("\n")
print("=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)


# -------------------------------------------------
# Save Results
# -------------------------------------------------
results_df.to_csv(
    "output/model_comparison.csv",
    index=False
)

print("\nModel comparison saved to:")
print("output/model_comparison.csv")