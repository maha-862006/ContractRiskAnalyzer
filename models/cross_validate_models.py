import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "data/contract_risk_statements_with_responsible_party.xlsx"
OUTPUT = "output/model_cross_validation.csv"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL CROSS-VALIDATION")
print("=" * 70)

df = pd.read_excel(DATASET)

df = df.dropna(subset=["Risk Statement", "Category"])

X = df["Risk Statement"].astype(str)
y = df["Category"].astype(str)

print(f"Dataset size: {len(df)}")
print(f"Categories: {sorted(y.unique())}")
print()


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Encoded categories:")
for i, category in enumerate(label_encoder.classes_):
    print(f"  {i}: {category}")

print()


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "SVM": SVC(
        probability=True,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Naive Bayes": MultinomialNB(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbosity=-1
    ),

    "CatBoost": CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        verbose=False,
        random_seed=RANDOM_STATE
    ),

    "MLP/DNN": MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
}


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


results = []


for name, model in models.items():

    print("-" * 70)
    print(f"Testing: {name}")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1
            )
        ),
        ("model", model)
    ])

    start_time = time.time()

    scores = cross_validate(
        pipeline,
        X,
        y_encoded,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro"
        },
        n_jobs=1,
        return_train_score=False
    )

    elapsed = time.time() - start_time

    accuracy_mean = scores["test_accuracy"].mean()
    accuracy_std = scores["test_accuracy"].std()

    precision_mean = scores["test_precision"].mean()
    recall_mean = scores["test_recall"].mean()
    f1_mean = scores["test_f1"].mean()

    results.append({
        "Model": name,
        "Accuracy": accuracy_mean,
        "Accuracy Std": accuracy_std,
        "Precision": precision_mean,
        "Recall": recall_mean,
        "F1": f1_mean,
        "Training/Evaluation Time (s)": elapsed
    })

    print(f"Accuracy : {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    print(f"Precision: {precision_mean:.4f}")
    print(f"Recall   : {recall_mean:.4f}")
    print(f"F1       : {f1_mean:.4f}")
    print(f"Time     : {elapsed:.2f}s")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["F1", "Accuracy"],
    ascending=False
).reset_index(drop=True)


results_df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# DISPLAY FINAL TABLE
# ============================================================

print()
print("=" * 70)
print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 70)

display_df = results_df.copy()

for column in [
    "Accuracy",
    "Accuracy Std",
    "Precision",
    "Recall",
    "F1"
]:
    display_df[column] = display_df[column].map(
        lambda x: f"{x * 100:.2f}%"
    )

display_df["Training/Evaluation Time (s)"] = (
    display_df["Training/Evaluation Time (s)"]
    .map(lambda x: f"{x:.2f}")
)

print(
    display_df.to_string(index=False)
)

print()
print("=" * 70)
print(f"Results saved to: {OUTPUT}")
print("=" * 70)