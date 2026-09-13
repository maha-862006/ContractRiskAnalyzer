import os
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "contract_risk_statements_with_responsible_party.xlsx"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "model_cross_validation.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel(DATA_PATH)

X = df["Risk Statement"].astype(str)
y_text = df["Category"].astype(str)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

print("=" * 75)
print("CONTRACT RISK ANALYZER — MODEL EVALUATION")
print("=" * 75)

print(f"\nTotal rows: {len(df)}")
print(f"Unique Risk Statements: {X.nunique()}")

duplicate_rate = 1 - (X.nunique() / len(X))

print(f"Duplicate rate: {duplicate_rate:.2%}")

print("\nIMPORTANT:")
print(
    "The dataset contains repeated Risk Statements. "
    "Therefore, ordinary random K-fold cross-validation can place "
    "the same statement in both training and validation folds."
)

print(
    "The resulting scores are reported as DUPLICATE-AFFECTED "
    "BENCHMARK RESULTS and must not be interpreted as "
    "generalization performance on unseen contracts."
)


# ============================================================
# MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        verbosity=-1,
        random_state=42
    ),

    "CatBoost": CatBoostClassifier(
        iterations=100,
        depth=5,
        learning_rate=0.1,
        verbose=False,
        random_seed=42
    ),

    "MLP/DNN": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        random_state=42
    )
}


# ============================================================
# TF-IDF
# ============================================================

def create_pipeline(model):

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                lowercase=True,
                sublinear_tf=True
            )
        ),
        ("model", model)
    ])


# ============================================================
# SCORING
# ============================================================

scoring = {
    "accuracy": make_scorer(accuracy_score),

    "precision_macro": make_scorer(
        precision_score,
        average="macro",
        zero_division=0
    ),

    "recall_macro": make_scorer(
        recall_score,
        average="macro",
        zero_division=0
    ),

    "f1_macro": make_scorer(
        f1_score,
        average="macro",
        zero_division=0
    )
}


# ============================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


for name, model in models.items():

    print(f"\nEvaluating: {name}")

    pipeline = create_pipeline(model)

    try:

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1
        )

        result = {
            "Model": name,

            "Accuracy": np.mean(
                scores["test_accuracy"]
            ),

            "Precision Macro": np.mean(
                scores["test_precision_macro"]
            ),

            "Recall Macro": np.mean(
                scores["test_recall_macro"]
            ),

            "F1 Macro": np.mean(
                scores["test_f1_macro"]
            ),

            "CV Type": "5-fold Stratified CV",
            "Dataset Unique Statements": int(X.nunique()),
            "Dataset Duplicate Rate": duplicate_rate,
            "Generalization Valid": False
        }

        results.append(result)

        print(
            f"  Accuracy : {result['Accuracy']:.4f}"
        )

        print(
            f"  Precision: {result['Precision Macro']:.4f}"
        )

        print(
            f"  Recall   : {result['Recall Macro']:.4f}"
        )

        print(
            f"  F1       : {result['F1 Macro']:.4f}"
        )

    except Exception as e:

        print(f"  FAILED: {e}")

        results.append({
            "Model": name,
            "Accuracy": np.nan,
            "Precision Macro": np.nan,
            "Recall Macro": np.nan,
            "F1 Macro": np.nan,
            "CV Type": "5-fold Stratified CV",
            "Dataset Unique Statements": int(X.nunique()),
            "Dataset Duplicate Rate": duplicate_rate,
            "Generalization Valid": False
        })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["F1 Macro", "Accuracy"],
    ascending=False,
    na_position="last"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("EVALUATION COMPLETE")
print("=" * 75)

print("\nResults:")
print(
    results_df[
        [
            "Model",
            "Accuracy",
            "Precision Macro",
            "Recall Macro",
            "F1 Macro"
        ]
    ].to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nInterpretation:")
print(
    "These scores are duplicate-affected benchmark results. "
    "They are NOT evidence that the models generalize to unseen "
    "contract clauses."
)

print(
    "\nThe scores will therefore NOT be used by themselves "
    "to declare a scientifically proven 'best 3' models."
)

print("=" * 75)