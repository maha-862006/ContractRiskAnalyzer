import joblib
from preprocessing.text_preprocessor import preprocess_sentence


# Load models
logistic_model = joblib.load(
    "saved_models/category_model.pkl"
)

svm_model = joblib.load(
    "saved_models/svm_category_model.pkl"
)

random_forest_model = joblib.load(
    "saved_models/random_forest_category_model.pkl"
)

naive_bayes_model = joblib.load(
    "saved_models/naive_bayes_category_model.pkl"
)

decision_tree_model = joblib.load(
    "saved_models/decision_tree_category_model.pkl"
)

xgboost_model = joblib.load(
    "saved_models/xgboost_category_model.pkl"
)

lightgbm_model = joblib.load(
    "saved_models/lightgbm_category_model.pkl"
)

catboost_model = joblib.load(
    "saved_models/catboost_category_model.pkl"
)

mlp_model = joblib.load(
    "saved_models/mlp_category_model.pkl"
)

party_model = joblib.load(
    "saved_models/party_model.pkl"
)


def predict_sentence(sentence, selected_model="Logistic Regression"):

    cleaned = preprocess_sentence(sentence)

    if selected_model == "Support Vector Machine":
        model = svm_model

    elif selected_model == "Random Forest":
        model = random_forest_model

    elif selected_model == "Naive Bayes":
        model = naive_bayes_model

    elif selected_model == "Decision Tree":
        model = decision_tree_model

    elif selected_model == "XGBoost":
        model = xgboost_model

    elif selected_model == "LightGBM":
        model = lightgbm_model

    elif selected_model == "CatBoost":
        model = catboost_model

    elif selected_model == "MLP / DNN":
        model = mlp_model

    else:
        model = logistic_model

    category = model.predict([cleaned])[0]

    category_confidence = max(
        model.predict_proba([cleaned])[0]
    )

    party = party_model.predict([cleaned])[0]

    party_confidence = max(
        party_model.predict_proba([cleaned])[0]
    )

    return {
        "sentence": sentence,
        "category": category,
        "category_confidence": round(
            category_confidence * 100,
            2
        ),
        "responsible_party": party,
        "party_confidence": round(
            party_confidence * 100,
            2
        ),
    }