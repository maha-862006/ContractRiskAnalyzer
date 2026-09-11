import joblib


# Category Models
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


# Responsible Party Model
party_model = joblib.load(
    "saved_models/party_model.pkl"
)


def get_category_model(model_name):

    if model_name == "Logistic Regression":
        return logistic_model

    elif model_name == "Support Vector Machine":
        return svm_model

    elif model_name == "Random Forest":
        return random_forest_model

    elif model_name == "Naive Bayes":
        return naive_bayes_model

    elif model_name == "Decision Tree":
        return decision_tree_model

    elif model_name == "XGBoost":
        return xgboost_model

    elif model_name == "LightGBM":
        return lightgbm_model

    elif model_name == "CatBoost":
        return catboost_model

    elif model_name == "MLP / DNN":
        return mlp_model

    else:
        return logistic_model