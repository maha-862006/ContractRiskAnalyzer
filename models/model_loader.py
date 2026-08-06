import joblib


# Load all category models once
logistic_model = joblib.load("saved_models/category_model.pkl")

svm_model = joblib.load("saved_models/svm_category_model.pkl")

random_forest_model = joblib.load(
    "saved_models/random_forest_category_model.pkl"
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

    else:
        return logistic_model