import joblib
from preprocessing.text_preprocessor import preprocess_sentence

# Load models
logistic_model = joblib.load("saved_models/category_model.pkl")
svm_model = joblib.load("saved_models/svm_category_model.pkl")
random_forest_model = joblib.load("saved_models/random_forest_category_model.pkl")
party_model = joblib.load("saved_models/party_model.pkl")


def predict_sentence(sentence, selected_model="Logistic Regression"):

    cleaned = preprocess_sentence(sentence)

    if selected_model == "Support Vector Machine":
        model = svm_model

    elif selected_model == "Random Forest":
        model = random_forest_model

    else:
        model = logistic_model

    category = model.predict([cleaned])[0]
    category_confidence = max(model.predict_proba([cleaned])[0])

    party = party_model.predict([cleaned])[0]
    party_confidence = max(party_model.predict_proba([cleaned])[0])

    return {
        "sentence": sentence,
        "category": category,
        "category_confidence": round(category_confidence * 100, 2),
        "responsible_party": party,
        "party_confidence": round(party_confidence * 100, 2),
    }