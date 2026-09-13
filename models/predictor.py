import joblib
import numpy as np

from preprocessing.text_preprocessor import preprocess_sentence


# ============================================================
# LOAD MODELS
# ============================================================

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


# ============================================================
# LABEL ENCODERS FOR TREE-BASED MODELS
# ============================================================

xgboost_label_encoder = joblib.load(
    "saved_models/xgboost_label_encoder.pkl"
)

lightgbm_label_encoder = joblib.load(
    "saved_models/lightgbm_label_encoder.pkl"
)

catboost_label_encoder = joblib.load(
    "saved_models/catboost_label_encoder.pkl"
)


# ============================================================
# STANDARD CATEGORY NAMES
# ============================================================

CATEGORY_NAMES = [
    "Administrative",
    "Financial",
    "Legal",
    "Operational",
    "Strategic"
]


# ============================================================
# HYBRID MODEL CONFIGURATION
# ============================================================

HYBRID_MODELS = [
    "Support Vector Machine",
    "Logistic Regression",
    "Naive Bayes",
]


# ============================================================
# MODEL SELECTION
# ============================================================

def get_category_model(
    selected_model="Logistic Regression"
):

    if selected_model in {
        "Support Vector Machine",
        "SVM"
    }:
        return svm_model

    elif selected_model == "Random Forest":
        return random_forest_model

    elif selected_model == "Naive Bayes":
        return naive_bayes_model

    elif selected_model == "Decision Tree":
        return decision_tree_model

    elif selected_model == "XGBoost":
        return xgboost_model

    elif selected_model == "LightGBM":
        return lightgbm_model

    elif selected_model == "CatBoost":
        return catboost_model

    elif selected_model in {
        "MLP / DNN",
        "MLP/DNN",
        "MLP",
        "DNN"
    }:
        return mlp_model

    else:
        return logistic_model


# ============================================================
# CONFIDENCE NORMALIZATION
# ============================================================

def normalize_confidence(
    probability
):
    """
    Convert a model probability into a valid percentage.

    This represents the model's predicted probability/confidence
    for its selected class. It is NOT model accuracy.

    Output is always between 0 and 100.
    """

    try:

        value = np.asarray(
            probability
        ).reshape(-1)[0]

        value = float(
            value
        )

    except Exception:

        return 0.0

    if not np.isfinite(value):

        return 0.0

    value = max(
        0.0,
        min(
            1.0,
            value
        )
    )

    return round(
        value * 100.0,
        2
    )


# ============================================================
# PROBABILITY DISTRIBUTION NORMALIZATION
# ============================================================

def normalize_probability_distribution(
    probabilities,
    classes,
    selected_model
):
    """
    Convert a model probability vector into a clean dictionary.

    XGBoost, LightGBM and CatBoost use numeric class labels,
    so their probability columns are decoded through their
    corresponding label encoders.

    Values are returned as probabilities between 0 and 1.
    """

    try:

        probabilities = np.asarray(
            probabilities,
            dtype=float
        ).reshape(-1)

        classes = np.asarray(
            classes
        ).reshape(-1)

        result = {}

        for index, probability in enumerate(
            probabilities
        ):

            if index >= len(classes):
                break

            probability = float(
                probability
            )

            if not np.isfinite(
                probability
            ):

                probability = 0.0

            probability = max(
                0.0,
                min(
                    1.0,
                    probability
                )
            )

            category = normalize_category_label(
                classes[index],
                selected_model
            )

            result[category] = round(
                probability,
                6
            )

        return result

    except Exception:

        return {}


# ============================================================
# SCALAR VALUE NORMALIZATION
# ============================================================

def normalize_scalar(
    value
):
    """
    Convert numpy/list/tuple prediction outputs into
    a single scalar value.
    """

    try:

        if isinstance(
            value,
            np.ndarray
        ):

            flattened = value.reshape(
                -1
            )

            if len(flattened) == 0:

                return ""

            return flattened[0]

        if isinstance(
            value,
            (list, tuple)
        ):

            if len(value) == 0:

                return ""

            return normalize_scalar(
                value[0]
            )

    except Exception:

        pass

    return value


# ============================================================
# CATEGORY LABEL NORMALIZATION
# ============================================================

def normalize_category_label(
    value,
    selected_model
):
    """
    Convert model-specific category output into one of
    the five human-readable risk categories.

    XGBoost, LightGBM and CatBoost use numeric encoded
    category labels and therefore require inverse decoding.
    """

    value = normalize_scalar(
        value
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    if selected_model == "XGBoost":

        try:

            decoded = (
                xgboost_label_encoder
                .inverse_transform(
                    [int(value)]
                )[0]
            )

            return str(
                decoded
            )

        except Exception:

            return str(
                value
            )

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    if selected_model == "LightGBM":

        try:

            decoded = (
                lightgbm_label_encoder
                .inverse_transform(
                    [int(value)]
                )[0]
            )

            return str(
                decoded
            )

        except Exception:

            return str(
                value
            )

    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    if selected_model == "CatBoost":

        try:

            decoded = (
                catboost_label_encoder
                .inverse_transform(
                    [int(value)]
                )[0]
            )

            return str(
                decoded
            )

        except Exception:

            return str(
                value
            )

    # --------------------------------------------------------
    # Other models
    # --------------------------------------------------------

    return str(
        value
    )


# ============================================================
# PARTY LABEL NORMALIZATION
# ============================================================

def normalize_party_label(
    value
):
    """
    Convert numpy/list party predictions into a clean
    human-readable string.
    """

    value = normalize_scalar(
        value
    )

    return str(
        value
    )


# ============================================================
# PREPROCESS ONCE
# ============================================================

def preprocess_sentences(
    sentences
):
    """
    Convert all original contract sentences into the
    preprocessed representation used by the ML models.
    """

    return [
        preprocess_sentence(
            str(sentence)
        )
        for sentence in sentences
    ]


# ============================================================
# SINGLE MODEL PREDICTION USING PREPROCESSED TEXT
# ============================================================

def predict_preprocessed_sentences(
    original_sentences,
    cleaned_sentences,
    selected_model="Logistic Regression"
):
    """
    Run one category model and the party model using
    already-preprocessed sentences.
    """

    if not original_sentences:

        return []

    # ========================================================
    # CATEGORY MODEL
    # ========================================================

    category_model = get_category_model(
        selected_model
    )

    categories = category_model.predict(
        cleaned_sentences
    )

    category_probabilities = (
        category_model.predict_proba(
            cleaned_sentences
        )
    )

    category_classes = (
        category_model.classes_
    )

    category_confidences = (
        category_probabilities.max(
            axis=1
        )
    )

    # ========================================================
    # RESPONSIBLE PARTY MODEL
    # ========================================================

    parties = party_model.predict(
        cleaned_sentences
    )

    party_probabilities = (
        party_model.predict_proba(
            cleaned_sentences
        )
    )

    party_confidences = (
        party_probabilities.max(
            axis=1
        )
    )

    # ========================================================
    # BUILD RESULTS
    # ========================================================

    results = []

    for index, sentence in enumerate(
        original_sentences
    ):

        probability_distribution = (
            normalize_probability_distribution(
                category_probabilities[index],
                category_classes,
                selected_model
            )
        )

        results.append(
            {
                "sentence": sentence,

                "category": normalize_category_label(
                    categories[index],
                    selected_model
                ),

                "category_confidence": (
                    normalize_confidence(
                        category_confidences[index]
                    )
                ),

                "category_probabilities": (
                    probability_distribution
                ),

                "responsible_party": (
                    normalize_party_label(
                        parties[index]
                    )
                ),

                "party_confidence": (
                    normalize_confidence(
                        party_confidences[index]
                    )
                ),
            }
        )

    return results


# ============================================================
# BATCH PREDICTION — SINGLE MODEL
# ============================================================

def predict_sentences(
    sentences,
    selected_model="Logistic Regression"
):
    """
    Predict multiple sentences for one selected model.

    Text preprocessing is performed once for this call.
    """

    if not sentences:

        return []

    original_sentences = [
        str(sentence)
        for sentence in sentences
    ]

    cleaned_sentences = (
        preprocess_sentences(
            original_sentences
        )
    )

    return predict_preprocessed_sentences(
        original_sentences,
        cleaned_sentences,
        selected_model
    )


# ============================================================
# BATCH PREDICTION — ALL MODELS
# ============================================================

def predict_all_models(
    sentences,
    model_options
):
    """
    Run all category models using ONE shared preprocessing pass.

    The party model is evaluated only once because its
    prediction is independent of the category model.

    All category labels are normalized into human-readable
    category names.

    All confidence values are returned as percentages
    between 0 and 100.

    Full category probability distributions are also returned
    for hybrid/ensemble prediction.
    """

    if not sentences:

        return {}

    # ========================================================
    # ORIGINAL SENTENCES
    # ========================================================

    original_sentences = [
        str(sentence)
        for sentence in sentences
    ]

    # ========================================================
    # PREPROCESS ONLY ONCE
    # ========================================================

    cleaned_sentences = (
        preprocess_sentences(
            original_sentences
        )
    )

    # ========================================================
    # PARTY MODEL — ONLY ONCE
    # ========================================================

    parties = party_model.predict(
        cleaned_sentences
    )

    party_probabilities = (
        party_model.predict_proba(
            cleaned_sentences
        )
    )

    party_confidences = (
        party_probabilities.max(
            axis=1
        )
    )

    # ========================================================
    # ALL CATEGORY MODELS
    # ========================================================

    all_predictions = {}

    for model_label, model_name in (
        model_options.items()
    ):

        category_model = get_category_model(
            model_name
        )

        # ----------------------------------------------------
        # CATEGORY PREDICTIONS
        # ----------------------------------------------------

        categories = category_model.predict(
            cleaned_sentences
        )

        # ----------------------------------------------------
        # CATEGORY PROBABILITIES
        # ----------------------------------------------------

        category_probabilities = (
            category_model.predict_proba(
                cleaned_sentences
            )
        )

        category_classes = (
            category_model.classes_
        )

        category_confidences = (
            category_probabilities.max(
                axis=1
            )
        )

        # ----------------------------------------------------
        # BUILD MODEL RESULTS
        # ----------------------------------------------------

        model_results = []

        for index, sentence in enumerate(
            original_sentences
        ):

            probability_distribution = (
                normalize_probability_distribution(
                    category_probabilities[index],
                    category_classes,
                    model_name
                )
            )

            model_results.append(
                {
                    "sentence": sentence,

                    "category": (
                        normalize_category_label(
                            categories[index],
                            model_name
                        )
                    ),

                    "category_confidence": (
                        normalize_confidence(
                            category_confidences[index]
                        )
                    ),

                    "category_probabilities": (
                        probability_distribution
                    ),

                    "responsible_party": (
                        normalize_party_label(
                            parties[index]
                        )
                    ),

                    "party_confidence": (
                        normalize_confidence(
                            party_confidences[index]
                        )
                    ),
                }
            )

        all_predictions[
            model_label
        ] = model_results

    return all_predictions


# ============================================================
# HYBRID MODEL CONFIGURATION
# ============================================================

HYBRID_METHODS = [
    "Majority Vote",
    "Confidence Weighted Hybrid",
    "Rank Weighted Hybrid",
]

HYBRID_MODELS = [
    "Support Vector Machine",
    "Logistic Regression",
    "Naive Bayes",
]


# ============================================================
# HYBRID PREDICTION
# ============================================================

def predict_hybrid_preprocessed_sentences(
    original_sentences,
    cleaned_sentences,
    hybrid_method="Confidence Weighted Hybrid"
):
    """
    Run one of three hybrid prediction methods using:

        - Support Vector Machine
        - Logistic Regression
        - Naive Bayes

    Supported hybrid methods:

        1. Majority Vote
        2. Confidence Weighted Hybrid
        3. Rank Weighted Hybrid

    The hybrid decision strength is a voting-strength measure.
    It is NOT calibrated probability and NOT accuracy.
    """

    if not original_sentences:
        return []

    if hybrid_method not in HYBRID_METHODS:
        hybrid_method = "Confidence Weighted Hybrid"

    model_distributions = {}

    # ========================================================
    # RUN THE THREE COMPONENT MODELS
    # ========================================================

    for model_name in HYBRID_MODELS:

        category_model = get_category_model(
            model_name
        )

        probabilities = (
            category_model.predict_proba(
                cleaned_sentences
            )
        )

        classes = category_model.classes_

        distributions = []

        for row in probabilities:

            distribution = (
                normalize_probability_distribution(
                    row,
                    classes,
                    model_name
                )
            )

            distributions.append(
                distribution
            )

        model_distributions[
            model_name
        ] = distributions

    # ========================================================
    # PARTY MODEL — RUN ONCE
    # ========================================================

    parties = party_model.predict(
        cleaned_sentences
    )

    party_probabilities = (
        party_model.predict_proba(
            cleaned_sentences
        )
    )

    party_confidences = (
        party_probabilities.max(
            axis=1
        )
    )

    # ========================================================
    # HYBRID PREDICTIONS
    # ========================================================

    results = []

    for index, sentence in enumerate(
        original_sentences
    ):

        model_information = []

        for model_name in HYBRID_MODELS:

            distribution = (
                model_distributions[
                    model_name
                ][index]
            )

            predicted_category = max(
                distribution,
                key=distribution.get
            )

            model_confidence = float(
                distribution[
                    predicted_category
                ]
            )

            model_information.append(
                {
                    "model": model_name,
                    "category": predicted_category,
                    "confidence": model_confidence,
                }
            )

        # ====================================================
        # 1. MAJORITY VOTE
        # ====================================================

        if hybrid_method == "Majority Vote":

            vote_counts = {}

            for item in model_information:

                category = item["category"]

                vote_counts[category] = (
                    vote_counts.get(
                        category,
                        0
                    ) + 1
                )

            hybrid_category = max(
                vote_counts,
                key=vote_counts.get
            )

            hybrid_strength = (
                vote_counts[
                    hybrid_category
                ]
                / len(model_information)
            )

            combined = {
                category:
                    vote_counts.get(
                        category,
                        0
                    )
                    / len(model_information)

                for category in CATEGORY_NAMES
            }

        # ====================================================
        # 2. CONFIDENCE-WEIGHTED VOTE
        # ====================================================

        elif hybrid_method == (
            "Confidence Weighted Hybrid"
        ):

            scores = {
                category: 0.0
                for category in CATEGORY_NAMES
            }

            total_confidence = 0.0

            for item in model_information:

                category = item["category"]
                confidence = item["confidence"]

                scores[category] += confidence
                total_confidence += confidence

            hybrid_category = max(
                scores,
                key=scores.get
            )

            if total_confidence > 0:

                hybrid_strength = (
                    scores[
                        hybrid_category
                    ]
                    / total_confidence
                )

                combined = {
                    category:
                        scores[category]
                        / total_confidence

                    for category in CATEGORY_NAMES
                }

            else:

                hybrid_strength = 0.0

                combined = {
                    category: 0.0
                    for category in CATEGORY_NAMES
                }

        # ====================================================
        # 3. RANK-WEIGHTED VOTE
        # ====================================================

        else:

            ordered = sorted(
                model_information,
                key=lambda item:
                    item["confidence"],
                reverse=True
            )

            rank_weights = [3, 2, 1]

            scores = {
                category: 0.0
                for category in CATEGORY_NAMES
            }

            total_weight = 0.0

            for rank, item in enumerate(
                ordered
            ):

                weight = rank_weights[
                    min(
                        rank,
                        len(rank_weights) - 1
                    )
                ]

                category = item["category"]

                scores[category] += weight
                total_weight += weight

            hybrid_category = max(
                scores,
                key=scores.get
            )

            if total_weight > 0:

                hybrid_strength = (
                    scores[
                        hybrid_category
                    ]
                    / total_weight
                )

                combined = {
                    category:
                        scores[category]
                        / total_weight

                    for category in CATEGORY_NAMES
                }

            else:

                hybrid_strength = 0.0

                combined = {
                    category: 0.0
                    for category in CATEGORY_NAMES
                }

        # ====================================================
        # BUILD RESULT
        # ====================================================

        results.append(
            {
                "sentence": sentence,

                "category":
                    hybrid_category,

                # Kept under the old field name for
                # compatibility with risk_selector.py.
                # The UI will display this as
                # "Hybrid Decision Strength".
                "category_confidence":
                    normalize_confidence(
                        hybrid_strength
                    ),

                "category_probabilities":
                    {
                        category:
                            round(
                                probability,
                                6
                            )

                        for category,
                        probability
                        in combined.items()
                    },

                "hybrid_method":
                    hybrid_method,

                "hybrid_models":
                    list(HYBRID_MODELS),

                "responsible_party":
                    normalize_party_label(
                        parties[index]
                    ),

                "party_confidence":
                    normalize_confidence(
                        party_confidences[index]
                    ),
            }
        )

    return results


# ============================================================
# BATCH HYBRID PREDICTION
# ============================================================

def predict_hybrid(
    sentences,
    hybrid_method="Confidence Weighted Hybrid"
):
    """
    Run one of the three hybrid methods on multiple
    contract sentences.

    The three component models are:

        SVM
        Logistic Regression
        Naive Bayes
    """

    if not sentences:
        return []

    original_sentences = [
        str(sentence)
        for sentence in sentences
    ]

    cleaned_sentences = (
        preprocess_sentences(
            original_sentences
        )
    )

    return (
        predict_hybrid_preprocessed_sentences(
            original_sentences,
            cleaned_sentences,
            hybrid_method
        )
    )


# ============================================================
# SINGLE-SENTENCE COMPATIBILITY WRAPPER
# ============================================================

def predict_sentence(
    sentence,
    selected_model="Logistic Regression"
):
    """
    Backward-compatible single sentence prediction.

    Supports both the nine individual models and the
    Confidence Weighted Hybrid.
    """

    # ========================================================
    # CONFIDENCE-WEIGHTED HYBRID
    # ========================================================

    if selected_model in {
    "Majority Vote",
    "Confidence Weighted Hybrid",
    "Rank Weighted Hybrid",
    "Hybrid",
}:

        hybrid_method = selected_model

    if selected_model == "Hybrid":
        hybrid_method = "Confidence Weighted Hybrid"

    results = predict_hybrid(
        [sentence],
        hybrid_method
    )

    if results:
        return results[0]

    return {
        "sentence": sentence,
        "category": "",
        "category_confidence": 0.0,
        "category_probabilities": {},
        "hybrid_method": hybrid_method,
        "hybrid_models": list(HYBRID_MODELS),
        "responsible_party": "",
        "party_confidence": 0.0,
    }

    # ========================================================
    # NORMAL SINGLE MODEL
    # ========================================================

    results = predict_sentences(
        [sentence],
        selected_model
    )

    if not results:

        return {
            "sentence": sentence,
            "category": "",
            "category_confidence": 0.0,
            "category_probabilities": {},
            "responsible_party": "",
            "party_confidence": 0.0,
        }

    return results[0]