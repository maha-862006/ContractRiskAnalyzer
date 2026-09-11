import re
import pandas as pd

from services.knowledge_base import knowledge_base
from models.predictor import predict_sentence


# ============================================================
# RISK LIBRARY SEARCH WRAPPER
# ============================================================

def find_similar_risks(text, top_k=30):
    return knowledge_base.find_similar_risks(
        text,
        top_k=top_k
    )


# ============================================================
# 1. RISK CONCEPT VOCABULARY
# ============================================================

RISK_CONCEPTS = {

    "payment": [
        "payment",
        "payments",
        "invoice",
        "invoices",
        "payable",
        "paid",
        "price",
        "pricing",
        "contract price",
        "advance",
        "retention"
    ],

    "delay": [
        "delay",
        "delayed",
        "late",
        "completion",
        "schedule",
        "extension of time",
        "liquidated damages"
    ],

    "liability": [
        "liability",
        "liable",
        "damages",
        "loss",
        "indemnity",
        "indemnification"
    ],

    "termination": [
        "termination",
        "terminate",
        "default",
        "breach",
        "insolvency"
    ],

    "scope_change": [
        "change order",
        "variation",
        "variation order",
        "change",
        "scope change",
        "scope"
    ],

    "dispute": [
        "dispute",
        "claim",
        "claims",
        "arbitration",
        "litigation",
        "court"
    ],

    "insurance": [
        "insurance",
        "insured",
        "insurer",
        "coverage",
        "policy"
    ],

    "performance": [
        "performance",
        "performance test",
        "performance guarantee",
        "acceptance",
        "service level",
        "standard"
    ],

    "quality": [
        "quality",
        "defect",
        "defective",
        "nonconformance",
        "non-compliance",
        "inspection",
        "testing"
    ],

    "safety": [
        "safety",
        "unsafe",
        "accident",
        "hazard",
        "hse",
        "health"
    ],

    "procurement": [
        "supplier",
        "supplier failure",
        "vendor",
        "procurement",
        "equipment",
        "material",
        "manufacturer",
        "delivery"
    ],

    "site": [
        "site condition",
        "site conditions",
        "site",
        "access",
        "land",
        "right of way",
        "ground conditions"
    ],

    "regulatory": [
        "permit",
        "permits",
        "approval",
        "authority",
        "environmental",
        "regulatory",
        "regulation",
        "compliance"
    ],

    "security": [
        "security",
        "confidentiality",
        "cyber",
        "cybersecurity",
        "data protection",
        "personal data"
    ],

    "warranty": [
        "warranty",
        "warranties",
        "remedy"
    ],

    "force_majeure": [
        "force majeure",
        "natural disaster",
        "earthquake",
        "flood",
        "war",
        "pandemic"
    ],

    "suspension": [
        "suspension",
        "suspend",
        "suspended"
    ],

    "tax": [
        "tax",
        "taxes",
        "duties",
        "withholding",
        "vat"
    ],

    "financial_security": [
        "performance security",
        "performance guarantee",
        "security deposit",
        "retention money",
        "bond"
    ],

    "environmental": [
        "pollution",
        "waste",
        "environmental impact",
        "environmental damage",
        "emissions"
    ],

    "interfaces": [
        "interface",
        "interfaces",
        "boundary",
        "coordination",
        "coordination failure"
    ],

    "governance": [
        "approval delay",
        "decision delay",
        "decision making",
        "governance",
        "authority",
        "instruction"
    ],
}


# ============================================================
# 2. BASIC TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_text(text):

    text = clean_text(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 3. DETECT RISK CONCEPTS
# ============================================================

def get_concepts(text):

    lower = str(text).lower()

    concepts = set()

    for concept, keywords in RISK_CONCEPTS.items():

        for keyword in keywords:

            if keyword in lower:

                concepts.add(
                    concept
                )

                break

    return concepts


# ============================================================
# 4. RISK SIGNAL
# ============================================================

def risk_signal_score(text):

    lower = str(text).lower()

    concepts = get_concepts(
        text
    )

    consequence_terms = [
        "failure",
        "fail",
        "delay",
        "breach",
        "default",
        "liable",
        "liability",
        "damages",
        "penalty",
        "terminate",
        "termination",
        "claim",
        "dispute",
        "loss",
        "cost",
        "compensation",
        "remedy",
        "suspend",
        "suspension",
    ]

    consequence_count = sum(
        1
        for term in consequence_terms
        if term in lower
    )

    score = 0.0

    if len(concepts) >= 3:

        score += 0.55

    elif len(concepts) == 2:

        score += 0.40

    elif len(concepts) == 1:

        score += 0.25

    if consequence_count >= 2:

        score += 0.30

    elif consequence_count == 1:

        score += 0.15

    return min(
        score,
        1.0
    )


# ============================================================
# 5. SENTENCE TYPE
# ============================================================

def classify_sentence_type(sentence):

    text = clean_text(
        sentence
    )

    if not text:

        return "Other"

    lower = text.lower()

    if re.search(
        r"\bmeans\b|"
        r"\bshall mean\b|"
        r"\bdefined as\b",
        lower
    ):

        return "Definition"

    if re.search(
        r"\bshall be entitled\b|"
        r"\bshall have the right\b|"
        r"\bright to\b|"
        r"\bmay claim\b|"
        r"\bmay be entitled\b",
        lower
    ):

        return "Right"

    signal = risk_signal_score(
        text
    )

    if signal >= 0.40:

        return "Risk"

    obligation = any(
        word in lower
        for word in [
            "shall",
            "must",
            "required to",
            "responsible for",
            "undertake",
            "ensure",
        ]
    )

    consequence = any(
        word in lower
        for word in [
            "failure",
            "delay",
            "breach",
            "default",
            "liable",
            "damages",
            "penalty",
            "terminate",
            "claim",
            "loss",
        ]
    )

    if obligation and consequence:

        return "Obligation"

    return "Other"


# ============================================================
# 6. REMOVE PDF JUNK
# ============================================================

def is_irrelevant_contract_text(text):

    text = clean_text(
        text
    )

    if not text:

        return True

    words = text.split()

    if len(words) < 10:

        return True

    lower = text.lower()

    if re.fullmatch(
        r"[\d\s./()\-]+",
        text
    ):

        return True

    unwanted = [
        "table of contents",
        "list of exhibits",
        "list of figures",
        "list of tables",
        "this page intentionally left blank",
        "document control",
    ]

    if any(
        phrase in lower
        for phrase in unwanted
    ):

        return True

    if re.match(
        r"^\d+(\.\d+)*\s+.*\s+\d+$",
        text
    ):

        return True

    return False


# ============================================================
# 7. CATEGORY COMPATIBILITY
# ============================================================

CATEGORY_MAPPING = {

    "Financial": [
        "financial",
        "commercial",
        "payment",
        "cost",
        "pricing",
        "tax",
        "insurance",
        "insurance & security",
    ],

    "Legal": [
        "legal",
        "contractual",
        "regulatory",
        "regulatory & change in law",
        "claims",
        "dispute",
        "governance & decision making",
    ],

    "Operational": [
        "operational",
        "technical",
        "construction",
        "engineering",
        "quality",
        "quality & defects",
        "safety",
        "hse",
        "logistics",
        "procurement",
        "procurement & supply chain",
        "plant & equipment",
        "interfaces",
        "environmental",
        "weather & natural hazards",
    ],

    "Administrative": [
        "administrative",
        "documentation",
        "permitting",
        "permits & approvals",
        "regulatory",
        "contractual",
    ],

    "Strategic": [
        "strategic",
        "commercial",
        "stakeholder",
        "market",
        "procurement",
        "logistics",
        "governance & decision making",
    ],
}


def category_compatibility(
    predicted,
    library
):

    if not predicted or not library:

        return 0.0

    predicted = str(
        predicted
    ).lower()

    library = str(
        library
    ).lower()

    allowed = CATEGORY_MAPPING.get(
        predicted,
        []
    )

    if library in allowed:

        return 1.0

    for item in allowed:

        if item in library:

            return 0.70

    return 0.0


# ============================================================
# 8. PARTY COMPATIBILITY
# ============================================================

def party_compatibility(
    predicted,
    library
):

    if not predicted or not library:

        return 0.0

    predicted = str(
        predicted
    ).lower()

    library = str(
        library
    ).lower()

    if predicted == library:

        return 1.0

    if library == "shared":

        return 0.50

    return 0.0


# ============================================================
# 9. LIBRARY RISK CORE
#
# Remove project type, geography and lifecycle information
# from the library risk statement because these are metadata
# rather than the core risk meaning.
# ============================================================

def extract_library_core(text):

    text = clean_text(
        text
    )

    match = re.search(
        r"\bthere is a risk\b.*?\bthat\b",
        text,
        flags=re.IGNORECASE
    )

    if match:

        text = text[
            match.end():
        ]

    stop_patterns = [
        r",?\s*combined with\b",
        r",?\s*which may\b",
        r",?\s*resulting in\b",
        r",?\s*leading to\b",
        r",?\s*may adversely affect\b",
    ]

    for pattern in stop_patterns:

        text = re.split(
            pattern,
            text,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0]

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip(
        " ,.;:"
    )

    return text


# ============================================================
# 10. RISK FAMILY
#
# Create a specific family key from the normalized core risk
# wording.
# ============================================================

def risk_family(text):

    core = extract_library_core(
        text
    )

    normalized = normalize_text(
        core
    )

    generic_words = {
        "risk",
        "project",
        "may",
        "affect",
        "objectives",
        "combined",
        "during",
        "potential",
        "possible",
        "failure",
        "weak",
        "poor",
        "lack",
        "inadequate",
        "adverse",
        "impact",
        "result",
        "resulting",
        "leading",
    }

    words = [
        word
        for word in normalized.split()
        if word not in generic_words
        and len(word) > 3
    ]

    words = words[:20]

    return tuple(words)


# ============================================================
# 11. CONCEPT MATCH
# ============================================================

def concept_match(
    contract_text,
    library_text
):

    contract = get_concepts(
        contract_text
    )

    library = get_concepts(
        extract_library_core(
            library_text
        )
    )

    if not contract or not library:

        return 0.0

    overlap = (
        contract & library
    )

    if len(overlap) >= 3:

        return 1.0

    if len(overlap) == 2:

        return 0.80

    if len(overlap) == 1:

        return 0.55

    return 0.0


# ============================================================
# 12. MAIN RISK SELECTOR
# ============================================================

def select_top_risks(
    sentences,
    top_k=20,
    selected_model="SVM"
):

    candidates = []

    # ========================================================
    # 1. GENERATE CANDIDATES
    # ========================================================

    for sentence in sentences:

        sentence = str(
            sentence
        ).strip()

        if not sentence:

            continue

        if is_irrelevant_contract_text(
            sentence
        ):

            continue

        sentence_type = classify_sentence_type(
            sentence
        )

        # Only analyze text that can represent a contractual
        # risk, obligation or right.
        if sentence_type not in {
            "Risk",
            "Obligation",
            "Right"
        }:

            continue

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        try:

            prediction = predict_sentence(
                sentence,
                selected_model
            )

        except Exception:

            continue

        predicted_category = prediction.get(
            "category",
            ""
        )

        category_confidence = float(
            prediction.get(
                "category_confidence",
                0
            )
        )

        predicted_party = prediction.get(
            "party",
            ""
        )

        party_confidence = float(
            prediction.get(
                "party_confidence",
                0
            )
        )

        # ----------------------------------------------------
        # Knowledge-base matching
        # ----------------------------------------------------

        try:

            matches = find_similar_risks(
                sentence,
                top_k=30
            )

        except Exception:

            continue

        for match in matches:

            risk_id = match.get(
                "Risk ID",
                ""
            )

            if not risk_id:

                continue

            risk_statement = str(
                match.get(
                    "Risk Statement",
                    ""
                )
            )

            if not risk_statement:

                continue

            risk_category = str(
                match.get(
                    "Risk Category",
                    ""
                )
            )

            similarity = float(
                match.get(
                    "similarity",
                    0
                )
            )

            # ------------------------------------------------
            # Concept compatibility
            # ------------------------------------------------

            concepts = concept_match(
                sentence,
                risk_statement
            )

            if concepts <= 0:

                continue

            # ------------------------------------------------
            # Category compatibility
            # ------------------------------------------------

            category_score = category_compatibility(
                predicted_category,
                risk_category
            )

            # ------------------------------------------------
            # Party compatibility
            # ------------------------------------------------

            party_score = party_compatibility(
                predicted_party,
                match.get(
                    "Employer / Contractor / Shared allocation",
                    match.get(
                        "Responsible Party",
                        ""
                    )
                )
            )

            # ------------------------------------------------
            # Contract-risk signal
            # ------------------------------------------------

            signal = risk_signal_score(
                sentence
            )

            # ------------------------------------------------
            # Overall relevance
            # ------------------------------------------------

            relevance = (
                similarity * 0.20
                + concepts * 0.40
                + category_score * 0.20
                + party_score * 0.05
                + signal * 0.15
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Use a lower threshold so that legitimate risks
            # from less dominant categories are not eliminated.
            # ------------------------------------------------

            if relevance < 0.25:

                continue

            # ------------------------------------------------
            # Risk-family keys
            # ------------------------------------------------

            family = risk_family(
                risk_statement
            )

            contract_family = risk_family(
                sentence
            )

            risk_core = extract_library_core(
                risk_statement
            )

            risk_text_key = normalize_text(
                risk_core
            )

            # ------------------------------------------------
            # Store candidate
            # ------------------------------------------------

            candidates.append({

                "Contract Sentence":
                    sentence,

                "Sentence Type":
                    sentence_type,

                "Risk ID":
                    risk_id,

                "Risk Statement":
                    risk_core,

                "Risk Category":
                    risk_category,

                "ML Predicted Category":
                    predicted_category,

                "Category Confidence (%)":
                    round(
                        category_confidence * 100,
                        2
                    ),

                "Responsible Party":
                    match.get(
                        "Employer / Contractor / Shared allocation",
                        match.get(
                            "Responsible Party",
                            ""
                        )
                    ),

                "ML Predicted Responsible Party":
                    predicted_party,

                "Party Confidence (%)":
                    round(
                        party_confidence * 100,
                        2
                    ),

                "Similarity":
                    similarity,

                "Risk Relevance Score":
                    relevance,

                "_risk_family":
                    family,

                "_contract_family":
                    contract_family,

                "_risk_text_key":
                    risk_text_key,

                "_category_score":
                    category_score,

                "_signal":
                    signal
            })

    # ========================================================
    # 2. REMOVE EXACT DUPLICATES
    #
    # Same Risk ID + same contract sentence should only appear
    # once. Keep the strongest occurrence.
    # ========================================================

    unique = {}

    for item in candidates:

        key = (
            item["Risk ID"],
            item["Contract Sentence"]
        )

        existing = unique.get(
            key
        )

        if (
            existing is None
            or
            item["Risk Relevance Score"]
            >
            existing["Risk Relevance Score"]
        ):

            unique[key] = item

    candidates = list(
        unique.values()
    )

    # ========================================================
    # 3. DEDUPLICATE BY ACTUAL RISK MEANING
    #
    # Different Risk IDs can represent exactly the same
    # underlying risk. Do not show the same risk repeatedly.
    #
    # We use:
    #
    #     Risk Category
    #     +
    #     normalized core Risk Statement
    #
    # as the actual-risk identity.
    # ========================================================

    best_by_risk = {}

    for candidate in candidates:

        risk_key = (
            candidate["Risk Category"],
            candidate["_risk_text_key"]
        )

        existing = best_by_risk.get(
            risk_key
        )

        if (
            existing is None
            or
            candidate["Risk Relevance Score"]
            >
            existing["Risk Relevance Score"]
        ):

            best_by_risk[risk_key] = candidate

    candidates = list(
        best_by_risk.values()
    )

    # ========================================================
    # 4. SORT CANDIDATES
    # ========================================================

    candidates.sort(
        key=lambda x:
            x["Risk Relevance Score"],
        reverse=True
    )

    # ========================================================
    # 5. DIVERSITY-AWARE SELECTION
    #
    # Pass 1:
    # Select the strongest distinct risk from every available
    # category.
    #
    # This prevents one category such as Change & Scope from
    # consuming the entire Top-20 list.
    #
    # Pass 2:
    # Fill remaining positions with the strongest remaining
    # distinct risks.
    #
    # The system returns UP TO top_k risks. It never creates
    # artificial risks merely to reach 20.
    # ========================================================

    selected = []

    selected_risk_keys = set()
    selected_categories = set()

    # --------------------------------------------------------
    # Pass 1: category diversity
    # --------------------------------------------------------

    for candidate in candidates:

        if len(selected) >= top_k:

            break

        risk_key = (
            candidate["Risk Category"],
            candidate["_risk_text_key"]
        )

        category = candidate[
            "Risk Category"
        ]

        if risk_key in selected_risk_keys:

            continue

        if category in selected_categories:

            continue

        selected.append(
            candidate
        )

        selected_risk_keys.add(
            risk_key
        )

        selected_categories.add(
            category
        )

    # --------------------------------------------------------
    # Pass 2: strongest remaining distinct risks
    # --------------------------------------------------------

    for candidate in candidates:

        if len(selected) >= top_k:

            break

        risk_key = (
            candidate["Risk Category"],
            candidate["_risk_text_key"]
        )

        if risk_key in selected_risk_keys:

            continue

        selected.append(
            candidate
        )

        selected_risk_keys.add(
            risk_key
        )

    # ========================================================
    # 6. REMOVE INTERNAL FIELDS
    # ========================================================

    for item in selected:

        item.pop(
            "_risk_family",
            None
        )

        item.pop(
            "_contract_family",
            None
        )

        item.pop(
            "_risk_text_key",
            None
        )

        item.pop(
            "_category_score",
            None
        )

        item.pop(
            "_signal",
            None
        )

        item.pop(
            "Similarity",
            None
        )

    # ========================================================
    # 7. FINAL SORT
    # ========================================================

    selected.sort(
        key=lambda x:
            x["Risk Relevance Score"],
        reverse=True
    )

    return selected[:top_k]