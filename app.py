import io
import re

import pandas as pd
import streamlit as st

from pdf_processing.extractor import extract_text_from_pdf
from services.risk_selector import select_top_risks
from services.knowledge_base import knowledge_base


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CACHED PDF EXTRACTION
# ============================================================

@st.cache_data(show_spinner=False)
def extract_contract_text(pdf_bytes):
    """
    Extract contract text once per uploaded PDF.

    Streamlit reruns the application whenever a widget changes.
    Caching prevents the PDF from being extracted repeatedly.
    """

    pdf_file = io.BytesIO(pdf_bytes)

    return extract_text_from_pdf(
        pdf_file
    )


# ============================================================
# CACHED SENTENCE SPLITTING
# ============================================================

@st.cache_data(show_spinner=False)
def split_contract_sentences(extracted_text):
    """
    Split extracted contract text into sentences.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        extracted_text
    )

    return tuple(
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    )


# ============================================================
# CACHED RISK ANALYSIS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=50
)
def analyze_contract(
    sentences,
    selected_model
):
    """
    Run the expensive risk-selection pipeline.

    Results are cached separately for each model.

    Therefore:

        PDF + SVM
        PDF + Logistic Regression
        PDF + Random Forest

    are treated as separate cached analyses.

    Switching back to a model that has already been run
    does not repeat the expensive risk-selection process.
    """

    return select_top_risks(
        list(sentences),
        top_k=20,
        selected_model=selected_model
    )


# ============================================================
# CACHED COMPLETE RISK LOOKUP
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=500
)
def get_complete_risk_information(risk_id):
    """
    Retrieve the complete risk record from the knowledge base.

    This includes the qualitative and quantitative information
    associated with the Risk ID.
    """

    try:

        complete = knowledge_base.get_complete_risk(
            risk_id
        )

        if complete:

            return dict(
                complete
            )

    except Exception:

        pass

    # Fallback to quantitative information if a complete
    # record is not available.
    try:

        quantitative = (
            knowledge_base.get_quantitative_risk(
                risk_id
            )
        )

        if quantitative:

            return dict(
                quantitative
            )

    except Exception:

        pass

    return {}


# ============================================================
# HEADER
# ============================================================

st.title(
    "📄 Contract Risk Analyzer"
)

st.write(
    "Upload a contract PDF to identify and prioritize "
    "the most relevant contractual risks using machine "
    "learning and a structured risk knowledge base."
)


# ============================================================
# MODEL SELECTION
# ============================================================

model_options = {
    "Logistic Regression":
        "Logistic Regression",

    "Support Vector Machine":
        "SVM",

    "Random Forest":
        "Random Forest",

    "Naive Bayes":
        "Naive Bayes",

    "Decision Tree":
        "Decision Tree",

    "XGBoost":
        "XGBoost",

    "LightGBM":
        "LightGBM",

    "CatBoost":
        "CatBoost",

    "MLP / DNN":
        "MLP"
}


selected_model_label = st.selectbox(
    "Choose Prediction Model",
    list(
        model_options.keys()
    ),
    index=1
)


selected_model = model_options[
    selected_model_label
]


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Contract PDF",
    type=["pdf"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # Read bytes once.
    pdf_bytes = uploaded_file.getvalue()

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # ========================================================
    # PDF EXTRACTION
    # ========================================================

    with st.spinner(
        "Extracting contract text..."
    ):

        extracted_text = extract_contract_text(
            pdf_bytes
        )

    if not extracted_text.strip():

        st.error(
            "No readable text was found in the PDF."
        )

        st.stop()

    # ========================================================
    # SENTENCE SPLITTING
    # ========================================================

    sentences = split_contract_sentences(
        extracted_text
    )

    st.subheader(
        f"📑 Contract Sentences: {len(sentences)}"
    )

    # ========================================================
    # RISK ANALYSIS
    # ========================================================

    with st.spinner(
        f"Analysing contract using {selected_model_label}..."
    ):

        results = analyze_contract(
            sentences,
            selected_model
        )

    if not results:

        st.warning(
            "No sufficiently relevant risks were identified "
            "in this contract."
        )

        st.stop()

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # COMPLETE RISK INFORMATION
    # ========================================================

    for index, row in results_df.iterrows():

        risk_id = row.get(
            "Risk ID"
        )

        if not risk_id:

            continue

        complete_risk = (
            get_complete_risk_information(
                risk_id
            )
        )

        if not complete_risk:

            continue

        # ----------------------------------------------------
        # Add every available field from the risk library.
        # Existing ML/contract fields are preserved.
        # ----------------------------------------------------

        for column, value in complete_risk.items():

            if (
                column not in results_df.columns
                or
                pd.isna(
                    results_df.loc[
                        index,
                        column
                    ]
                )
            ):

                results_df.loc[
                    index,
                    column
                ] = value

    # ========================================================
    # NORMALIZE COLUMN NAMES
    # ========================================================

    # Probability

    if "Probability" not in results_df.columns:

        if "Probability (L/M/H)" in results_df.columns:

            results_df["Probability"] = (
                results_df[
                    "Probability (L/M/H)"
                ]
            )

        else:

            results_df["Probability"] = None

    # Impact

    if "Impact" not in results_df.columns:

        if (
            "Impact (Cost / Time / Quality / HSE)"
            in results_df.columns
        ):

            results_df["Impact"] = (
                results_df[
                    "Impact (Cost / Time / Quality / HSE)"
                ]
            )

        else:

            results_df["Impact"] = None

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    numeric_columns = [

        "Gross Exposure (% TCV)",

        "Gross Exposure (EUR)",

        "Probability Factor",

        "Expected Monetary Value (EUR)",

        "Mitigation Effectiveness",

        "Residual Exposure (EUR)",

        "Residual Exposure (% TCV)",

        "Risk Relevance Score"
    ]

    for column in numeric_columns:

        if column in results_df.columns:

            results_df[column] = pd.to_numeric(
                results_df[column],
                errors="coerce"
            )

    # ========================================================
    # RANK BY RESIDUAL EXPOSURE
    # ========================================================

    if "Residual Exposure (EUR)" in results_df.columns:

        results_df = results_df.sort_values(
            by="Residual Exposure (EUR)",
            ascending=False,
            na_position="last"
        ).reset_index(
            drop=True
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader(
        "📊 Risk Analysis Summary"
    )

    col1, col2, col3, col4 = st.columns(4)

    # --------------------------------------------------------
    # Risks
    # --------------------------------------------------------

    with col1:

        st.metric(
            "Risks Identified",
            len(results_df)
        )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    with col2:

        category_count = (
            results_df[
                "Risk Category"
            ]
            .dropna()
            .nunique()
        )

        st.metric(
            "Risk Categories",
            category_count
        )

    # --------------------------------------------------------
    # EMV
    # --------------------------------------------------------

    with col3:

        total_emv = (
            results_df[
                "Expected Monetary Value (EUR)"
            ]
            .fillna(0)
            .sum()
        )

        st.metric(
            "Total EMV",
            f"€{total_emv:,.2f}"
        )

    # --------------------------------------------------------
    # Residual Exposure
    # --------------------------------------------------------

    with col4:

        total_residual = (
            results_df[
                "Residual Exposure (EUR)"
            ]
            .fillna(0)
            .sum()
        )

        st.metric(
            "Residual Exposure",
            f"€{total_residual:,.2f}"
        )

    st.caption(
        f"Prediction model used: {selected_model_label}"
    )

    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    st.subheader(
        "📈 Risk Distribution"
    )

    chart_col1, chart_col2 = st.columns(2)

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    with chart_col1:

        st.write(
            "Risk Category"
        )

        category_counts = (
            results_df[
                "Risk Category"
            ]
            .fillna("Not Available")
            .value_counts()
        )

        st.bar_chart(
            category_counts
        )

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    with chart_col2:

        st.write(
            "Probability"
        )

        probability_counts = (
            results_df[
                "Probability"
            ]
            .fillna("Not Available")
            .value_counts()
        )

        st.bar_chart(
            probability_counts
        )

    chart_col3, chart_col4 = st.columns(2)

    # --------------------------------------------------------
    # IMPACT
    # --------------------------------------------------------

    with chart_col3:

        st.write(
            "Impact"
        )

        impact_counts = (
            results_df[
                "Impact"
            ]
            .fillna("Not Available")
            .value_counts()
        )

        st.bar_chart(
            impact_counts
        )

    # --------------------------------------------------------
    # RESPONSIBLE PARTY
    # --------------------------------------------------------

    with chart_col4:

        st.write(
            "Responsible Party"
        )

        party_counts = (
            results_df[
                "Responsible Party"
            ]
            .fillna("Not Available")
            .value_counts()
        )

        st.bar_chart(
            party_counts
        )

    # ========================================================
    # TOP RISKS
    # ========================================================

    st.subheader(
        f"🚨 Top {len(results_df)} Contract Risks"
    )

    # ========================================================
    # DISPLAY COLUMNS
    # ========================================================

    # Source Basis intentionally excluded.

    display_columns = [

        "Contract Sentence",

        "Sentence Type",

        "Risk ID",

        "Risk Statement",

        "Risk Category",

        "ML Predicted Category",

        "Category Confidence (%)",

        "Responsible Party",

        "ML Predicted Responsible Party",

        "Party Confidence (%)",

        "Probability",

        "Impact",

        "Project Type",

        "Energy Segment",

        "Geographic Context",

        "Lifecycle Phase",

        "Gross Exposure (% TCV)",

        "Gross Exposure (EUR)",

        "Probability Factor",

        "Expected Monetary Value (EUR)",

        "Mitigation Effectiveness",

        "Residual Exposure (EUR)",

        "Residual Exposure (% TCV)",

        "Quantification Basis",

        "Risk Owner",

        "Early Warning Indicators",

        "Mitigation Actions",

        "Risk Relevance Score"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in results_df.columns
    ]

    # ========================================================
    # TABLE
    # ========================================================

    st.dataframe(
        results_df[
            display_columns
        ],
        use_container_width=True,
        height=700
    )

    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    csv = (
        results_df[
            display_columns
        ]
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        label="📥 Download Risk Analysis CSV",
        data=csv,
        file_name="contract_top_20_risks.csv",
        mime="text/csv"
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.caption(
        "This system provides automated risk-identification "
        "and prioritization support based on contract text "
        "and a structured risk knowledge base. Results should "
        "be reviewed by qualified contract, legal, commercial, "
        "or project-risk professionals."
    )

else:

    st.info(
        "Please upload a contract PDF to begin analysis."
    )