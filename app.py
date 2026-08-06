import streamlit as st
import pandas as pd

from pdf_processing.extractor import extract_text_from_pdf
from preprocessing.sentence_splitter import split_into_sentences
from preprocessing.sentence_filter import is_valid_sentence
from services.knowledge_base import knowledge_base
from models.predictor import predict_sentence

st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Contract Risk Analyzer")
st.write("Upload a construction contract PDF for AI-based risk analysis.")

selected_model = st.selectbox(
    "Choose Prediction Model",
    [
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest"
    ]
)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    extracted_text = extract_text_from_pdf(uploaded_file)
    sentences = split_into_sentences(extracted_text)

    st.subheader(f"Total Sentences Found: {len(sentences)}")

    results = []

    for sentence in sentences:

        if not is_valid_sentence(sentence):
            continue

        prediction = predict_sentence(
            sentence,
            selected_model
        )

        risk_info = knowledge_base.get_risk_information(
            prediction["category"]
        )

        if risk_info is None:
            risk_info = {}

        results.append({
            "Sentence": sentence,
            "Risk Category": prediction["category"],
            "Category Confidence (%)": prediction["category_confidence"],
            "Responsible Party": prediction["responsible_party"],
            "Party Confidence (%)": prediction["party_confidence"],
            "Description": risk_info.get("Description", ""),
            "Potential Impact": risk_info.get("Potential_Impact", ""),
            "Mitigation": risk_info.get("Mitigation", "")
        })

    results_df = pd.DataFrame(results)

    st.subheader("📊 Analysis Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Analysed Sentences", len(results_df))

    with col2:
        st.metric(
            "Unique Risk Categories",
            results_df["Risk Category"].nunique()
        )

    st.subheader("📈 Risk Category Distribution")
    st.bar_chart(results_df["Risk Category"].value_counts())

    st.subheader("👥 Responsible Party Distribution")
    st.bar_chart(results_df["Responsible Party"].value_counts())

    st.subheader("📋 Analysis Results")
    st.dataframe(results_df, use_container_width=True)

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Results as CSV",
        csv,
        "contract_analysis.csv",
        "text/csv"
    )

else:
    st.info("Please upload a PDF to begin analysis.")