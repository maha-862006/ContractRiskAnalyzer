import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBase:

    def __init__(self):

        # =====================================================
        # Original 184-row knowledge base
        # =====================================================

        self.df = pd.read_excel(
            "data/Starter_Risk_Knowledge_Base.xlsx"
        )

        # =====================================================
        # Qualitative 3000-risk library
        # =====================================================

        self.qualitative_df = pd.read_excel(
            "data/Global_Energy_Project_Risk_Library_3000.xlsx"
        )

        # =====================================================
        # Quantitative 3000-risk library
        # Risk Library is the required sheet
        # =====================================================

        self.quantitative_df = pd.read_excel(
            "data/Global_Energy_Project_Risk_Library_3000_Quantified.xlsx",
            sheet_name="Risk Library"
        )

        # =====================================================
        # Clean column names
        # =====================================================

        self.qualitative_df.columns = (
            self.qualitative_df.columns
            .str.strip()
        )

        self.quantitative_df.columns = (
            self.quantitative_df.columns
            .str.strip()
        )

        # =====================================================
        # Ensure Risk Statement is text
        # =====================================================

        self.qualitative_df["Risk Statement"] = (
            self.qualitative_df["Risk Statement"]
            .fillna("")
            .astype(str)
        )

        self.quantitative_df["Risk Statement"] = (
            self.quantitative_df["Risk Statement"]
            .fillna("")
            .astype(str)
        )

        # =====================================================
        # Create CORE risk statements
        #
        # Example:
        #
        # "For a Solar PV project in Europe, there is a risk
        # during Construction that supplier failure may cause
        # delays..."
        #
        # becomes mainly:
        #
        # "supplier failure may cause delays..."
        #
        # This prevents project/location/lifecycle information
        # from dominating similarity matching.
        # =====================================================

        self.qualitative_df["Core Risk Statement"] = (
            self.qualitative_df["Risk Statement"]
            .apply(self._extract_core_risk)
        )

        # =====================================================
        # TF-IDF index
        # =====================================================

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )

        self.risk_vectors = (
            self.vectorizer.fit_transform(
                self.qualitative_df[
                    "Core Risk Statement"
                ]
            )
        )

        print(
            f"Qualitative Risk Library Loaded: "
            f"{len(self.qualitative_df)} risks"
        )

        print(
            f"Quantitative Risk Library Loaded: "
            f"{len(self.quantitative_df)} risks"
        )

    # =========================================================
    # Extract core risk from library statement
    # =========================================================

    def _extract_core_risk(
        self,
        risk_statement
    ):

        text = str(
            risk_statement
        )

        # Remove everything before:
        # "there is a risk during ..."
        match = re.search(
            r"\bthere is a risk during\b.*?\bthat\b",
            text,
            flags=re.IGNORECASE
        )

        if match:
            text = text[
                match.end():
            ]

        # Remove generic ending
        text = re.sub(
            r"\bmay adversely affect project objectives\b.*$",
            "",
            text,
            flags=re.IGNORECASE
        )

        # Remove excessive whitespace
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    # =========================================================
    # Original knowledge-base method
    # =========================================================

    def get_risk_information(
        self,
        category
    ):

        matches = self.df[
            self.df["Risk_Category"]
            .astype(str)
            .str.lower()
            == str(category).lower()
        ]

        if matches.empty:
            return None

        return matches.iloc[0].to_dict()

    # =========================================================
    # Find similar risks
    # =========================================================

    def find_similar_risks(
        self,
        sentence,
        top_k=20,
        similarity_threshold=0.10
    ):

        if (
            not sentence
            or not str(sentence).strip()
        ):
            return []

        # -----------------------------------------------------
        # Clean query
        # -----------------------------------------------------

        query = re.sub(
            r"\s+",
            " ",
            str(sentence)
        ).strip()

        # -----------------------------------------------------
        # Convert contract clause to TF-IDF vector
        # -----------------------------------------------------

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        # -----------------------------------------------------
        # Calculate cosine similarity
        # -----------------------------------------------------

        similarities = cosine_similarity(
            query_vector,
            self.risk_vectors
        )[0]

        # -----------------------------------------------------
        # Rank highest similarity first
        # -----------------------------------------------------

        ranked_indices = (
            similarities.argsort()[::-1]
        )

        results = []

        # -----------------------------------------------------
        # Collect results
        # -----------------------------------------------------

        for index in ranked_indices:

            similarity_score = float(
                similarities[index]
            )

            if (
                similarity_score
                < similarity_threshold
            ):
                break

            risk = (
                self.qualitative_df
                .iloc[index]
                .to_dict()
            )

            risk["Similarity Score"] = round(
                similarity_score,
                4
            )

            results.append(
                risk
            )

            if len(results) >= top_k:
                break

        return results

    # =========================================================
    # Get quantitative risk by Risk ID
    # =========================================================

    def get_quantitative_risk(
        self,
        risk_id
    ):

        if risk_id is None:
            return None

        matches = self.quantitative_df[
            self.quantitative_df["Risk ID"]
            .astype(str)
            == str(risk_id)
        ]

        if matches.empty:
            return None

        return matches.iloc[0].to_dict()

    # =========================================================
    # Get complete qualitative + quantitative risk
    # =========================================================

    def get_complete_risk(
        self,
        risk_id
    ):

        if risk_id is None:
            return None

        # -----------------------------------------------------
        # Qualitative
        # -----------------------------------------------------

        qualitative_matches = (
            self.qualitative_df[
                self.qualitative_df["Risk ID"]
                .astype(str)
                == str(risk_id)
            ]
        )

        if qualitative_matches.empty:
            return None

        qualitative = (
            qualitative_matches
            .iloc[0]
            .to_dict()
        )

        # -----------------------------------------------------
        # Quantitative
        # -----------------------------------------------------

        quantitative_matches = (
            self.quantitative_df[
                self.quantitative_df["Risk ID"]
                .astype(str)
                == str(risk_id)
            ]
        )

        if not quantitative_matches.empty:

            quantitative = (
                quantitative_matches
                .iloc[0]
                .to_dict()
            )

            # Add quantitative fields that are
            # not already present.
            for key, value in (
                quantitative.items()
            ):

                if key not in qualitative:
                    qualitative[key] = value

        return qualitative


# =============================================================
# Shared Knowledge Base instance
# =============================================================

knowledge_base = KnowledgeBase()