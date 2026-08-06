import pandas as pd


class KnowledgeBase:

    def __init__(self):

        self.df = pd.read_excel(
            "data/Starter_Risk_Knowledge_Base.xlsx"
        )

    def get_risk_information(self, category):

        matches = self.df[
            self.df["Risk_Category"].str.lower()
            == category.lower()
        ]

        if matches.empty:
            return None

        return matches.iloc[0].to_dict()


knowledge_base = KnowledgeBase()