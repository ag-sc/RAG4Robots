import pandas as pd
from sentence_transformers import SentenceTransformer

from RAG4Robots.src.manager import RAGManager, EMBEDDING_MODEL
from RAG4Robots.src.utils.enums import ResourceType


class RAGPlanManager(RAGManager):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            super().__init__([(ResourceType.PLANS, 1.0)])
            self._initialized = True

    @staticmethod
    def add_new_plan(new_p: str):
        plan_df = RAGPlanManager._instance.get_databases()[0].get_db_as_df()

        vectors = SentenceTransformer(EMBEDDING_MODEL).encode([new_p], show_progress_bar=True)
        dim_cols = [f"dim_{i}" for i in range(vectors.shape[1])]
        new_row = pd.DataFrame(
            [[new_p] + vectors[0].tolist()],
            columns=["text"] + dim_cols
        )
        plan_df = pd.concat([plan_df, new_row], ignore_index=True)

        RAGPlanManager._instance.get_databases()[0].replace_database(plan_df)
