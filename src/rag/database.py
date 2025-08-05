from os import path
from pathlib import Path
from typing import List, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.rag import vectorizer
from src.utils.enums import ResourceType
from src.utils.sim_calc import calculate_similarity


class RAGDatabase:
    def __init__(self, db_type: ResourceType, data_usage=1.0,
                 embed_mod='sentence-transformers/all-MiniLM-L6-v2') -> None:
        self._database = pd.DataFrame()
        self._database_name = db_type.type
        self._database_type = db_type
        self._database_path = Path(path.join(path.dirname(__file__), "..", "..", "vector_dbs/", db_type.file_name))
        self._data_usage_percentage = data_usage
        self._embedding_model = SentenceTransformer(embed_mod)

        first_file_path = self._database_path.with_name(f"{self._database_path.stem}_0{self._database_path.suffix}")
        if not first_file_path.is_file():
            vectorizer.create_new_from_file(self._database_type, self._embedding_model, self._database_path)
        self._database = self.load_split_embeddings(self._data_usage_percentage)
        print(f"Created a RAG vector database for: {self._database_name} ({len(self._database)} entries at {self._data_usage_percentage * 100}% usage)")

    def load_split_embeddings(self, usage=1.0) -> pd.DataFrame:
        pattern = f"{self._database_path.stem}_*.csv"
        files = sorted(self._database_path.parent.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No split CSV files found for pattern: {pattern}")
        df_list = [pd.read_csv(file) for file in files]
        data = pd.concat(df_list, ignore_index=True)
        return data.sample(round(usage * len(data)))

    def query_current_db(self, query: str, hits_to_return=3) -> List[Any]:
        embed_query = self._embedding_model.encode(query)
        similarities = []
        for _, row in self._database.iterrows():
            chunk = row['text']
            chunk_embedding = row.iloc[1:].to_numpy().astype(np.float32)
            sim = calculate_similarity(embed_query, chunk_embedding)
            similarities.append((chunk, sim))
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:hits_to_return]

    def get_current_db(self) -> str:
        return self._database_name

    def get_current_db_type(self) -> ResourceType:
        return self._database_type

    def get_path_current_db(self) -> Path:
        return self._database_path

    def get_embedding_model(self) -> SentenceTransformer:
        return self._embedding_model
