from os import path
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.enums import ResourceType
from src.rag import vectorizer


class RagDBManager:
    def __init__(self, db_name: str, db_type: ResourceType, embed_mod='sentence-transformers/all-MiniLM-L6-v2') -> None:
        self._database = pd.DataFrame()
        self._database_name = db_name
        self._database_type = db_type
        self._database_path = Path(path.join(path.dirname(__file__), "..", "..", "vector_dbs/", db_type.file_name))
        self._embedding_model = SentenceTransformer(embed_mod)

        if not self._database_path.is_file():
            vectorizer.create_new_from_file(self._database_type, self._embedding_model, self._database_path)
        self._database = pd.read_csv(self._database_path)

    def query_current_db(self, query: str) -> str:
        pass

    def get_current_db(self) -> str:
        return self._database_name

    def get_current_db_type(self) -> ResourceType:
        return self._database_type

    def get_path_current_db(self) -> Path:
        return self._database_path

    def get_embedding_model(self) -> SentenceTransformer:
        return self._embedding_model
