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
        self._database_path = Path(self.get_db_file_path(db_type))
        self._embedding_model = SentenceTransformer(embed_mod)

        if self._database_path.is_file():
            self._database = pd.read_csv(self._database_path)
        else:
            self._database = vectorizer.create_new_from_file(self._database_type, self._embedding_model)
            self._database.to_csv(self._database_path, index=False)

    @staticmethod
    def get_db_file_path(db_type: ResourceType) -> str:
        file_ending = 'csv'
        base_path = path.join(path.dirname(__file__), "..", "..", "vector_dbs/")
        file_map = {
            ResourceType.RECIPES: f'recipes1m+.{file_ending}',
            ResourceType.WIKIHOW: f'wikihow.{file_ending}',
            ResourceType.CUTTING_TUTORIALS: f'cutting_tutorials.{file_ending}'
        }
        return path.join(base_path, file_map[db_type])

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
