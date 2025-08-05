from typing import List, Any, Tuple

from src.rag.database import RAGDatabase
from src.utils.enums import ResourceType


class RAGManager:
    def __init__(self, db_types: List[Tuple[ResourceType, float]],
                 embed_mod='sentence-transformers/all-MiniLM-L6-v2') -> None:
        self._databases = []
        self._embedding_model = embed_mod
        for db in db_types:
            self._databases.append(RAGDatabase(db[0], db[1], embed_mod))

    def add_new_database(self, db_type: ResourceType, usage=1.0):
        self._databases.append(RAGDatabase(db_type, usage, self._embedding_model))

    def get_databases(self) -> List[RAGDatabase]:
        return self._databases

    def query_all_dbs(self, query: str, hits_to_return=3) -> List[Any]:
        all_res = []
        for db in self._databases:
            for res in db.query_current_db(query, hits_to_return):
                all_res.append((res[0], res[1], db.get_current_db()))
        return sorted(all_res, key=lambda x: x[1], reverse=True)[:hits_to_return]
