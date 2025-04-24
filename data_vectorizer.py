from os import path
from typing import List

import faiss
from sentence_transformers import SentenceTransformer

base_path = "vector_db/"
file_end = ".faiss"


def convert_texts_to_vector(texts: List[str]):
    encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embd = encoder_model.encode(texts)
    vector_dimension = embd.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(embd)
    index.add(embd)
    return index


def save_vector_db(index, file: str):
    file_p = path.join(base_path, f'{file}{file_end}')
    faiss.write_index(index, file_p)


def load_vector_db(file: str):
    file_p = path.join(base_path, f'{file}{file_end}')
    return faiss.read_index(file_p)
