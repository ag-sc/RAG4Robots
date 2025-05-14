import json
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

base_path = os.path.join(os.path.dirname(__file__), "..", "vector_db/")
file_end = ".faiss"
encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def convert_texts_to_vector(texts: List[str]):
    embd = encoder_model.encode(texts)
    vector_dimension = embd.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(embd)
    index.add(embd)
    return index


def save_vector_db_and_chunks(chunks: List[str], index, file: str):
    file_v = os.path.join(base_path, f'{file}{file_end}')
    file_c = os.path.join(base_path, f'{file}_chunks.json')
    faiss.write_index(index, file_v)
    with open(file_c, 'w+') as f:
        f.write(json.dumps(chunks))


def load_vector_db(file: str):
    file_p = os.path.join(os.path.dirname(__file__), base_path, f'{file}{file_end}')
    return faiss.read_index(file_p)


def load_chunks(file: str) -> List[str]:
    file_c = os.path.join(os.path.dirname(__file__), base_path, f'{file}_chunks.json')
    with open(file_c, "r") as f:
        chunks = json.load(f)
    return chunks


def get_db_file_name(file_type: str) -> str:
    file_map = {
        'recipes': 'all_recipes_cs500_co50',
        'wikihow': 'all_wikihow_cs500_co50',
        'tutorials': 'tutorials_cs500_co50',
        'combined': 'all_recwhowtut_cs500_co50'
    }
    return file_map[file_type]


def get_context_chunks(file_name: str, question: str, k=3) -> List[str]:
    index = load_vector_db(file_name)
    chunks = load_chunks(file_name)
    quest_embed = encoder_model.encode(question)
    _vector = np.array([quest_embed])
    faiss.normalize_L2(_vector)
    D, I = index.search(_vector, k=k)
    cont = []
    for i in I[0]:
        cont.append(chunks[i])
    return cont
