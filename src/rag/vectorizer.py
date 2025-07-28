from typing import List

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.enums import ResourceType
from src.rag.io_handler import get_data_from_resource


def create_new_from_file(res_type: ResourceType, model: SentenceTransformer) -> pd.DataFrame:
    text_docs = get_data_from_resource(res_type)
    if len(text_docs) == 0:
        return pd.DataFrame()

    all_chunks = []
    all_vectors = []
    for chunk in tqdm(chunk_text_documents(text_docs), 'Embedding the chunks'):
        vectors = model.encode(chunk)
        all_chunks.append(chunk)
        all_vectors.append(vectors)

    dim_cols = [f'dim_{i}' for i in range(len(all_vectors[0]))]
    df_vectors = pd.DataFrame(all_vectors, columns=dim_cols)
    df_vectors.insert(0, 'text', all_chunks)
    return df_vectors


def chunk_text_documents(texts: List[str], chunk_size=500, chunk_overlap=50) -> List[str]:
    texts = [str(t) for t in texts if isinstance(t, str) or t is not None]
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = chunker.create_documents(texts)
    chunk_texts = [doc.page_content for doc in tqdm(chunks, 'Creating chunks')]
    return chunk_texts

# def get_context_chunks(file_name: str, question: str, k=3) -> List[str]:
#    index = load_vector_db(file_name)
#    chunks = load_chunks(file_name)
#    quest_embed = encoder_model.encode(question)
#    _vector = np.array([quest_embed])
#    faiss.normalize_L2(_vector)
#    D, I = index.search(_vector, k=k)
#    cont = []
#    for i in I[0]:
#        cont.append(chunks[i])
#    return cont
