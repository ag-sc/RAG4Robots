from pathlib import Path
from typing import List, Generator

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.enums import ResourceType
from src.rag.io_handler import get_data_from_resource


def batch_iterator(items: List[str], batch_size=256) -> Generator[List[str], None, None]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def create_new_from_file(res_type: ResourceType, model: SentenceTransformer, out_path: Path):
    text_docs = get_data_from_resource(res_type)
    if len(text_docs) == 0:
        return

    chunks = chunk_text_documents(text_docs) if res_type.needs_chunking else text_docs
    dim = model.get_sentence_embedding_dimension()
    dim_cols = [f'dim_{i}' for i in range(dim)]
    for batch in tqdm(batch_iterator(chunks, 5), 'Embedding the chunks'):
        vectors = model.encode(batch)
        df_batch = pd.DataFrame(vectors, columns=dim_cols)
        df_batch.insert(0, 'text', batch)
        df_batch.to_csv(out_path, mode='a', index=False, header=not out_path.is_file())


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
