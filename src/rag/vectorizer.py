import io
import time
from pathlib import Path
from typing import List, Generator

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.enums import ResourceType
from src.rag.io_handler import get_data_from_resource

MAX_FILE_SIZE = 90 * 1024 * 1024  # 90MB in bytes (circumvent GitHub file size limit of 100MB)

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

    file_index = 0
    max_retries = 5
    retry_delay = 1.0
    current_file = out_path.with_name(f"{out_path.stem}_{file_index}{out_path.suffix}")
    current_size = current_file.stat().st_size if current_file.exists() else 0
    header_written = current_file.exists()

    for batch in tqdm(batch_iterator(chunks), 'Embedding the chunks'):
        vectors = model.encode(batch)
        df_batch = pd.DataFrame(vectors, columns=dim_cols)
        df_batch.insert(0, 'text', batch)

        # Estimate size of this batch in CSV format
        buffer = io.StringIO()
        df_batch.to_csv(buffer, index=False, header=not header_written)
        batch_bytes = buffer.getvalue().encode('utf-8')
        batch_size = len(batch_bytes)

        # Check if this batch fits in the current file
        if current_size + batch_size > MAX_FILE_SIZE:
            # Start a new file
            file_index += 1
            current_file = out_path.with_name(f"{out_path.stem}_{file_index}{out_path.suffix}")
            current_size = 0
            header_written = False

        for attempt in range(max_retries):
            try:
                with current_file.open('a', encoding='utf-8') as f:
                    f.write(buffer.getvalue())
                break
            except BlockingIOError:
                time.sleep(retry_delay)
        else:
            raise RuntimeError(f"Failed to write after {max_retries} retries with {retry_delay}s in between")

        current_size += batch_size
        header_written = True


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
