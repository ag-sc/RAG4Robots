import csv
import gc
import time
from pathlib import Path
from typing import List, Generator, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from RAG4Robots.src.utils.enums import ResourceType
from RAG4Robots.src.rag.io_handler import get_data_from_resource

MAX_FILE_SIZE = 90 * 1024 * 1024  # 90MB in bytes (circumvent GitHub file size limit of 100MB)


def batch_iterator(items: Iterable[str], batch_size=128) -> Generator[List[str], None, None]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
        vectors = model.encode(batch, batch_size=16, show_progress_bar=False)
        for attempt in range(max_retries):
            try:
                with current_file.open('a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)

                    if not header_written:
                        writer.writerow(['text'] + dim_cols)
                        current_size += len(','.join(['text'] + dim_cols)) + 1  # Rough estimate
                        header_written = True

                    for text, vec in zip(batch, vectors):
                        row = [text] + list(map(str, vec))
                        row_data = ','.join(row) + '\n'
                        row_size = len(row_data.encode('utf-8'))

                        # Check size before writing
                        if current_size + row_size > MAX_FILE_SIZE:
                            file_index += 1
                            current_file = out_path.with_name(f"{out_path.stem}_{file_index}{out_path.suffix}")
                            current_size = 0
                            header_written = False
                            break

                        writer.writerow(row)
                        current_size += row_size
                break
            except BlockingIOError:
                time.sleep(retry_delay)
        else:
            raise RuntimeError(f"Failed to write after {max_retries} retries with {retry_delay}s in between")

        # Explicitly release memory used by these temporary variables
        del vectors
        gc.collect()


def chunk_text_documents(texts: List[str], chunk_size=500, chunk_overlap=0) -> Generator[str, None, None]:
    texts = [str(t) for t in texts if isinstance(t, str) or t is not None]
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        keep_separator='end',
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = chunker.create_documents(texts)
    for doc in tqdm(chunks, 'Creating chunks'):
        yield doc.page_content