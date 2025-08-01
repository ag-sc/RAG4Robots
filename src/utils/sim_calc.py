from numpy import dot, ndarray, linalg


def calculate_similarity(embedding_1: ndarray, embedding_2: ndarray) -> float:
    assert len(embedding_1) == len(embedding_2)
    return dot(embedding_1, embedding_2) / (linalg.norm(embedding_1) * linalg.norm(embedding_2))
