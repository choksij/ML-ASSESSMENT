import numpy as np
from typing import List

def load_sentences(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def save_embeddings(array: np.ndarray, path: str) -> None:
    np.save(path, array)

def save_embeddings_txt(array: np.ndarray, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for vec in array:
            f.write(" ".join(f"{x:.6f}" for x in vec) + "\n")

def pairwise_cosine_similarity(embs: np.ndarray) -> np.ndarray:
    return embs @ embs.T
