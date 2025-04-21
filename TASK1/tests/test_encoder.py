import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.encoder import SentenceTransformerEncoder

def test_encode_shape_and_type():
    sentences = ["Hello world", "Testing embeddings"]
    encoder = SentenceTransformerEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        pooling="mean")
    embs = encoder.encode(sentences, batch_size=2, normalize=False)
    assert isinstance(embs, np.ndarray)
    assert embs.ndim == 2
    assert embs.shape[0] == len(sentences)

if __name__ == "__main__":
    sentences = ["Hello world", "Testing embeddings"]              # Demo run
    encoder = SentenceTransformerEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        pooling="mean"
    )
    embs = encoder.encode(sentences, batch_size=2, normalize=False)
    print("Embeddings shape:", embs.shape)
    print(embs)