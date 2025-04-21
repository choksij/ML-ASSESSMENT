import argparse
import logging
import yaml
import numpy as np
from src.encoder import SentenceTransformerEncoder
from src.utils import load_sentences, save_embeddings, save_embeddings_txt
from src.utils import pairwise_cosine_similarity

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Sentence Transformer Encoder")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config"
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    logging.info("Loading %d sentences", len(open(cfg['data']['sample_file']).readlines()))

    sentences = load_sentences(cfg['data']['sample_file'])
    encoder = SentenceTransformerEncoder(
        model_name=cfg['model']['name'],
        pooling=cfg['model']['pooling']
    )
    embeddings = encoder.encode(
        sentences,
        batch_size=cfg['inference']['batch_size'],
        normalize=cfg['model']['normalize']
    )

    print(f"\nEmbeddings shape: {embeddings.shape}\n")
    logging.info("Embeddings shape: %s", embeddings.shape)

    save_embeddings(embeddings, cfg['output']['embeddings_file'])
    save_embeddings_txt(embeddings, cfg['output']['results_file'])
    logging.info("Saved embeddings to %s and %s",
                 cfg['output']['embeddings_file'],
                 cfg['output']['results_file'])

    for sent, emb in zip(sentences, embeddings):
        print(f"> {sent}\nEmbedding[:5]: {emb[:5].tolist()}\n{'-'*40}")


    if cfg['model']['normalize']:
        from src.utils import pairwise_cosine_similarity
        import numpy as np

        sim_matrix = pairwise_cosine_similarity(embeddings)                         # Computation of N×N cosine‑similarity matrix

        formatted = np.array2string(                                         # Formatting to 2 decimals
            sim_matrix,
            formatter={'float_kind': lambda x: f"{x:.2f}"}
        )

        print("\nCosine similarity matrix:\n", formatted)

if __name__ == "__main__":
    main()
