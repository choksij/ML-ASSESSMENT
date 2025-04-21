import argparse
import logging
import yaml
import numpy as np

from src.multitask_encoder import MultiTaskSentenceTransformer
from src.utils import (
    load_task2_data,
    save_classification_predictions,
    save_sentiment_predictions,
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Task 2: Multi‑Task Inference")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    cls_labels = cfg["tasks"]["classification"]["labels"]                  # Task label mappings
    sent_labels = cfg["tasks"]["sentiment"]["labels"]
    cls2idx = {l: i for i, l in enumerate(cls_labels)}
    sent2idx = {l: i for i, l in enumerate(sent_labels)}

    sentences, _, _ = load_task2_data(
        cfg["data"]["sample_file"],
        cls2idx,
        sent2idx
    )
    logging.info("Loaded %d sentences", len(sentences))

    model = MultiTaskSentenceTransformer(                                      # Build model
        model_name=cfg["model"]["name"],
        pooling=cfg["model"]["pooling"],
        normalize=cfg["model"]["normalize"],
        num_classes=len(cls_labels),
        num_sentiments=len(sent_labels)
    )

    cls_logits, sent_logits = model.predict(                            # Inference
        sentences,
        batch_size=cfg["inference"]["batch_size"]
    )

    print(f"Classification logits shape: {cls_logits.shape}")
    print(f"Sentiment logits shape:    {sent_logits.shape}")

    save_classification_predictions(
        sentences, cls_logits, cls_labels, cfg["output"]["classification_predictions"]
    )
    save_sentiment_predictions(
        sentences, sent_logits, sent_labels, cfg["output"]["sentiment_predictions"]
    )
    logging.info("Saved predictions to %s and %s",
                 cfg["output"]["classification_predictions"],
                 cfg["output"]["sentiment_predictions"])

    for s, c_logit, st_logit in zip(sentences, cls_logits, sent_logits):                       # Summary
        c_pred = cls_labels[int(np.argmax(c_logit))]
        st_pred = sent_labels[int(np.argmax(st_logit))]
        print(f"> {s}\n  Class: {c_pred} | Sentiment: {st_pred}\n{'-'*40}")

if __name__ == "__main__":
    main()
