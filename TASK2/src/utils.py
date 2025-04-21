import csv
from typing import List, Tuple

def load_task2_data(
    file_path: str,
    class2idx: dict,
    sentiment2idx: dict
) -> Tuple[List[str], List[int], List[int]]:
    """
    Reads CSV with columns: sentence,class_label,sentiment_label
    Returns:
      • sentences: List[str]
      • class_ids: List[int]
      • sentiment_ids: List[int]
    """
    sentences, class_ids, sentiment_ids = [], [], []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences.append(row["sentence"].strip('"'))
            class_ids.append(class2idx[row["class_label"]])
            sentiment_ids.append(sentiment2idx[row["sentiment_label"]])
    return sentences, class_ids, sentiment_ids

def save_classification_predictions(
    sentences: List[str],
    logits: object,
    label_list: List[str],
    out_file: str
):
    import numpy as np
    preds = np.argmax(logits, axis=1)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "predicted_class"])
        for s, p in zip(sentences, preds):
            writer.writerow([s, label_list[p]])

def save_sentiment_predictions(
    sentences: List[str],
    logits: object,
    label_list: List[str],
    out_file: str
):
    import numpy as np
    preds = np.argmax(logits, axis=1)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "predicted_sentiment"])
        for s, p in zip(sentences, preds):
            writer.writerow([s, label_list[p]])
