import csv
import torch
from torch.utils.data import Dataset

class Task2Dataset(Dataset):
    """
    Loads sentence,class_label,sentiment_label CSV.
    """
    def __init__(self, file_path: str, cls2idx: dict, sent2idx: dict):
        self.sentences = []
        self.class_ids = []
        self.sentiment_ids = []
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.sentences.append(row["sentence"].strip('"'))
                self.class_ids.append(cls2idx[row["class_label"]])
                self.sentiment_ids.append(sent2idx[row["sentiment_label"]])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return (
            self.sentences[idx],
            torch.tensor(self.class_ids[idx], dtype=torch.long),
            torch.tensor(self.sentiment_ids[idx], dtype=torch.long),
        )

def collate_fn(batch):
    """
    Batch is list of (sentence, class_id, sentiment_id).
    """
    sentences, class_ids, sentiment_ids = zip(*batch)
    class_ids = torch.stack(class_ids)
    sentiment_ids = torch.stack(sentiment_ids)
    return list(sentences), class_ids, sentiment_ids
