import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class SentimentHead(nn.Module):
    def __init__(self, in_dim: int, num_sentiments: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_sentiments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class MultiTaskSentenceTransformer:                               # Sentence and sentiment classification
   
    def __init__(
        self,
        model_name: str,
        pooling: str,
        normalize: bool,
        num_classes: int,
        num_sentiments: int,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.pooling = pooling.lower()
        self.normalize = normalize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classification_head = ClassificationHead(hidden_size, num_classes)                  # Multi‑task heads
        self.sentiment_head = SentimentHead(hidden_size, num_sentiments)

        self.backbone.to(self.device).eval()                                              # Move everything to device
        self.classification_head.to(self.device).eval()
        self.sentiment_head.to(self.device).eval()

    def _pool(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0]
        mask = mask.unsqueeze(-1).expand(hidden_states.size()).float()                       # mean‑pool
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def predict(
        self,
        sentences: List[str],
        batch_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
       
        all_cls_logits = []
        all_sent_logits = []

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                out = self.backbone(**enc, return_dict=True)
                pooled = self._pool(out.last_hidden_state, enc.attention_mask)

                if self.normalize:
                    pooled = nn.functional.normalize(pooled, p=2, dim=1)

                cls_logits = self.classification_head(pooled)
                sent_logits = self.sentiment_head(pooled)

                all_cls_logits.append(cls_logits.cpu().numpy())
                all_sent_logits.append(sent_logits.cpu().numpy())

        return (
            np.vstack(all_cls_logits),
            np.vstack(all_sent_logits)
        )
