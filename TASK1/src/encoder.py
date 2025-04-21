from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List

class SentenceTransformerEncoder:                         # Encodes sentences into fixed-size embeddings using a Transformer backbone plus pooling.
    
    def __init__(                                                                  
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        pooling: str = "mean",
        device: str | None = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = pooling.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0]
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()                   # mean pooling
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**encoded, return_dict=True)
                pooled = self._pool(outputs.last_hidden_state, encoded.attention_mask)
                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.append(pooled.cpu().numpy())
        return np.vstack(all_embeddings)

if __name__ == "__main__":
    import argparse
    from src.utils import load_sentences, save_embeddings, save_embeddings_txt

    parser = argparse.ArgumentParser(
        description="Encode sentences to embeddings with configurable pooling"
    )
    parser.add_argument("--model_name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace model ID")
    parser.add_argument("--pooling", choices=["mean", "cls"],
                        default="mean", help="Pooling strategy")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to newline‑separated sentences")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--normalize", action="store_true",
                        help="L2‑normalize embeddings")
    parser.add_argument("--output_npy", type=str, default="embeddings.npy")
    parser.add_argument("--output_txt", type=str, default="embeddings.txt")
    args = parser.parse_args()

    sentences = load_sentences(args.input_file)
    encoder = SentenceTransformerEncoder(
        model_name=args.model_name,
        pooling=args.pooling
    )
    embeddings = encoder.encode(
        sentences,
        batch_size=args.batch_size,
        normalize=args.normalize
    )
    save_embeddings(embeddings, args.output_npy)
    save_embeddings_txt(embeddings, args.output_txt)

    print(f"Used pooling = {args.pooling}")
    print(f"Wrote binary embeddings to {args.output_npy}")
    print(f"Wrote human‑readable embeddings to {args.output_txt}")