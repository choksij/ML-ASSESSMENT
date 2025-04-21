import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.multitask_encoder import MultiTaskSentenceTransformer

def test_multitask_output_shapes():
    sentences = ["A", "B", "C"]
    num_cls = 3
    num_sent = 3

    model = MultiTaskSentenceTransformer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        pooling="mean",
        normalize=False,
        num_classes=num_cls,
        num_sentiments=num_sent
    )
    cls_logits, sent_logits = model.predict(sentences, batch_size=2)

    assert isinstance(cls_logits, np.ndarray)
    assert isinstance(sent_logits, np.ndarray)
    assert cls_logits.shape == (3, num_cls)
    assert sent_logits.shape == (3, num_sent)
