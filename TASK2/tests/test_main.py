import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import pytest
from src.main import main

def test_main_task2(tmp_path, monkeypatch):
    sample = tmp_path / "data.csv"                            # Sample csv
    sample.write_text(
        "sentence,class_label,sentiment_label\n"
        "Hello world,tech,positive\n"
        "Bad news,finance,negative\n"
    )

    out = tmp_path / "out"                              # Output directory
    out.mkdir()

    cfg = {                                                           # Config
        "model": {"name":"sentence-transformers/all-MiniLM-L6-v2","pooling":"mean","normalize":False},
        "tasks": {
            "classification":{"labels":["entertainment","finance","tech"]},
            "sentiment":{"labels":["negative","neutral","positive"]}
        },
        "inference":{"batch_size":2},
        "data":{"sample_file": str(sample)},
        "output":{
            "classification_predictions": str(out/"class.csv"),
            "sentiment_predictions": str(out/"sent.csv")
        }
    }
    cfg_file = tmp_path/"cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr("sys.argv", ["main.py","--config",str(cfg_file)])

    main()

    assert (out/"class.csv").exists()
    assert (out/"sent.csv").exists()
