import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import pytest
from src.main import main

def test_main_smoke(tmp_path, monkeypatch):
    sample_file = tmp_path / "sentences.txt"                              # Sample file
    sample_file.write_text("Hello world\nAnother line.\n")

    out_dir = tmp_path / "out"                                          # Output directory
    out_dir.mkdir()

    cfg = {                                                               # Config
        "model": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "pooling": "mean",
            "normalize": False
        },
        "inference": {
            "batch_size": 2
        },
        "data": {
            "sample_file": str(sample_file)
        },
        "output": {
            "embeddings_file": str(out_dir / "emb.npy"),
            "results_file": str(out_dir / "emb.txt")
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr("sys.argv", ["main.py", "--config", str(config_file)])

    main()

    assert (out_dir / "emb.npy").exists(), "Binary embeddings file not found"
    assert (out_dir / "emb.txt").exists(), "Text embeddings file not found"
