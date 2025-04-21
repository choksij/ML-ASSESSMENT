import argparse
import torch
import os
from transformers import AutoTokenizer, AutoModel
from typing import Literal
from torch.optim import AdamW
import matplotlib.pyplot as plt

def freeze_entire_model(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def freeze_backbone(model: torch.nn.Module):
    for p in model.backbone.parameters():
        p.requires_grad = False

def freeze_head(model: torch.nn.Module, head: Literal["classification", "sentiment"]):
    if head == "classification":
        for p in model.classification_head.parameters():
            p.requires_grad = False
    elif head == "sentiment":
        for p in model.sentiment_head.parameters():
            p.requires_grad = False
    else:
        raise ValueError("head must be 'classification' or 'sentiment'")

def print_grad_status(model):
    print("Backbone requires_grad:", 
          any(p.requires_grad for p in model.backbone.parameters()))
    print("Classification head requires_grad:", 
          any(p.requires_grad for p in model.classification_head.parameters()))
    print("Sentiment head requires_grad:", 
          any(p.requires_grad for p in model.sentiment_head.parameters()))

def get_optimizer(model: torch.nn.Module) -> AdamW:
    """
    Demonstrates setting up different learning rates for backbone vs. heads.
    """
    optimizer = AdamW([
        {"params": model.backbone.parameters(),          "lr": 1e-5},
        {"params": model.classification_head.parameters(), "lr": 1e-4},
        {"params": model.sentiment_head.parameters(),     "lr": 1e-4},
    ], weight_decay=0.01)
    return optimizer

def print_optimizer_config(optimizer: AdamW):
    print("\nOptimizer parameter groups and learning rates:")
    for i, group in enumerate(optimizer.param_groups):
        lr = group.get("lr", group.get("initial_lr", None))
        num_params = sum(p.numel() for p in group["params"])
        print(f"  Group {i}: lr={lr:.1e}, params={num_params}")


def get_trainability_status(model):
    return {
        "Backbone": any(p.requires_grad for p in model.backbone.parameters()),
        "Classification Head": any(p.requires_grad for p in model.classification_head.parameters()),
        "Sentiment Head": any(p.requires_grad for p in model.sentiment_head.parameters()),
    }

def plot_trainability_status(model, save_path="outputs/trainability.png"):
    status = get_trainability_status(model)
    labels = list(status.keys())
    values = [int(v) for v in status.values()]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values)

    for bar, val in zip(bars, values):
        hatch = "///" if val else "xxx"
        bar.set_hatch(hatch)
        txt = "Trainable" if val else "Frozen"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            txt,
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Trainable (1) / Frozen (0)")
    ax.set_title("Module Trainability Status")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    print(f"Saved trainability plot to {save_path}")
    return fig

class DummyMultiTaskModel(torch.nn.Module):
   
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        hidden = self.backbone.config.hidden_size
        self.classification_head = torch.nn.Linear(hidden, 3)
        self.sentiment_head     = torch.nn.Linear(hidden, 3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all","backbone","cls_head","sent_head"], required=True)
    args = parser.parse_args()

    model = DummyMultiTaskModel().eval()

    if args.mode == "all":
        freeze_entire_model(model)
    elif args.mode == "backbone":
        freeze_backbone(model)
    elif args.mode == "cls_head":
        freeze_head(model, "classification")
    else: 
        freeze_head(model, "sentiment")

    print_grad_status(model)

    optimizer = get_optimizer(model)
    print_optimizer_config(optimizer)

    plot_trainability_status(model)

if __name__ == "__main__":
    main()
