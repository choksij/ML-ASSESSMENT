import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import matplotlib.pyplot as plt

from src.utils import Task2Dataset, collate_fn

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name, pooling, normalize, num_classes, num_sentiments):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone  = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        self.pooling   = pooling
        self.normalize = normalize
        self.class_head = nn.Linear(hidden, num_classes)
        self.sent_head  = nn.Linear(hidden, num_sentiments)

    def _pool(self, hidden_states, mask):
        if self.pooling == "cls":
            return hidden_states[:, 0]
        mask = mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, sentences):
        enc = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.backbone.device)
        out = self.backbone(**enc, return_dict=True)
        pooled = self._pool(out.last_hidden_state, enc.attention_mask)
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        cls_logits  = self.class_head(pooled)
        sent_logits = self.sent_head(pooled)
        return cls_logits, sent_logits

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False

def freeze_head(model, head):
    target = model.class_head if head == "cls_head" else model.sent_head
    for p in target.parameters():
        p.requires_grad = False

def plot_curve(x, ys, labels, xlabel, ylabel, title, save_path):
    plt.figure()
    for y, lbl in zip(ys, labels):
        plt.plot(x, y, label=lbl)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved {title} plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    cls2idx = {l: i for i, l in enumerate(cfg["tasks"]["classification"]["labels"])}                # label mappings & dataset split
    sent2idx = {l: i for i, l in enumerate(cfg["tasks"]["sentiment"]["labels"])}
    full_ds = Task2Dataset(cfg["data"]["sample_file"], cls2idx, sent2idx)

    raw_val  = int(len(full_ds) * cfg["training"]["val_split"])
    val_size = max(1, raw_val)
    train_size = len(full_ds) - val_size
    if train_size < 1:
        raise ValueError(f"Train size {train_size} < 1; lower val_split or add data.")

    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"                                           # model + device setup
    model = MultiTaskSentenceTransformer(
        cfg["model"]["name"], cfg["model"]["pooling"], cfg["model"]["normalize"],
        len(cls2idx), len(sent2idx)
    ).to(device)
    
    mode = cfg["training"]["freeze_mode"]                                                      # apply freeze strategy 
    if mode == "all":
        freeze_all(model)
    elif mode == "backbone":
        freeze_backbone(model)
    elif mode in ["cls_head", "sent_head"]:
        freeze_head(model, mode)

    backbone_lr = float(cfg["training"]["learning_rates"]["backbone"])                                     # optimizer & scheduler setup
    head_lr     = float(cfg["training"]["learning_rates"]["head"])
    optimizer = AdamW([
        {"params": model.backbone.parameters(),   "lr": backbone_lr},
        {"params": model.class_head.parameters(), "lr": head_lr},
        {"params": model.sent_head.parameters(),  "lr": head_lr},
    ], weight_decay=0.01)

    total_steps  = cfg["training"]["epochs"] * len(train_loader)
    warmup_steps = int(cfg["training"]["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    train_losses, val_losses = [], []                                                                  # metrics storage
    train_cls_accs, val_cls_accs = [], []
    train_sent_accs, val_sent_accs = [], []

    ce_loss = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs("checkpoints", exist_ok=True)


    for epoch in range(1, cfg["training"]["epochs"] + 1):                                               # training loop 
        model.train()
        tloss = tcls = tsent = 0
        for sentences, cls_ids, sent_ids in train_loader:
            cls_ids, sent_ids = cls_ids.to(device), sent_ids.to(device)
            cls_logits, sent_logits = model(sentences)

            loss_cls  = ce_loss(cls_logits, cls_ids)
            loss_sent = ce_loss(sent_logits, sent_ids)
            loss = (
                cfg["training"]["loss_weights"]["classification"] * loss_cls +
                cfg["training"]["loss_weights"]["sentiment"]      * loss_sent
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            tloss += loss.item() * cls_ids.size(0)
            tcls  += (cls_logits.argmax(1) == cls_ids).sum().item()
            tsent += (sent_logits.argmax(1) == sent_ids).sum().item()

        tloss /= train_size
        tcls  /= train_size
        tsent /= train_size

        model.eval()                                                                          # Validation phase
        vloss = vcls = vsent = 0
        with torch.no_grad():
            for sentences, cls_ids, sent_ids in val_loader:
                cls_ids, sent_ids = cls_ids.to(device), sent_ids.to(device)
                cls_logits, sent_logits = model(sentences)
                loss = ce_loss(cls_logits, cls_ids) + ce_loss(sent_logits, sent_ids)
                vloss += loss.item() * cls_ids.size(0)
                vcls  += (cls_logits.argmax(1) == cls_ids).sum().item()
                vsent += (sent_logits.argmax(1) == sent_ids).sum().item()

        vloss /= val_size
        vcls  /= val_size
        vsent /= val_size

        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")                                               # Print per-epoch metrics
        print(f"  Train Loss: {tloss:.4f} | Cls Acc: {tcls:.4f} | Sent Acc: {tsent:.4f}")
        print(f"  Val   Loss: {vloss:.4f} | Cls Acc: {vcls:.4f} | Sent Acc: {vsent:.4f}")

        train_losses.append(tloss)                                                             # Storing
        val_losses.append(vloss)
        train_cls_accs.append(tcls)
        val_cls_accs.append(vcls)
        train_sent_accs.append(tsent)
        val_sent_accs.append(vsent)

        if vloss < best_val_loss:                                                   # Checkpoint & early stopping
            best_val_loss = vloss
            torch.save(model.state_dict(), f"checkpoints/best_epoch_{epoch}.pt")
            print("New best model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement, patience {patience_counter}/{cfg['training']['patience']}")
            if patience_counter >= cfg["training"]["patience"]:
                print("Early stopping triggered.")
                break

    print(f"\n Training complete after {len(train_losses)} epochs.")

    x_axis = list(range(1, len(train_losses) + 1))
    plot_curve(x_axis, [train_losses, val_losses], ["train", "val"],
               "Epoch", "Loss", "Training vs Validation Loss", "outputs/loss_curve.png")
    plot_curve(x_axis, [train_cls_accs, val_cls_accs], ["train_cls", "val_cls"],
               "Epoch", "Accuracy", "Classification Accuracy", "outputs/cls_accuracy.png")
    plot_curve(x_axis, [train_sent_accs, val_sent_accs], ["train_sent", "val_sent"],
               "Epoch", "Accuracy", "Sentiment Accuracy", "outputs/sent_accuracy.png")

if __name__ == "__main__":
    main()
