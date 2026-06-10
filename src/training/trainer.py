"""
trainer.py
==========
Training loop for Raqeeb 23-class Arabic MT error classifier.

Usage:
    python src/training/trainer.py --config configs/arabert_v2.yaml --fold 0

Produces in experiments/{model}/{arch}/fold_{N}/:
    checkpoint/           saved model weights
    predictions_val.csv   val-set predictions
    training_log.json     per-epoch metrics
    metrics.json          fold summary (gold fields filled by evaluator.py)
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))
from src.models.encoder import RaqeebEncoder


# ── Dataset ──────────────────────────────────────────────────

class RaqeebDataset(Dataset):
    def __init__(self, df: pd.DataFrame, model: RaqeebEncoder, label_map: dict):
        self.df        = df.reset_index(drop=True)
        self.model     = model
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pair   = self.model.build_input_pair(row.to_dict())
        label  = self.label_map[row["Sub-Subtype"]]
        return pair, label, str(row.get("sample_id", idx))


def collate_fn(batch, model, device):
    pairs, labels, ids = zip(*batch)
    encoded = model.tokenise_batch(list(pairs), device)
    labels  = torch.tensor(labels, dtype=torch.long).to(device)
    return encoded, labels, list(ids)


# ── Training loop ─────────────────────────────────────────────

def train_fold(cfg: dict, fold: int):
    # ── Paths ──
    model_name = cfg["model"]["name"]
    arch_cfg   = cfg.get("architecture", "flat")
    arch       = arch_cfg if isinstance(arch_cfg, str) else arch_cfg.get("type", "flat")
    model_slug = model_name.split("/")[-1].lower().replace("-", "_")
    out_dir    = BASE / cfg["output"]["base_dir"] / arch / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Seed ──
    seed = fold  # fold 0 → seed 0, fold 1 → seed 1, etc.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Model  : {model_name}")
    print(f"Fold   : {fold}  (seed={seed})")
    print(f"Device : {device}")
    print(f"Output : {out_dir}")
    print(f"{'='*60}\n")

    # ── Load data ──
    proc_dir   = BASE / "data" / "processed"
    splits_dir = BASE / "data" / "splits"

    label_map_path = proc_dir / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}

    train_file  = BASE / cfg["data"]["train_file"]
    splits_file = BASE / cfg["data"]["splits_file"]

    df_full = pd.read_csv(train_file, encoding="utf-8-sig")
    with open(splits_file) as f:
        splits = json.load(f)

    fold_key  = f"fold_{fold}"
    train_idx = splits[fold_key]["train"]
    val_idx   = splits[fold_key]["val"]

    df_train = df_full.iloc[train_idx].reset_index(drop=True)
    df_val   = df_full.iloc[val_idx].reset_index(drop=True)

    print(f"Train: {len(df_train)} rows | Val: {len(df_val)} rows")

    # ── Model ──
    input_format = cfg["model"].get("input_format", "A")
    max_length   = cfg["model"].get("max_length", 256)
    dropout      = cfg["model"].get("dropout", 0.1)

    model = RaqeebEncoder(
        model_name=model_name,
        num_labels=len(label_map),
        input_format=input_format,
        max_length=max_length,
        dropout=dropout,
    ).to(device)

    # ── Class weights ──
    use_weighted = cfg.get("loss", {}).get("weighted", True)
    if use_weighted:
        labels_array = df_train["Sub-Subtype"].map(label_map).values
        present_classes = np.unique(labels_array)
        weights_present = compute_class_weight(
            "balanced", classes=present_classes, y=labels_array
        )
        # Build full weight array — missing classes get weight 0.0
        class_weights = np.ones(len(label_map))
        for cls_id, w in zip(present_classes, weights_present):
            class_weights[cls_id] = w
        for i in range(len(label_map)):
            if i not in present_classes:
                class_weights[i] = 0.0
        weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── DataLoaders ──
    batch_size   = cfg["training"]["batch_size"]
    grad_accum   = cfg["training"].get("gradient_accumulation_steps", 1)

    train_dataset = RaqeebDataset(df_train, model, label_map)
    val_dataset   = RaqeebDataset(df_val,   model, label_map)

    def make_collate(m, d):
        def fn(batch):
            return collate_fn(batch, m, d)
        return fn

    collate = make_collate(model, device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        collate_fn=collate, num_workers=0
    )

    # ── Optimiser + scheduler ──
    lr           = cfg["training"]["learning_rate"]
    epochs       = cfg["training"]["epochs"]
    warmup_ratio = cfg["training"].get("warmup_ratio", 0.1)
    wd           = cfg["training"].get("weight_decay", 0.01)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps   = (len(train_loader) // grad_accum) * epochs
    warmup_steps  = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    fp16 = cfg["training"].get("fp16", False) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if fp16 else None

    patience        = cfg["training"].get("early_stopping_patience", 3)
    best_val_f1     = -1.0
    best_val_wf1    = -1.0
    best_val_acc    = -1.0
    best_epoch      = 0
    no_improve      = 0
    epoch_log       = []
    gpu_peak_mb     = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()

    # ── Epoch loop ──
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (encoded, labels, _) in enumerate(train_loader):
            with torch.amp.autocast('cuda', enabled=fp16):
                logits = model(**encoded)
                loss   = criterion(logits, labels) / grad_accum

            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                if fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # Validate
        val_macro_f1, val_weighted_f1, val_acc, val_loss = evaluate(
            model, val_loader, criterion, label_map, device, fp16
        )

        epoch_wall_sec = time.time() - epoch_start

        # Track GPU peak memory
        if torch.cuda.is_available():
            gpu_peak_mb = max(gpu_peak_mb,
                              torch.cuda.max_memory_allocated(device) / (1024**2))

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f} | "
            f"wall={epoch_wall_sec:.0f}s | "
            f"gpu_peak={gpu_peak_mb:.0f}MB"
        )

        epoch_log.append({
            "epoch":          epoch,
            "train_loss":     round(avg_train_loss, 4),
            "val_loss":       round(val_loss, 4),
            "val_macro_f1":   round(val_macro_f1, 4),
            "val_weighted_f1": round(val_weighted_f1, 4),
            "val_accuracy":   round(val_acc, 4),
            "wall_sec":       round(epoch_wall_sec, 1),
            "gpu_peak_mb":    round(gpu_peak_mb, 0),
        })

        # Early stopping + checkpoint
        if val_macro_f1 > best_val_f1:
            best_val_f1  = val_macro_f1
            best_val_wf1 = val_weighted_f1
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            ckpt_dir = out_dir / "checkpoint"
            ckpt_dir.mkdir(exist_ok=True)
            model.encoder.save_pretrained(ckpt_dir)
            model.tokenizer.save_pretrained(ckpt_dir)
            torch.save(model.classifier.state_dict(), ckpt_dir / "classifier_head.pt")
            torch.save({
                "input_format": model.input_format,
                "num_labels":   model.num_labels,
                "max_length":   model.max_length,
            }, ckpt_dir / "config.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    elapsed = (time.time() - start_time) / 3600

    # ── Predictions on val set ──
    preds_df = get_predictions(model, val_loader, label_map, id2label, device, fp16)
    preds_df.to_csv(out_dir / "predictions_val.csv", index=False, encoding="utf-8-sig")

    # ── Per-class F1 ──
    from sklearn.metrics import f1_score, classification_report
    per_class = {}
    report = classification_report(
        preds_df["true_label"], preds_df["pred_label"],
        labels=list(label_map.keys()), output_dict=True, zero_division=0
    )
    for cls in label_map:
        per_class[cls] = round(report.get(cls, {}).get("f1-score", 0.0), 4)

    # ── Save training log ──
    training_log = {
        "model":                model_name,
        "fold":                 fold,
        "best_epoch":           best_epoch,
        "early_stop_triggered": no_improve >= patience,
        "early_stop_patience":  patience,
        "epochs":               epoch_log,
    }
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # ── Save metrics (val fields only — gold filled by evaluator.py) ──
    metrics = {
        "model":              model_slug,
        "huggingface_id":     model_name,
        "fold":               fold,
        "architecture":       arch,
        "input_format":       input_format,
        "training_data":      Path(cfg["data"]["train_file"]).stem,
        "weighted_loss":      use_weighted,
        "n_train":            len(df_train),
        "n_val":              len(df_val),
        "n_epochs_run":       len(epoch_log),
        "best_epoch":         best_epoch,
        "early_stop_triggered": no_improve >= patience,
        "val_macro_f1":       round(best_val_f1, 4),
        "val_weighted_f1":    round(best_val_wf1, 4),
        "val_accuracy":       round(best_val_acc, 4),
        "gold_macro_f1":      None,
        "gold_weighted_f1":   None,
        "gold_accuracy":      None,
        "gold_macro_f1_ci_lower":  None,
        "gold_macro_f1_ci_upper":  None,
        "transfer_macro_f1":  None,
        "per_class_f1_val":   per_class,
        "per_class_f1_gold":  None,
        "hardware":           str(device),
        "gpu_name":           torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "gpu_peak_memory_mb": round(gpu_peak_mb, 0),
        "training_time_hours": round(elapsed, 2),
        "training_wall_sec":  round(time.time() - start_time, 1),
        "seed":               seed,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Fold {fold} complete — best val Macro-F1 = {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"   Saved to: {out_dir}")
    print(f"   Training time: {elapsed:.2f}h")
    print(f"\n   Next: run fold {fold + 1} or compare val F1 across folds and run evaluator.py")


def evaluate(model, loader, criterion, label_map, device, fp16):
    from sklearn.metrics import f1_score, accuracy_score
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for encoded, labels, _ in loader:
            with torch.amp.autocast('cuda', enabled=fp16):
                logits = model(**encoded)
                loss   = criterion(logits, labels)
            total_loss  += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy    = accuracy_score(all_labels, all_preds)
    avg_loss    = total_loss / len(loader)
    return macro_f1, weighted_f1, accuracy, avg_loss


def get_predictions(model, loader, label_map, id2label, device, fp16):
    model.eval()
    rows = []
    with torch.no_grad():
        for encoded, labels, ids in loader:
            with torch.amp.autocast('cuda', enabled=fp16):
                logits = model(**encoded)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)
            confs  = probs.max(dim=1).values

            for sid, true, pred, conf in zip(
                ids,
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                confs.cpu().numpy(),
            ):
                rows.append({
                    "sample_id":      sid,
                    "true_label":     id2label[int(true)],
                    "pred_label":     id2label[int(pred)],
                    "pred_confidence": round(float(conf), 4),
                    "true_label_id":  int(true),
                    "pred_label_id":  int(pred),
                })
    return pd.DataFrame(rows)


# ── Config loading ────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    base_path = BASE / "configs" / "base_encoder.yaml"
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    with open(config_path) as f:
        override = yaml.safe_load(f)

    # Deep merge override into base
    def merge(base, over):
        for k, v in over.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                merge(base[k], v)
            else:
                base[k] = v
    merge(cfg, override)
    return cfg


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Raqeeb classifier — one fold")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--fold",   required=True, type=int, choices=[0,1,2,3,4],
                        help="Fold to train (0-4)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_fold(cfg, args.fold)
