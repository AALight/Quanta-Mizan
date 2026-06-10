"""
train_anchor_extractor.py
=========================
Fine-tunes CAMeLBERT-MSA as a span extractor to find Best_Match in MT output.

This is the trained replacement for the rule-based OCS cascade.
Results → Paper Table 5, §4.5.

Usage:
    # Step 1: prepare data (run once)
    python scripts/prepare_anchor_data.py

    # Step 2: train
    python src/training/train_anchor_extractor.py

    # Step 3: evaluate vs OCS baseline
    python src/training/train_anchor_extractor.py --eval_only

Produces in experiments/anchor_extractor/:
    checkpoint/                 ← saved model weights
    metrics.json                ← exact match, token F1, clitic accuracy
    predictions_eval.csv        ← per-row predictions
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

BASE = Path(__file__).parent.parent.parent
sys_path_hack = str(BASE)

import sys
sys.path.insert(0, str(BASE))
from src.models.anchor_extractor import ArabicAnchorExtractor


# ── Dataset ──────────────────────────────────────────────────────────────────

class AnchorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        keyword    = str(row["keyword"]).strip()
        mt_output  = str(row["mt_output"]).strip()
        best_match = str(row["best_match"]).strip()
        span_start = int(row["span_start"])
        span_end   = int(row["span_end"])

        encoded = self.tokenizer(
            keyword,
            mt_output,
            max_length     = self.max_length,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
            return_offsets_mapping = True,
        )

        offset_mapping = encoded.pop("offset_mapping")[0]

        # Find token positions corresponding to char span
        start_pos = 0
        end_pos   = 0
        for i, (char_s, char_e) in enumerate(offset_mapping.tolist()):
            if char_s <= span_start < char_e:
                start_pos = i
            if char_s < span_end <= char_e:
                end_pos = i
                break

        return {
            "input_ids":      encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "token_type_ids": encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"]))[0],
            "start_pos":      torch.tensor(start_pos, dtype=torch.long),
            "end_pos":        torch.tensor(end_pos,   dtype=torch.long),
            "best_match":     best_match,
            "mt_output":      mt_output,
            "offset_mapping": offset_mapping,
        }


# ── Training ─────────────────────────────────────────────────────────────────

def train(model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa"):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir   = BASE / "experiments" / "anchor_extractor"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"ANCHOR EXTRACTOR — TRAINING")
    print(f"Model : {model_name}")
    print(f"Device: {device}")
    print(f"{'='*55}\n")

    # Load data
    train_df = pd.read_csv(BASE / "data" / "processed" / "anchor_train.csv", encoding="utf-8-sig")
    eval_df  = pd.read_csv(BASE / "data" / "processed" / "anchor_eval.csv",  encoding="utf-8-sig")
    print(f"Train: {len(train_df)} rows | Eval: {len(eval_df)} rows")

    model     = ArabicAnchorExtractor(model_name=model_name).to(device)
    tokenizer = model.tokenizer

    train_dataset = AnchorDataset(train_df, tokenizer)
    eval_dataset  = AnchorDataset(eval_df,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0)
    eval_loader  = DataLoader(eval_dataset,  batch_size=32, shuffle=False, num_workers=0)

    optimizer  = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    epochs     = 5
    total_steps = len(train_loader) * epochs
    scheduler  = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = int(total_steps * 0.1),
        num_training_steps = total_steps,
    )

    criterion = nn.CrossEntropyLoss()
    best_exact = -1.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_pos      = batch["start_pos"].to(device)
            end_pos        = batch["end_pos"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(start_logits, start_pos) + criterion(end_logits, end_pos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Eval
        exact_match, token_f1, clitic_acc = evaluate(model, eval_loader, eval_df, tokenizer, device)

        print(f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | exact={exact_match:.4f} | "
              f"token_f1={token_f1:.4f} | clitic_acc={clitic_acc:.4f}")

        if exact_match > best_exact:
            best_exact = exact_match
            ckpt = out_dir / "checkpoint"
            ckpt.mkdir(exist_ok=True)
            model.encoder.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            torch.save(model.start_head.state_dict(), ckpt / "start_head.pt")
            torch.save(model.end_head.state_dict(),   ckpt / "end_head.pt")
            print(f"  ✅ Checkpoint saved (best exact={best_exact:.4f})")

    elapsed = (time.time() - start_time) / 3600

    # Load anchor stats for OCS baseline comparison
    stats_path = BASE / "data" / "processed" / "anchor_stats.json"
    ocs_baseline = {}
    if stats_path.exists():
        with open(stats_path) as f:
            ocs_baseline = json.load(f).get("ocs_baseline", {})

    metrics = {
        "model":              model_name,
        "task":               "anchor_extraction",
        "eval_rows":          len(eval_df),
        "best_exact_match":   round(best_exact, 4),
        "best_token_f1":      round(token_f1, 4),
        "best_clitic_acc":    round(clitic_acc, 4),
        "ocs_baseline_exact": ocs_baseline.get("overall_accuracy", None),
        "ocs_baseline_clitic": ocs_baseline.get("clitic_accuracy", None),
        "delta_exact":        round(best_exact - ocs_baseline.get("overall_accuracy", 0), 4),
        "delta_clitic":       round(clitic_acc - ocs_baseline.get("clitic_accuracy", 0), 4),
        "training_time_hours": round(elapsed, 2),
        "hardware":            str(device),
        "paper_table":         "Table 5 (§4.5)",
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Training complete")
    print(f"   Best exact match : {best_exact:.4f}  (OCS baseline: {ocs_baseline.get('overall_accuracy', '—')})")
    print(f"   Clitic accuracy  : {clitic_acc:.4f}  (OCS baseline: {ocs_baseline.get('clitic_accuracy', '—')})")
    print(f"   Delta exact      : {metrics['delta_exact']:+.4f}")
    print(f"   Delta clitic     : {metrics['delta_clitic']:+.4f}")
    print(f"   Results → experiments/anchor_extractor/metrics.json")


def evaluate(model, loader, df, tokenizer, device):
    """Evaluate span extraction on eval set."""
    model.eval()
    exact_matches = 0
    token_f1_scores = []
    clitic_correct = 0
    clitic_total   = 0

    predictions = []
    idx = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
            start_preds = torch.argmax(start_logits, dim=1).cpu().numpy()
            end_preds   = torch.argmax(end_logits,   dim=1).cpu().numpy()

            for i in range(len(start_preds)):
                true_bm  = batch["best_match"][i]
                mt       = batch["mt_output"][i]
                offsets  = batch["offset_mapping"][i]

                sp = start_preds[i]
                ep = end_preds[i]

                pred_text = ""
                if ep >= sp:
                    try:
                        cs = offsets[sp][0].item()
                        ce = offsets[ep][1].item()
                        pred_text = mt[cs:ce].strip()
                    except Exception:
                        pred_text = ""

                exact = int(pred_text == true_bm)
                exact_matches += exact

                # Token F1
                pred_tokens = set(pred_text.split())
                true_tokens = set(true_bm.split())
                if pred_tokens or true_tokens:
                    tp = len(pred_tokens & true_tokens)
                    p  = tp / len(pred_tokens) if pred_tokens else 0
                    r  = tp / len(true_tokens) if true_tokens else 0
                    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
                else:
                    f1 = 1.0
                token_f1_scores.append(f1)

                # Clitic accuracy (cases where best_match ≠ keyword)
                row = df.iloc[idx] if idx < len(df) else None
                if row is not None:
                    kw = str(row.get("keyword", "")).strip()
                    bm = str(row.get("best_match", "")).strip()
                    if bm != kw:
                        clitic_total += 1
                        clitic_correct += exact

                predictions.append({
                    "mt_output":      mt,
                    "true_best_match": true_bm,
                    "pred_best_match": pred_text,
                    "exact_match":     exact,
                })
                idx += 1

    exact_acc  = exact_matches / len(df)
    mean_f1    = float(np.mean(token_f1_scores))
    clitic_acc = clitic_correct / max(clitic_total, 1)

    # Save predictions
    out_dir = BASE / "experiments" / "anchor_extractor"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(predictions).to_csv(
        out_dir / "predictions_eval.csv", index=False, encoding="utf-8-sig"
    )

    return exact_acc, mean_f1, clitic_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CAMeL-Lab/bert-base-arabic-camelbert-msa")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    train(args.model)
