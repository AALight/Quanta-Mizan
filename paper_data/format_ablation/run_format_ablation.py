"""
Format Ablation — AraBERT v2 on fixed T1+T2
Compares: Format A (blind: source + MT) vs Format D (anchor: kw/bm + source + MT)

Clean ablation on IDENTICAL fixed data — isolates the effect of the anchor input.

Usage:
    cd /path/to/delta_upload
    python delta/run_format_ablation.py
"""

import json
import os
import sys
import time
import random
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
PROJ = Path(os.getcwd())
sys.path.insert(0, str(PROJ))
print("Working directory: %s" % PROJ)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: %s" % device)

MODEL_NAME = "aubmindlab/bert-base-arabertv02"
MAX_LENGTH = 512
LR = 5e-5
BATCH_SIZE = 32
EPOCHS = 5
N_FOLDS = 5

with open("data/processed/label_map.json") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)

# ── Load fixed T1+T2 (same as benchmark) ──
t2_df = pd.read_csv("data/processed/train_t2.csv", encoding="utf-8-sig")
gold_df = pd.read_csv("data/processed/test_gold.csv", encoding="utf-8-sig")
t1t2_raw = pd.read_csv("data/processed/train_t1t2.csv", encoding="utf-8-sig")
t1_df = t1t2_raw[t1t2_raw["source_tier"] == "tier1"].reset_index(drop=True)
t1t2_df = pd.concat([t1_df, t2_df], ignore_index=True)

print("T1+T2: %d | Gold: %d" % (len(t1t2_df), len(gold_df)))


class PairDataset(Dataset):
    def __init__(self, df, tokenizer, label_map, fmt):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.fmt = fmt  # "A" or "D"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ar = str(row.get("ar", "")).strip()
        mt = str(row.get("mt_output", "")).strip()

        if self.fmt == "D":
            kw = str(row.get("Keyword", "")).strip()
            bm = str(row.get("Best_Match", "")).strip()
            if not bm or bm in ["[OMITTED]", "nan", "None", ""]:
                bm = "[OMITTED]"
            # Format D: anchor = kw+bm, context = ar+mt
            anchor = ("%s %s" % (kw, bm)).strip() if kw and kw != "nan" else ar
            context = ("%s %s" % (ar, mt)).strip()
            label = self.label_map[row["Sub-Subtype"]]
            return (anchor, context), label
        else:
            # Format A (blind): just source and MT, no anchor
            label = self.label_map[row["Sub-Subtype"]]
            return (ar, mt), label


def collate_fn(batch, tokenizer, device):
    pairs, labels = zip(*batch)
    enc = tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
                    padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}, torch.tensor(labels, dtype=torch.long).to(device)


def train_fold(df_train, df_val, fold, fmt):
    seed = fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, NUM_CLASSES)).to(device)

    labels_array = df_train["Sub-Subtype"].map(label_map).values
    present = np.unique(labels_array)
    w = compute_class_weight("balanced", classes=present, y=labels_array)
    cw = np.ones(NUM_CLASSES)
    for c, wt in zip(present, w):
        cw[c] = wt
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float).to(device))

    def make_collate(tok, dev):
        def fn(batch): return collate_fn(batch, tok, dev)
        return fn

    train_loader = DataLoader(PairDataset(df_train, tokenizer, label_map, fmt),
                              batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=make_collate(tokenizer, device))
    val_loader = DataLoader(PairDataset(df_val, tokenizer, label_map, fmt),
                            batch_size=BATCH_SIZE * 2, shuffle=False,
                            collate_fn=make_collate(tokenizer, device))

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    scaler = torch.amp.GradScaler("cuda")

    best_val_f1 = -1
    best_es, best_cs = None, None

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        classifier.train()
        for enc, lab in train_loader:
            with torch.amp.autocast("cuda"):
                out = encoder(**enc)
                logits = classifier(out.last_hidden_state[:, 0, :])
                loss = criterion(logits, lab)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        encoder.eval()
        classifier.eval()
        preds, labs = [], []
        with torch.no_grad():
            for enc, lab in val_loader:
                with torch.amp.autocast("cuda"):
                    out = encoder(**enc)
                    logits = classifier(out.last_hidden_state[:, 0, :])
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                labs.extend(lab.cpu().numpy())

        val_f1 = f1_score(labs, preds, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_es = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_cs = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    encoder.load_state_dict(best_es)
    classifier.load_state_dict(best_cs)
    encoder.to(device).eval()
    classifier.to(device).eval()

    gold_loader = DataLoader(PairDataset(gold_df, tokenizer, label_map, fmt),
                             batch_size=BATCH_SIZE * 2, shuffle=False,
                             collate_fn=make_collate(tokenizer, device))
    preds, labs = [], []
    with torch.no_grad():
        for enc, lab in gold_loader:
            with torch.amp.autocast("cuda"):
                out = encoder(**enc)
                logits = classifier(out.last_hidden_state[:, 0, :])
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            labs.extend(lab.cpu().numpy())

    gold_f1 = f1_score(labs, preds, average="macro", zero_division=0)
    gold_acc = accuracy_score(labs, preds)

    del encoder, classifier, optimizer
    torch.cuda.empty_cache()

    return best_val_f1, gold_f1, gold_acc, preds, labs


def run_format(fmt):
    print("\n" + "=" * 60)
    print("FORMAT %s — AraBERT v2 — T1+T2 (fixed)" % fmt)
    print("=" * 60)

    label_counts = t1t2_df["Sub-Subtype"].value_counts()
    rare = label_counts[label_counts < N_FOLDS].index.tolist()
    rare_mask = t1t2_df["Sub-Subtype"].isin(rare)
    df_rare = t1t2_df[rare_mask].reset_index(drop=True)
    df_strat = t1t2_df[~rare_mask].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    labels = df_strat["Sub-Subtype"].values

    fold_results = []
    best_gold = -1
    best_preds = None
    best_labels = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_strat, labels)):
        df_ft = pd.concat([df_strat.iloc[train_idx], df_rare], ignore_index=True)
        df_fv = df_strat.iloc[val_idx]

        print("\n--- Fold %d ---" % fold)
        t = time.time()
        val_f1, gold_f1, gold_acc, preds, labs = train_fold(df_ft, df_fv, fold, fmt)
        print("  val=%.4f gold=%.4f acc=%.4f (%.1f min)" % (val_f1, gold_f1, gold_acc, (time.time() - t) / 60))

        fold_results.append({"fold": fold, "val_f1": round(val_f1, 4),
                            "gold_f1": round(gold_f1, 4), "gold_acc": round(gold_acc, 4)})
        if gold_f1 > best_gold:
            best_gold = gold_f1
            best_preds = preds
            best_labels = labs

    val_f1s = [r["val_f1"] for r in fold_results]
    gold_f1s = [r["gold_f1"] for r in fold_results]
    gold_accs = [r["gold_acc"] for r in fold_results]

    print("\n" + "-" * 40)
    print("Format %s:  Mean=%.4f+-%.4f  Best=%.4f" % (
        fmt, statistics.mean(gold_f1s), statistics.stdev(gold_f1s), max(gold_f1s)))

    # Per-class
    true_str = [id2label[l] for l in best_labels]
    pred_str = [id2label[p] for p in best_preds]
    report = classification_report(true_str, pred_str, labels=list(label_map.keys()), zero_division=0, digits=3)

    return {
        "format": fmt,
        "fold_results": fold_results,
        "mean_gold_f1": round(statistics.mean(gold_f1s), 4),
        "std_gold_f1": round(statistics.stdev(gold_f1s), 4),
        "best_gold_f1": round(max(gold_f1s), 4),
        "mean_gold_acc": round(statistics.mean(gold_accs), 4),
        "mean_val_f1": round(statistics.mean(val_f1s), 4),
        "best_preds": best_preds,
        "best_labels": best_labels,
        "per_class_report": report,
    }


# ── Main ──
print("\n" + "#" * 60)
print("FORMAT ABLATION — A (blind) vs D (anchor)")
print("Same model (AraBERT v2), same data (fixed T1+T2), same hyperparams")
print("#" * 60)

total_start = time.time()
result_a = run_format("A")
result_d = run_format("D")

# ── Comparison ──
print("\n" + "=" * 70)
print("FORMAT ABLATION RESULTS")
print("=" * 70)
print("%-10s %12s %12s %12s %12s" % ("Format", "Mean F1", "Std", "Best F1", "Mean Acc"))
print("-" * 70)
print("%-10s %12.4f %12.4f %12.4f %12.4f" % (
    "A (blind)", result_a["mean_gold_f1"], result_a["std_gold_f1"],
    result_a["best_gold_f1"], result_a["mean_gold_acc"]))
print("%-10s %12.4f %12.4f %12.4f %12.4f" % (
    "D (anchor)", result_d["mean_gold_f1"], result_d["std_gold_f1"],
    result_d["best_gold_f1"], result_d["mean_gold_acc"]))
print("-" * 70)
diff = result_d["mean_gold_f1"] - result_a["mean_gold_f1"]
rel = (diff / result_a["mean_gold_f1"]) * 100 if result_a["mean_gold_f1"] > 0 else 0
print("Delta (D - A):  %+.4f (%+.1f%% relative)" % (diff, rel))

# ── Paired bootstrap significance test ──
print("\n" + "=" * 70)
print("PAIRED BOOTSTRAP SIGNIFICANCE TEST")
print("=" * 70)

preds_a = np.array(result_a["best_preds"])
labs_a = np.array(result_a["best_labels"])
preds_d = np.array(result_d["best_preds"])
labs_d = np.array(result_d["best_labels"])

assert len(labs_a) == len(labs_d), "Label sets must match"
n = len(labs_a)
n_boot = 10000
rng = np.random.RandomState(42)

boot_a, boot_d, boot_diff = [], [], []
for _ in range(n_boot):
    idx = rng.choice(n, size=n, replace=True)
    fa = f1_score(labs_a[idx], preds_a[idx], average="macro", zero_division=0)
    fd = f1_score(labs_d[idx], preds_d[idx], average="macro", zero_division=0)
    boot_a.append(fa)
    boot_d.append(fd)
    boot_diff.append(fd - fa)

boot_diff = np.array(boot_diff)
p_value = (boot_diff <= 0).mean()  # one-sided: H1 = D > A
ci_low = np.percentile(boot_diff, 2.5)
ci_high = np.percentile(boot_diff, 97.5)

print("Paired bootstrap (%d resamples on best-fold gold predictions):" % n_boot)
print("  Mean diff (D - A):  %+.4f" % boot_diff.mean())
print("  95%% CI of diff:     [%+.4f, %+.4f]" % (ci_low, ci_high))
print("  One-sided p-value:  %.4f  (H1: Format D > Format A)" % p_value)
if p_value < 0.001:
    print("  *** Highly significant (p < 0.001)")
elif p_value < 0.01:
    print("  ** Significant (p < 0.01)")
elif p_value < 0.05:
    print("  * Significant (p < 0.05)")
else:
    print("  Not significant (p >= 0.05)")

# ── Save ──
os.makedirs("results", exist_ok=True)
save_data = {
    "protocol": "5-fold CV format ablation, AraBERT v2, fixed T1+T2",
    "model": MODEL_NAME,
    "training_size": len(t1t2_df),
    "format_A": {
        "mean_gold_f1": result_a["mean_gold_f1"],
        "std_gold_f1": result_a["std_gold_f1"],
        "best_gold_f1": result_a["best_gold_f1"],
        "mean_gold_acc": result_a["mean_gold_acc"],
        "fold_results": result_a["fold_results"],
    },
    "format_D": {
        "mean_gold_f1": result_d["mean_gold_f1"],
        "std_gold_f1": result_d["std_gold_f1"],
        "best_gold_f1": result_d["best_gold_f1"],
        "mean_gold_acc": result_d["mean_gold_acc"],
        "fold_results": result_d["fold_results"],
    },
    "paired_bootstrap": {
        "n_resamples": n_boot,
        "mean_diff": round(float(boot_diff.mean()), 4),
        "ci_95_low": round(float(ci_low), 4),
        "ci_95_high": round(float(ci_high), 4),
        "p_value_one_sided": round(float(p_value), 4),
    },
}
with open("results/format_ablation.json", "w") as f:
    json.dump(save_data, f, indent=2)

with open("results/per_class_format_ablation.txt", "w") as f:
    f.write("FORMAT ABLATION — AraBERT v2 — T1+T2 (fixed)\n")
    f.write("=" * 60 + "\n\n")
    f.write("Format A (blind: source + MT) | Mean F1: %.4f | Best: %.4f\n" % (
        result_a["mean_gold_f1"], result_a["best_gold_f1"]))
    f.write("=" * 60 + "\n")
    f.write(result_a["per_class_report"])
    f.write("\n\n")
    f.write("Format D (anchor: kw/bm + source + MT) | Mean F1: %.4f | Best: %.4f\n" % (
        result_d["mean_gold_f1"], result_d["best_gold_f1"]))
    f.write("=" * 60 + "\n")
    f.write(result_d["per_class_report"])
    f.write("\n\n")
    f.write("Delta (D - A): %+.4f | p-value: %.4f\n" % (diff, p_value))

print("\nSaved: results/format_ablation.json")
print("Saved: results/per_class_format_ablation.txt")
print("\nTotal time: %.1f min" % ((time.time() - total_start) / 60))
