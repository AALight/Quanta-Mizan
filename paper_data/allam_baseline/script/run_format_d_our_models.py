"""
Run Format D with lab protocol (70-15-15, 10 seeds, 5 epochs)
for CAMeLBERT-MSA, CAMeLBERT-Mix, ALLaM zero-shot, ALLaM 5-shot.

Usage:
    cd /path/to/delta_upload
    python delta/run_format_d_our_models.py

Estimated time: ~2.5 hours
  MSA: 10 seeds x ~3 min = 30 min
  Mix: 10 seeds x ~3 min = 30 min
  ALLaM zero-shot: 438 inferences = 30 min
  ALLaM 5-shot: 438 inferences = 30 min
"""

import json
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
PROJ = Path(os.getcwd())
sys.path.insert(0, str(PROJ))
print("Working directory: %s" % PROJ)

from src.models.encoder import RaqeebEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: %s" % device)

# ── Config ──────────────────────────────────────────────────────

INPUT_FORMAT = "D"
MAX_LENGTH = 512
LR = 5e-5
BATCH_SIZE = 32
EPOCHS = 5
NUM_SEEDS = 10

# ── Load data ───────────────────────────────────────────────────

with open("data/processed/label_map.json") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}
LABEL_LIST = list(label_map.keys())

train_df = pd.read_csv("data/processed/train_t2.csv", encoding="utf-8-sig")
gold_df = pd.read_csv("data/processed/test_gold.csv", encoding="utf-8-sig")

# Fixed 70-15-15 split (same as AraBERT v2 run)
train_data, devtest = train_test_split(
    train_df, test_size=0.30, random_state=42, stratify=train_df["Sub-Subtype"]
)
dev_data, internal_test = train_test_split(
    devtest, test_size=0.50, random_state=42, stratify=devtest["Sub-Subtype"]
)
print("Split: train=%d | dev=%d | internal_test=%d" % (len(train_data), len(dev_data), len(internal_test)))


# ── Dataset ─────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, df, model, label_map):
        self.df = df.reset_index(drop=True)
        self.model = model
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pair = self.model.build_input_pair(row.to_dict())
        label = self.label_map[row["Sub-Subtype"]]
        return pair, label


def collate_fn(batch, model, device):
    pairs, labels = zip(*batch)
    encoded = model.tokenise_batch(list(pairs), device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    return encoded, labels


# ── Train one seed ──────────────────────────────────────────────

def train_one_seed(seed, model_name):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = RaqeebEncoder(
        model_name=model_name, num_labels=len(label_map),
        input_format=INPUT_FORMAT, max_length=MAX_LENGTH,
    ).to(device)

    labels_array = train_data["Sub-Subtype"].map(label_map).values
    present = np.unique(labels_array)
    w = compute_class_weight("balanced", classes=present, y=labels_array)
    cw = np.ones(len(label_map))
    for c, wt in zip(present, w):
        cw[c] = wt
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float).to(device))

    def make_collate(m, d):
        def fn(batch): return collate_fn(batch, m, d)
        return fn

    train_loader = DataLoader(
        PairDataset(train_data, model, label_map),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=make_collate(model, device), num_workers=0
    )
    dev_loader = DataLoader(
        PairDataset(dev_data, model, label_map),
        batch_size=BATCH_SIZE * 2, shuffle=False,
        collate_fn=make_collate(model, device), num_workers=0
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler('cuda')

    best_dev_f1 = -1
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for encoded, labels in train_loader:
            with torch.amp.autocast('cuda'):
                logits = model(**encoded)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for encoded, labels in dev_loader:
                with torch.amp.autocast('cuda'):
                    logits = model(**encoded)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        dev_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model = model.to(device)
    return model, best_dev_f1


def evaluate_on_gold(model):
    model.eval()
    def make_collate(m, d):
        def fn(batch): return collate_fn(batch, m, d)
        return fn

    gold_loader = DataLoader(
        PairDataset(gold_df, model, label_map),
        batch_size=BATCH_SIZE * 2, shuffle=False,
        collate_fn=make_collate(model, device), num_workers=0
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for encoded, labels in gold_loader:
            with torch.amp.autocast('cuda'):
                logits = model(**encoded)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    gold_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    gold_acc = accuracy_score(all_labels, all_preds)
    return gold_f1, gold_acc, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════
# PART 1: ENCODER MODELS (MSA + Mix) — 10 seeds each
# ══════════════════════════════════════════════════════════════════

import statistics

ENCODER_MODELS = [
    ("CAMeLBERT-MSA", "CAMeL-Lab/bert-base-arabic-camelbert-msa"),
    ("CAMeLBERT-Mix", "CAMeL-Lab/bert-base-arabic-camelbert-mix"),
]

os.makedirs("results", exist_ok=True)
all_model_results = {}

for display_name, model_id in ENCODER_MODELS:
    print("\n" + "=" * 60)
    print("FORMAT D — %s (10 seeds, 5 epochs)" % display_name)
    print("=" * 60)

    seed_results = []
    start_total = time.time()

    for seed in range(NUM_SEEDS):
        start = time.time()
        model, dev_f1 = train_one_seed(seed, model_id)
        gold_f1, gold_acc, preds, labels = evaluate_on_gold(model)
        elapsed = time.time() - start

        seed_results.append({
            "seed": seed, "dev_f1": round(dev_f1, 4),
            "gold_f1": round(gold_f1, 4), "gold_acc": round(gold_acc, 4),
        })
        print("  Seed %d: dev=%.4f gold=%.4f (%.1f min)" % (seed, dev_f1, gold_f1, elapsed / 60))

        # Save best seed per-class report
        if gold_f1 >= max(r["gold_f1"] for r in seed_results):
            best_preds = preds
            best_labels = labels

        del model
        torch.cuda.empty_cache()

    dev_f1s = [r["dev_f1"] for r in seed_results]
    gold_f1s = [r["gold_f1"] for r in seed_results]

    print("\n%s Results:" % display_name)
    print("  Mean Gold F1: %.4f +/- %.4f" % (statistics.mean(gold_f1s), statistics.stdev(gold_f1s)))
    print("  Best Gold F1: %.4f (seed %d)" % (max(gold_f1s), gold_f1s.index(max(gold_f1s))))
    print("  Time: %.1f min" % ((time.time() - start_total) / 60))

    # Per-class report for best seed
    report = classification_report(
        [id2label[l] for l in best_labels],
        [id2label[p] for p in best_preds],
        labels=LABEL_LIST, zero_division=0, digits=3
    )
    fname = "results/per_class_format_d_%s.txt" % display_name.lower().replace("-", "_").replace(" ", "_")
    with open(fname, "w") as f:
        f.write("Format D %s (best seed)\n\n%s" % (display_name, report))
    print("  Saved: %s" % fname)

    all_model_results[display_name] = {
        "model_id": model_id,
        "seed_results": seed_results,
        "mean_gold_f1": round(statistics.mean(gold_f1s), 4),
        "std_gold_f1": round(statistics.stdev(gold_f1s), 4),
        "best_gold_f1": round(max(gold_f1s), 4),
        "mean_dev_f1": round(statistics.mean(dev_f1s), 4),
    }


# ══════════════════════════════════════════════════════════════════
# PART 2: ALLaM ZERO-SHOT + 5-SHOT (Format D prompts)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ALLaM-7B ZERO-SHOT + 5-SHOT (Format D)")
print("=" * 60)

TAXONOMY = """ERROR TAXONOMY (23 classes with definitions):

ACCURACY ERRORS:
- Meaning Shift: translation conveys incorrect or distorted meaning
- Total Omission: a source element is completely missing from MT
- Partial Translation: named entity or key term only partially translated
- Name Entity Error: named entity incorrectly translated or transliterated
- Terminology Substitution: domain-specific term replaced with wrong term
- Literal Translation: word-for-word translation ignoring Arabic idiom
- Hypernym for Hyponym: broader term used instead of specific term
- Hyponym for Hypernym: more specific term used instead of broader term

MORPHOLOGICAL FLUENCY ERRORS:
- Tanween Omission: Arabic nunation omitted where required
- Gender Disagreement: masculine/feminine agreement violation
- Definiteness Shift: incorrect use of definite article
- Perfective to Progressive: perfective aspect rendered as progressive
- Progressive to Perfective: progressive aspect rendered as perfective

SYNTACTIC FLUENCY ERRORS:
- Tense Shift Under Negation: tense changes incorrectly under negation
- Wrong Structure: sentence structure does not follow Arabic grammar
- Wrong Word Order: constituents in wrong syntactic order
- Active to Passive Voice: active voice rendered as passive
- Passive to Active Voice: passive voice rendered as active

TERMINOLOGY ERRORS:
- Noun to Adjective: noun rendered as adjective
- Adjective to Noun: adjective rendered as noun
- Register Mismatch: formal/informal register inconsistency
- Spelling Error: orthographic error in Arabic output
- Invalid Pattern: morphologically impossible Arabic word form"""

# Build 5-shot bank
bank = (train_df.groupby("Sub-Subtype", group_keys=False)
        .apply(lambda x: x.sample(min(3, len(x)), random_state=42)))
diverse_classes = ["Meaning Shift", "Tanween Omission", "Active to Passive Voice",
                   "Register Mismatch", "Total Omission"]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading ALLaM-7B...")
    MODEL_ID = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    allam_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    allam_model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("ALLaM loaded.")
    ALLAM_LOADED = True
except Exception as e:
    print("ALLaM loading failed: %s" % e)
    ALLAM_LOADED = False

if ALLAM_LOADED:
    def predict(prompt, max_new_tokens=30):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            outputs = allam_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                           temperature=0.01, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    def parse_prediction(raw):
        raw_lower = raw.lower().strip().strip("\"'.*\n")
        for prefix in ["the error is", "error class:", "error type:", "answer:", "class:"]:
            if raw_lower.startswith(prefix):
                raw_lower = raw_lower[len(prefix):].strip()
        for label in LABEL_LIST:
            if label.lower() in raw_lower:
                return label
        first_line = raw.split("\n")[0].strip().lower()
        for label in LABEL_LIST:
            if label.lower() in first_line:
                return label
        return "Unknown"

    labels_str = "\n".join("- %s" % l for l in LABEL_LIST)

    for mode in ["zero_shot", "few_shot_5"]:
        print("\n--- ALLaM %s (Format D) ---" % mode)
        results = []
        start = time.time()

        for i, row in gold_df.iterrows():
            ar = str(row.get("ar", "")).strip()
            mt = str(row.get("mt_output", "")).strip()
            kw = str(row.get("Keyword", "")).strip()
            bm = str(row.get("Best_Match", "")).strip()
            true_label = str(row.get("Sub-Subtype", "Unknown"))

            if mode == "zero_shot":
                prompt = """You are an expert Arabic computational linguist specialising in machine translation error analysis.

Your task: Given an Arabic reference sentence and its machine translation output, along with a specific keyword that changed, classify the TYPE of linguistic shift that occurred.

%s

RULES:
- Respond with EXACTLY ONE class name from the list above
- No explanation, no extra text
- Just the class name

Keyword (correct form in reference): %s
Best_Match (form found in MT output): %s
Reference (ar): %s
MT Output: %s

What type of linguistic shift caused '%s' to become '%s'?""" % (TAXONOMY, kw, bm, ar, mt, kw, bm)

            else:
                examples_text = ""
                for cls in diverse_classes:
                    cls_ex = bank[bank["Sub-Subtype"] == cls]
                    if len(cls_ex) > 0:
                        ex = cls_ex.iloc[0]
                        examples_text += """--- Example (%s) ---
Keyword: %s -> Became: %s
Reference: %s
MT Output: %s
Answer: %s

""" % (cls, str(ex.get("Keyword", ""))[:80], str(ex.get("Best_Match", ""))[:80],
       str(ex.get("ar", ""))[:150], str(ex.get("mt_output", ""))[:150], cls)

                prompt = """You are an expert Arabic computational linguist specialising in machine translation error analysis.

%s

EXAMPLES (5 diverse):
%s
--- NOW CLASSIFY ---
Keyword (correct form in reference): %s
Best_Match (form found in MT output): %s
Reference (ar): %s
MT Output: %s

What type of linguistic shift caused '%s' to become '%s'?

Answer with ONLY the class name, nothing else.""" % (TAXONOMY, examples_text, kw, bm, ar, mt, kw, bm)

            raw = predict(prompt)
            pred = parse_prediction(raw)
            results.append({"true_label": true_label, "pred_label": pred, "raw_response": raw})

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print("  %d/%d (%.1f/s)" % (i + 1, len(gold_df), (i + 1) / elapsed))

        df_res = pd.DataFrame(results)
        macro_f1 = f1_score(df_res["true_label"], df_res["pred_label"],
                            average="macro", zero_division=0, labels=LABEL_LIST)
        acc = accuracy_score(df_res["true_label"], df_res["pred_label"])
        unknown = (df_res["pred_label"] == "Unknown").sum()

        print("\n%s: Macro-F1=%.4f Acc=%.4f Unknown=%d/%d" % (mode, macro_f1, acc, unknown, len(df_res)))

        report = classification_report(df_res["true_label"], df_res["pred_label"],
                                       labels=LABEL_LIST, zero_division=0, digits=3)
        print(report)

        out_dir = PROJ / "experiments" / "allam_format_d" / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_dir / "predictions_test_gold.csv", index=False, encoding="utf-8-sig")
        with open(out_dir / "metrics.json", "w") as f:
            json.dump({"model": "ALLaM-7B", "mode": mode, "format": "D",
                       "gold_macro_f1": round(macro_f1, 4), "gold_accuracy": round(acc, 4),
                       "unknown_rate": round(unknown / len(df_res), 4)}, f, indent=2)

        with open("results/per_class_allam_%s_format_d.txt" % mode, "w") as f:
            f.write("ALLaM %s Format D\n\n%s" % (mode, report))

        all_model_results["ALLaM %s" % mode] = {
            "gold_macro_f1": round(macro_f1, 4),
            "gold_accuracy": round(acc, 4),
            "unknown_rate": round(unknown / len(df_res), 4),
        }


# ══════════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("FORMAT D — ALL MODELS COMPARISON")
print("=" * 60)

print("\n%-25s %15s %15s %12s" % ("Model", "Mean Gold F1", "Best Gold F1", "Method"))
print("-" * 70)

# AraBERT v2 (already done)
print("%-25s %15s %15s %12s" % ("AraBERT v2", "0.3641+/-0.0146", "0.3871", "10 seeds"))

for name, res in all_model_results.items():
    if "seed_results" in res:
        print("%-25s %11.4f+/-%.4f %15.4f %12s" % (
            name, res["mean_gold_f1"], res["std_gold_f1"], res["best_gold_f1"], "10 seeds"))
    else:
        print("%-25s %15.4f %15s %12s" % (name, res["gold_macro_f1"], "—", "single run"))

# Save all
with open("results/format_d_all_models.json", "w") as f:
    json.dump(all_model_results, f, indent=2, default=str)

print("\nAll results saved to results/format_d_all_models.json")
