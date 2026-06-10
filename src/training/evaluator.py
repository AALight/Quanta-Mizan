"""
evaluator.py
============
Evaluate a trained Raqeeb checkpoint on any test set.
Run this AFTER training all 5 folds and selecting the best checkpoint.

Usage:
    # Primary gold evaluation (run ONCE per model)
    python src/training/evaluator.py \\
        --checkpoint experiments/arabert_v2/flat/fold_1/ \\
        --test_file  data/processed/test_gold.csv

    # Transfer test (same checkpoint)
    python src/training/evaluator.py \\
        --checkpoint experiments/arabert_v2/flat/fold_1/ \\
        --test_file  data/processed/test_transfer.csv

Outputs:
    predictions_{test_name}.csv   in the checkpoint folder
    Updates metrics.json with gold/transfer fields
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))
from src.models.encoder import RaqeebEncoder


def load_checkpoint(ckpt_dir: Path, label_map: dict, device) -> RaqeebEncoder:
    """Load a saved Raqeeb checkpoint."""
    config = torch.load(ckpt_dir / "checkpoint" / "config.pt", map_location=device)

    model = RaqeebEncoder(
        model_name   = str(ckpt_dir / "checkpoint"),
        num_labels   = config["num_labels"],
        input_format = config["input_format"],
        max_length   = config["max_length"],
    )
    model.tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir / "checkpoint"))
    model.encoder   = AutoModel.from_pretrained(str(ckpt_dir / "checkpoint"))

    classifier_state = torch.load(
        ckpt_dir / "checkpoint" / "classifier_head.pt",
        map_location=device
    )
    model.classifier.load_state_dict(classifier_state)
    model = model.to(device)
    model.eval()
    return model


def evaluate_on_test(
    ckpt_dir:  Path,
    test_file: Path,
    bootstrap_n: int = 1000,
    bootstrap_ci: float = 0.95,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Test file:  {test_file}")

    # Load label map
    label_map_path = BASE / "data" / "processed" / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}

    # Load model
    model = load_checkpoint(ckpt_dir, label_map, device)

    # Load test data
    df_test = pd.read_csv(test_file, encoding="utf-8-sig")
    print(f"Test rows: {len(df_test)} · Classes: {df_test['Sub-Subtype'].nunique()}/23")

    # Build dataset
    from src.training.trainer import RaqeebDataset, collate_fn

    # Filter to known classes (transfer set may have subset)
    known_mask = df_test["Sub-Subtype"].isin(label_map)
    if not known_mask.all():
        n_unknown = (~known_mask).sum()
        print(f"⚠️  {n_unknown} rows with unknown labels skipped")
        df_test = df_test[known_mask].reset_index(drop=True)

    def make_collate(m, d):
        def fn(batch):
            return collate_fn(batch, m, d)
        return fn

    dataset = RaqeebDataset(df_test, model, label_map)
    loader  = DataLoader(
        dataset, batch_size=32, shuffle=False,
        collate_fn=make_collate(model, device), num_workers=0
    )

    # Predict
    all_preds, all_labels, all_confs, all_ids = [], [], [], []
    model.eval()
    with torch.no_grad():
        for encoded, labels, ids in loader:
            logits = model(**encoded)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)
            confs  = probs.max(dim=1).values

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            all_ids.extend(ids)

    # Metrics
    from sklearn.metrics import (
        f1_score, accuracy_score, classification_report
    )

    macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy    = accuracy_score(all_labels, all_preds)

    # Bootstrap confidence intervals on Macro-F1
    rng  = np.random.RandomState(42)
    n    = len(all_labels)
    boot = []
    for _ in range(bootstrap_n):
        idx = rng.choice(n, n, replace=True)
        boot.append(
            f1_score(
                np.array(all_labels)[idx],
                np.array(all_preds)[idx],
                average="macro", zero_division=0
            )
        )
    alpha    = 1 - bootstrap_ci
    ci_lower = float(np.percentile(boot, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))

    print(f"\n{'='*50}")
    print(f"Macro-F1    : {macro_f1:.4f}  [{ci_lower:.4f}, {ci_upper:.4f}] 95% CI")
    print(f"Weighted-F1 : {weighted_f1:.4f}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"{'='*50}\n")

    # Per-class F1
    test_classes = [id2label[l] for l in sorted(set(all_labels))]
    report = classification_report(
        [id2label[l] for l in all_labels],
        [id2label[p] for p in all_preds],
        labels=test_classes,
        output_dict=True, zero_division=0
    )
    per_class = {cls: round(report.get(cls, {}).get("f1-score", 0.0), 4)
                 for cls in label_map}

    # Save predictions CSV
    test_name = test_file.stem  # e.g. "test_gold" or "test_transfer"
    pred_df = pd.DataFrame({
        "sample_id":       all_ids,
        "true_label":      [id2label[l] for l in all_labels],
        "pred_label":      [id2label[p] for p in all_preds],
        "pred_confidence": [round(float(c), 4) for c in all_confs],
        "true_label_id":   all_labels,
        "pred_label_id":   all_preds,
    })
    pred_path = ckpt_dir / f"predictions_{test_name}.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"Predictions saved: {pred_path}")

    # Update metrics.json
    metrics_path = ckpt_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    if test_name == "test_gold":
        metrics["gold_macro_f1"]          = round(macro_f1, 4)
        metrics["gold_weighted_f1"]        = round(weighted_f1, 4)
        metrics["gold_accuracy"]           = round(accuracy, 4)
        metrics["gold_macro_f1_ci_lower"]  = round(ci_lower, 4)
        metrics["gold_macro_f1_ci_upper"]  = round(ci_upper, 4)
        metrics["per_class_f1_gold"]       = per_class
    elif test_name == "test_transfer":
        metrics["transfer_macro_f1"]       = round(macro_f1, 4)
        metrics["transfer_weighted_f1"]    = round(weighted_f1, 4)
        metrics["transfer_accuracy"]       = round(accuracy, 4)
        metrics["transfer_n_classes"]      = len(test_classes)
        metrics["transfer_note"]           = "12/23 classes only — not comparable to gold"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"metrics.json updated: {metrics_path}")

    return macro_f1, weighted_f1, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Raqeeb checkpoint on a test set"
    )
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to fold directory e.g. experiments/arabert_v2/flat/fold_1/")
    parser.add_argument("--test_file",   required=True,
                        help="Path to test CSV e.g. data/processed/test_gold.csv")
    parser.add_argument("--bootstrap_n", type=int, default=1000,
                        help="Bootstrap iterations for CI (default 1000)")
    args = parser.parse_args()

    evaluate_on_test(
        ckpt_dir     = BASE / args.checkpoint,
        test_file    = BASE / args.test_file,
        bootstrap_n  = args.bootstrap_n,
    )


def paired_bootstrap_test(
    predictions_a: Path,
    predictions_b: Path,
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap significance test comparing two models on the same test set.
    Tests whether Model A is significantly better than Model B on Macro-F1.

    Usage:
        python src/training/evaluator.py --compare \\
            experiments/arabert_v2/flat/fold_1/predictions_gold.csv \\
            experiments/camelbert_msa/flat/fold_0/predictions_gold.csv

    Args:
        predictions_a: predictions_gold.csv for model A (claimed better)
        predictions_b: predictions_gold.csv for model B (baseline)
        n_iterations:  bootstrap iterations (10,000 recommended for papers)
        seed:          random seed for reproducibility

    Returns:
        dict with p_value, significant, delta_macro_f1, ci_lower, ci_upper
    """
    from sklearn.metrics import f1_score
    import numpy as np

    df_a = pd.read_csv(predictions_a, encoding="utf-8-sig")
    df_b = pd.read_csv(predictions_b, encoding="utf-8-sig")

    # Align on sample_id
    df = df_a[["sample_id", "true_label", "pred_label"]].merge(
        df_b[["sample_id", "pred_label"]].rename(columns={"pred_label": "pred_b"}),
        on="sample_id"
    )

    true   = df["true_label"].values
    pred_a = df["pred_label"].values
    pred_b = df["pred_b"].values

    f1_a = f1_score(true, pred_a, average="macro", zero_division=0)
    f1_b = f1_score(true, pred_b, average="macro", zero_division=0)
    observed_delta = f1_a - f1_b

    rng = np.random.RandomState(seed)
    n   = len(true)
    count_greater = 0
    deltas = []

    for _ in range(n_iterations):
        idx      = rng.choice(n, n, replace=True)
        boot_a   = f1_score(true[idx], pred_a[idx], average="macro", zero_division=0)
        boot_b   = f1_score(true[idx], pred_b[idx], average="macro", zero_division=0)
        boot_d   = boot_a - boot_b
        deltas.append(boot_d)
        if boot_d > 0:
            count_greater += 1

    p_value  = 1 - (count_greater / n_iterations)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    result = {
        "model_a":          str(predictions_a),
        "model_b":          str(predictions_b),
        "macro_f1_a":       round(f1_a, 4),
        "macro_f1_b":       round(f1_b, 4),
        "delta_macro_f1":   round(observed_delta, 4),
        "p_value":          round(p_value, 4),
        "significant_p05":  bool(p_value < 0.05),
        "ci_95_lower":      round(ci_lower, 4),
        "ci_95_upper":      round(ci_upper, 4),
        "n_iterations":     n_iterations,
        "interpretation":   (
            f"Model A is {'significantly' if p_value < 0.05 else 'NOT significantly'} "
            f"better than Model B (p={p_value:.4f}, delta={observed_delta:+.4f}, "
            f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}])"
        )
    }

    print(f"\n{'='*55}")
    print(f"PAIRED BOOTSTRAP SIGNIFICANCE TEST")
    print(f"{'='*55}")
    print(f"Model A  : {Path(str(predictions_a)).parts[-4]}")
    print(f"Model B  : {Path(str(predictions_b)).parts[-4]}")
    print(f"Macro-F1 A = {f1_a:.4f}")
    print(f"Macro-F1 B = {f1_b:.4f}")
    print(f"Delta      = {observed_delta:+.4f}")
    print(f"p-value    = {p_value:.4f}  {'✅ significant (p<0.05)' if p_value < 0.05 else '❌ NOT significant'}")
    print(f"95% CI     = [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"{'='*55}\n")

    return result


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()

    # Single model evaluation
    parser.add_argument("--checkpoint",  help="Fold dir for single model eval")
    parser.add_argument("--test_file",   help="Test CSV path")
    parser.add_argument("--bootstrap_n", type=int, default=1000)

    # Significance test between two models
    parser.add_argument("--compare",     nargs=2, metavar=("PRED_A", "PRED_B"),
        help="Two predictions_gold.csv files to compare")
    parser.add_argument("--sig_n",       type=int, default=10000,
        help="Bootstrap iterations for significance test (default 10000)")
    parser.add_argument("--output",      help="Save significance result to JSON")

    args = parser.parse_args()

    if args.compare:
        result = paired_bootstrap_test(
            predictions_a = BASE / args.compare[0],
            predictions_b = BASE / args.compare[1],
            n_iterations  = args.sig_n,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to {args.output}")
    elif args.checkpoint and args.test_file:
        evaluate_on_test(
            ckpt_dir    = BASE / args.checkpoint,
            test_file   = BASE / args.test_file,
            bootstrap_n = args.bootstrap_n,
        )
    else:
        parser.print_help()
