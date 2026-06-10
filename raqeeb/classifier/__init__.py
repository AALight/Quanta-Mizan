"""
raqeeb.classifier — 23-class Mizan classifier (AraBERT v2 fold-3 best).

Public API
----------
    from raqeeb.classifier import RaqeebClassifier

Quick start
-----------
    clf = RaqeebClassifier(
        checkpoint_dir="path/to/arabert_v2_fold3",
        label_map_path="path/to/label_map.json",
    )
    predictions = clf.predict([
        {"keyword": "السلام", "best_match": "الأمن",
         "ar_ref": "...", "mt_output": "..."},
    ])
    for p in predictions:
        print(p["predicted_label"], p["confidence"])

Paper 1 reports a 5-fold mean Gold macro-F1 = 0.614 ± 0.017 (95% CI
[0.563, 0.675] from 1,000-resample bootstrap on best-fold predictions;
best fold is fold-0). The released checkpoint is the fold-3 single-fold
model (single-fold macro-F1 = 0.589), selected for downstream cross-domain
reproducibility rather than for being the best-scoring fold. See §6 for details.

For training a NEW classifier on your own domain, use the multi-level
training script at experiments/E06_multilevel_classifier/scripts/train_multilevel.py.
"""
from raqeeb.lib.classifier import RaqeebClassifier

__all__ = ["RaqeebClassifier"]
