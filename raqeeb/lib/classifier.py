"""
classifier.py — Raqeeb 23-class classifier wrapper.

Loads a frozen AraBERT v2 fold checkpoint and exposes a single predict(...)
method that takes a list of (keyword, best_match, ar_ref, mt_output) tuples
and returns predicted Mizan class labels + confidences. The default deployable
released in Paper 1 is fold-0 (the best fold, single-fold Gold macro-F1 =
0.635); fold-3 (0.589) is the cross-domain / companion-lineage checkpoint.

The architecture and tokenisation are byte-for-byte identical to the
training run that produced the checkpoint, via the RaqeebEncoder wrapper.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .raqeeb_encoder import RaqeebEncoder

MODEL_HF_NAME = "aubmindlab/bert-base-arabertv02"


class _InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoder: RaqeebEncoder):
        self.df = df.reset_index(drop=True)
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return self.encoder.build_input_pair(
            keyword=str(row["keyword"]),
            best_match=str(row["best_match"]),
            ar=str(row["ar_ref"]),
            mt=str(row["mt_output"]),
        )


def _collate(batch, encoder: RaqeebEncoder):
    return encoder.tokenise_batch(batch)


class RaqeebClassifier:
    """Raqeeb 23-class classifier (AraBERT v2; default deployable = fold-0)."""

    def __init__(self, checkpoint_dir: str | Path, label_map_path: str | Path,
                 device: str = "cpu", batch_size: int = 16):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.batch_size = batch_size

        with open(label_map_path, encoding="utf-8") as f:
            self.label_to_id = json.load(f)
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.num_labels = len(self.label_to_id)

        self.encoder = RaqeebEncoder(
            model_hf_name=MODEL_HF_NAME,
            num_labels=self.num_labels,
            checkpoint_dir=self.checkpoint_dir,
        )
        self.encoder.to(device)
        self.encoder.eval()

    @torch.no_grad()
    def predict(self, items: Iterable[dict]) -> List[dict]:
        """
        items: list of dicts with keys: keyword, best_match, ar_ref, mt_output
        returns: list of dicts with keys: predicted_label, predicted_id, confidence
        """
        df = pd.DataFrame(list(items))
        ds = _InferenceDataset(df, self.encoder)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            collate_fn=lambda b: _collate(b, self.encoder),
        )

        out: List[dict] = []
        for batch in tqdm(loader, desc="classifying", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            logits = self.encoder.forward_logits(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1).cpu().numpy()
            confs = probs.max(dim=-1).values.cpu().numpy()
            for pid, c in zip(pred_ids, confs):
                out.append({
                    "predicted_id": int(pid),
                    "predicted_label": self.id_to_label[int(pid)],
                    "confidence": float(c),
                })
        return out
