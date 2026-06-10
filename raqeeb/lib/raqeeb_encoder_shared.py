"""
Alternative encoder wrapper that uses `delta_upload/shared.py` preprocessing:
  - Best_Match == '[OMITTED]' is STRIPPED to empty string (not kept literal)
  - Empty-anchor fallback: anchor = ar (NOT ar+mt)
  - Other fields identical to RaqeebEncoder

Used only for A/B testing against raqeeb_encoder.RaqeebEncoder.
"""
from __future__ import annotations

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn


class RaqeebEncoderShared(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 23,
                 input_format: str = "D", max_length: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.input_format = input_format
        self.num_labels = num_labels
        self.max_length = max_length

        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def build_input_pair(self, row: dict) -> tuple:
        """Shared.py Format D: strip [OMITTED] to empty; fallback is (ar, mt)."""
        ar = str(row.get("ar", "")).strip()
        mt = str(row.get("mt_output", "")).strip()
        kw = str(row.get("Keyword", "")).strip()
        bm = str(row.get("Best_Match", "")).strip()

        # shared.py convention: STRIP [OMITTED] and NaN-like values to empty string
        if bm in ("[OMITTED]", "nan", "None", "", "NaN"):
            bm = ""

        if kw and kw not in ("nan", "None", ""):
            anchor = f"{kw} {bm}".strip()
        else:
            anchor = ar

        context = f"{ar} {mt}".strip()
        return (anchor, context)

    def tokenise_batch(self, pairs: list, device) -> dict:
        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]
        encoded = self.tokenizer(
            texts_a, texts_b,
            padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls)
