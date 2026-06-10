"""
encoder.py
==========
HuggingFace encoder + classification head for Raqeeb 23-class classifier.

Supports three input formats:
  A: [CLS] ar [SEP] mt_output [SEP]          (blind — text pair only)
  B: [CLS] ar [SEP] mt_output [SEP] KW [SEP] BM [SEP]  (anchor-aware)
  C: [CLS] mt_output [SEP]                    (error span only)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional


class RaqeebEncoder(nn.Module):
    """
    Fine-tunable encoder for Arabic MT error classification.

    Args:
        model_name:   HuggingFace model ID
        num_labels:   number of output classes (23 for Raqeeb)
        input_format: 'A' | 'B' | 'C'
        max_length:   max tokenised sequence length (256 recommended)
        dropout:      classifier head dropout rate
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 23,
        input_format: str = "A",
        max_length: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert input_format in ("A", "B", "C", "D"), \
            f"input_format must be A, B, C, or D — got {input_format}"

        self.model_name    = model_name
        self.num_labels    = num_labels
        self.input_format  = input_format
        self.max_length    = max_length

        self.encoder   = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size    = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def build_input_pair(self, row: dict) -> tuple:
        """
        Construct input pair(s) for a single sample based on input_format.

        Returns:
            tuple of (text_a, text_b) for sentence-pair tokenization.
            text_b may be None for Format C (single sentence).

        Format A: text_a=ar, text_b=mt_output
        Format B: text_a=ar, text_b="mt_output keyword"
        Format C: text_a=mt_output, text_b=None
        Format D: text_a="keyword best_match", text_b="ar mt_output"
                  The classifier sees the anchor pair as sentence A
                  and the full context as sentence B.
        """
        ar  = str(row.get("ar", "")).strip()
        mt  = str(row.get("mt_output", "")).strip()
        kw  = str(row.get("Keyword", "")).strip()
        bm  = str(row.get("Best_Match", "")).strip()

        # Normalize NaN/None Best_Match to [OMITTED] for consistency
        # Training data uses "[OMITTED]" for omission errors; gold may have NaN
        if not bm or bm.lower() == "nan" or bm == "None":
            bm = "[OMITTED]"

        if self.input_format == "A":
            return (ar, mt)

        if self.input_format == "B":
            extra = mt
            if kw and kw != "nan":
                extra += f" {kw}"
            if bm and bm != "nan":
                extra += f" {bm}"
            return (ar, extra)

        if self.input_format == "C":
            return (mt, None)

        if self.input_format == "D":
            # Sentence A: the anchor pair (what changed)
            anchor = kw if kw and kw != "nan" else ""
            match = bm if bm and bm != "nan" else ""
            text_a = f"{anchor} {match}".strip()
            # Sentence B: the full context
            text_b = f"{ar} {mt}".strip()
            if not text_a:
                text_a = ar  # fallback if no anchor
                text_b = mt
            return (text_a, text_b)

        return (ar, mt)

    # Keep backward compat for any code that calls build_input_text
    def build_input_text(self, row: dict) -> str:
        a, b = self.build_input_pair(row)
        if b:
            return f"{a} [SEP] {b}"
        return a

    def tokenise_batch(self, pairs: list, device) -> dict:
        """
        Tokenise a list of (text_a, text_b) pairs using proper sentence-pair encoding.

        This produces correct [CLS] text_a [SEP] text_b [SEP] with token_type_ids
        distinguishing sentence A (0) from sentence B (1).
        """
        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]

        if all(b is None for b in texts_b):
            # Format C: single sentence
            encoded = self.tokenizer(
                texts_a,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            # Format A/B: sentence pair
            encoded = self.tokenizer(
                texts_a,
                texts_b,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        return {k: v.to(device) for k, v in encoded.items()}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            logits: (batch_size, num_labels)
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        cls_repr = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        logits   = self.classifier(cls_repr)
        return logits

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return [CLS] embeddings without classification head (for analysis)."""
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        with torch.no_grad():
            outputs = self.encoder(**kwargs)
        return outputs.last_hidden_state[:, 0, :]
