"""
anchor_extractor.py
===================
Learned Arabic anchor extractor — replaces the rule-based OCS cascade.

Given a reference keyword and an MT output sentence, finds the span in
the MT output that is the morphological variant of the keyword (Best_Match).

This is a span extraction task (same architecture as Arabic QA):
  Input:  [CLS] keyword [SEP] mt_output [SEP]
  Output: start/end token positions of best_match in mt_output

Trained on QUANTA keyword/best_match pairs.
Evaluated against OCS rule-based baseline.
Results → Paper Table 5, §4.5.

Usage:
    python src/training/train_anchor_extractor.py --fold 0
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional


class ArabicAnchorExtractor(nn.Module):
    """
    BERT-based span extractor for Arabic anchor word detection.

    Fine-tuned on (keyword, mt_output, best_match) triplets from QUANTA.
    Predicts the start and end token position of best_match in mt_output.

    Args:
        model_name: HuggingFace model ID (default: CAMeLBERT-MSA)
        max_length:  max tokenised sequence length
    """

    def __init__(
        self,
        model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa",
        max_length: int = 256,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length  = max_length

        self.encoder   = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size    = self.encoder.config.hidden_size

        # Two linear heads: one for start position, one for end position
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head   = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        """
        Returns:
            start_logits: (batch, seq_len) — probability of each token being span start
            end_logits:   (batch, seq_len) — probability of each token being span end
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs     = self.encoder(**kwargs)
        sequence    = outputs.last_hidden_state          # (batch, seq_len, hidden)
        start_logits = self.start_head(sequence).squeeze(-1)  # (batch, seq_len)
        end_logits   = self.end_head(sequence).squeeze(-1)    # (batch, seq_len)
        return start_logits, end_logits

    def build_input(self, keyword: str, mt_output: str) -> dict:
        """Tokenise a (keyword, mt_output) pair for span extraction."""
        encoded = self.tokenizer(
            keyword,
            mt_output,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        return encoded

    def predict(self, keyword: str, mt_output: str, device) -> str | None:
        """
        Predict the best_match span in mt_output for a given keyword.

        Returns:
            The predicted best_match string, or None if no span found.
        """
        self.eval()
        encoded = self.build_input(keyword, mt_output)
        offset_mapping = encoded.pop("overflow_to_sample_mapping", None)
        offsets = encoded.pop("offset_mapping")[0]

        inputs = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            start_logits, end_logits = self(**inputs)

        start_idx = torch.argmax(start_logits[0]).item()
        end_idx   = torch.argmax(end_logits[0]).item()

        if end_idx < start_idx:
            return None

        # Map token positions back to character offsets in mt_output
        # offset_mapping gives (char_start, char_end) per token
        # Tokens in second sequence (mt_output) start after [SEP]
        try:
            char_start = offsets[start_idx][0].item()
            char_end   = offsets[end_idx][1].item()
            return mt_output[char_start:char_end].strip()
        except Exception:
            return None
