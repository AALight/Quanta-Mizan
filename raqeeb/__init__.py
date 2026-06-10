"""
Raqeeb — Anchor-Constrained Detection of Fine-Grained Arabic MT Errors.

A toolkit for fine-grained 23-class Arabic MT error classification with
anchor-constrained validation, cross-vendor LLM audit, and dual-layer
human evaluation support.

Public API
----------
Top-level convenience class:
    from raqeeb import Raqeeb

Per-component access:
    from raqeeb.trivet import Trivet              # validator + clitic cascade + ESV
    from raqeeb.classifier import RaqeebClassifier  # 23-class AraBERT v2 model
    from raqeeb.etca import ETCAAuditor             # cross-vendor Claude audit

Quick start
-----------
    from raqeeb import Raqeeb

    r = Raqeeb()
    candidates = r.detect(
        ar_ref="...arabic reference...",
        mt_output="...mt output...",
    )
    for c in candidates:
        print(c['predicted_label'], c['confidence'])

See examples/raqeeb_demo.py and examples/trivet_demo.py for runnable end-to-end
usage. See README.md for installation, CLI, and cross-domain adaptation.
"""

__version__ = "1.0.0"

# Top-level public API — re-exports from submodules so users can do
#   from raqeeb import Trivet, RaqeebClassifier, ETCAAuditor
# without traversing internal submodule paths. Heavy deps (torch/transformers)
# load lazily inside the classes, so these imports stay cheap.
from raqeeb.trivet import Trivet, calculate_mpqs, clitic_aware_match  # noqa: E402
from raqeeb.classifier import RaqeebClassifier  # noqa: E402
from raqeeb.etca import ETCAAuditor  # noqa: E402

__all__ = [
    "__version__",
    "Trivet",
    "RaqeebClassifier",
    "ETCAAuditor",
    "calculate_mpqs",
    "clitic_aware_match",
]
