"""raqeeb.data — bundled taxonomy + dataset loaders.

    from raqeeb.data import mizan, load_quanta

- `mizan`     : the 23-class Mizan error taxonomy (label→id map + per-class severity /
                parent group / TQA category from canonical_mizan_v7.json).
- `load_quanta(path=None)` : load the QUANTA expert-annotated gold pool (Mizan source
                data; ships as gold_pool_446.csv) as a pandas DataFrame.
"""
from __future__ import annotations

import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent


def _load_mizan() -> dict:
    """23-class Mizan taxonomy: label_map + (severity/group/tqa) per class when available."""
    label_map = json.loads((_DATA / "label_map.json").read_text(encoding="utf-8"))
    taxonomy = {
        "n_classes": len(label_map),
        "label_map": label_map,                       # name -> id (0..22)
        "id_to_label": {v: k for k, v in label_map.items()},
        "classes": {},
    }
    v7_path = _DATA / "canonical_mizan_v7.json"
    if v7_path.exists():
        v7 = json.loads(v7_path.read_text(encoding="utf-8"))
        for name in label_map:
            c = v7.get("classes", {}).get(name, {})
            taxonomy["classes"][name] = {
                "id": label_map[name],
                "severity_score": c.get("severity_score"),
                "group": c.get("group"),
                "tqa_category": c.get("tqa_category"),
            }
    else:  # minimal fallback
        for name, idx in label_map.items():
            taxonomy["classes"][name] = {"id": idx}
    return taxonomy


# Module-level taxonomy object (cheap JSON load at import).
mizan: dict = _load_mizan()


def load_quanta(path: str | None = None):
    """Load the QUANTA expert-annotated gold pool as a pandas DataFrame.

    Args:
        path: optional override CSV path. Defaults to the bundled
              `gold_pool_446.csv` (446-row Mizan-annotated QUANTA gold).
    """
    import pandas as pd
    csv = Path(path) if path else (_DATA / "gold_pool_446.csv")
    if not csv.exists():
        raise FileNotFoundError(f"QUANTA gold pool not found at {csv}.")
    return pd.read_csv(csv)


__all__ = ["mizan", "load_quanta"]
