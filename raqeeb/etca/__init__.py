"""
raqeeb.etca — Cross-vendor LLM-as-judge audit (ETCA-Lite).

Public API
----------
    from raqeeb.etca import ETCAAuditor

ETCA (Error Taxonomy Cross-vendor Audit) uses Claude Sonnet 4 as the judge
to validate candidate (keyword, best_match) anchor pairs against the Mizan
taxonomy. The cross-vendor design (GPT-4o generates T2 synthetic; Claude
audits) mitigates self-preference bias.

Quick start
-----------
    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

    from raqeeb.etca import ETCAAuditor
    auditor = ETCAAuditor()
    results = auditor.audit([
        {"sub_subtype": "Terminology Substitution",
         "keyword": "السلام", "best_match": "الأمن",
         "ar_ref": "...", "mt_output": "..."},
    ])
    for r in results:
        print(r["dqi"], r["etca_accepted"])

The verbatim Claude judge prompt is at:
    raqeeb/data/prompts/etca_audit_prompt.txt

The verbatim GPT-4o T2 generation prompt is at:
    raqeeb/data/prompts/t2_generation_prompt.txt

Cost: ~$0.001-0.003 per candidate audited (Claude Sonnet 4 pricing).
"""
from raqeeb.lib.etca_auditor import audit_candidates, load_prompt


class ETCAAuditor:
    """Stateless wrapper around the ETCA audit function for ergonomic use."""

    def __init__(self, model: str = "claude-sonnet-4-5",
                 dqi_min: float = 0.75,
                 max_tokens: int = 1024,
                 temperature: float = 0.0):
        self.model = model
        self.dqi_min = dqi_min
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_template = load_prompt()

    def audit(self, candidates: list[dict]) -> list[dict]:
        """Run the ETCA audit on a list of candidate dicts.

        Each candidate must have: sub_subtype, keyword, best_match, ar_ref, mt_output.
        Returns the same list with added: anchor_validity, phenomenon_clarity,
        arabic_naturalness, collateral_severity, seed_recommendation, dqi,
        etca_accepted, justification.

        Requires ANTHROPIC_API_KEY in env. If anthropic library or key are missing,
        prints warning and adds 'etca_skipped': True per candidate.
        """
        return audit_candidates(
            candidates,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            dqi_min=self.dqi_min,
        )

    def __repr__(self) -> str:
        return f"ETCAAuditor(model={self.model!r}, dqi_min={self.dqi_min})"


__all__ = ["ETCAAuditor"]
