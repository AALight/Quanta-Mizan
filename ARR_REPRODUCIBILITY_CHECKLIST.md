# Reproducibility Checklist (ACL/EMNLP/NAACL 2026)

Mapped to the ARR (Anonymous Review Review) reproducibility checklist. Items are checked from the perspective of a third-party reviewer with access to this anonymized repository.

## Section A — For all submitted papers

### A1. Did you describe the limitations of your work?
**Yes.** §7 (*Discussion → Limitations*) and Appendix H (*Limitations*).
- Gold test set is small (n = 438) and single-domain (UN Security Council)
- Bootstrap CIs overlap across all five encoders
- Meaning Shift remains the frontier class (F1 = 0.272)
- Synthetic T2 alone is substantially less sample-efficient than validated real T1
- IAA on 147 stratified instances; full annotation round planned for camera-ready

### A2. Did you discuss potential negative societal impact?
**Yes.** Appendix Q (*Broader Impact*) discusses (i) misuse risk if the classifier is treated as ground-truth quality assessment without human review, (ii) implications for translator workflows, (iii) language-data ownership.

### A3. Did you discuss Ethics?
**Yes.** Appendix P (*Ethics Statement*) covers annotator compensation, language-data provenance, LLM disclosure, and reviewer-relevant ethical considerations.

### A4. Have you read the ethics review guidelines and ensured your paper conforms?
**Yes.**

## Section B — For experimental work

### B1. Did you describe how to access the dataset?
**Yes.** The QUANTA dataset is released in `delta_upload/data/processed/` (`train_t1t2.csv`, `test_gold.csv`) and is also packaged with the toolkit (`raqeeb/data/`).

### B2. Did you provide a description of the dataset?
**Yes.** §5 (*The QUANTA Dataset*), Appendix C (*Multi-Scale Mapping and Per-Class Severity Weights*), Appendix D (*Human Evaluation Protocol and IAA*), Appendix G (*Data Audit*).

### B3. Did you provide the code, dataset, and instructions to reproduce the results?
**Yes.** This repository.
- Toolkit + CLI: `raqeeb/`
- Training scripts: `delta_upload/`
- Per-experiment scripts: `experiments/E01_cross_domain_classifier/`, `experiments/E03_L1_layer1_validation/`, `experiments/E04_xlmr_fold1_retrain/`
- Significance verification: `experiments/E04_xlmr_fold1_retrain/scripts/pairwise_significance.py`

### B4. Did you specify all the training details?
**Yes.** Appendix F (*Reproducibility*).
- Stratified 5-fold CV with `random_state = 42`
- Learning rate `5e-5`, batch size 32, 5 epochs
- AdamW optimizer, weighted cross-entropy loss
- Linear warmup of 10% of total training steps
- Mixed-precision via `torch.amp.autocast`
- Max input length 512 tokens
- Best validation-fold checkpoint selected by macro-F1
- All 5 encoders trained with identical hyperparameters

### B5. Did you specify all the hyperparameter search details?
**Yes.** No hyperparameter search was performed; learning rate `5e-5` is a standard default for transformer fine-tuning at this dataset scale. The paper explicitly states *"a fixed learning rate of 5e-5"* in §6 to ensure all encoders are compared fairly.

### B6. Did you describe how the dataset is split?
**Yes.** Stratified 5-fold CV (`StratifiedKFold(random_state=42)`) on `train_t1t2.csv` (8,594 instances), with the 438-instance Gold test set held-out throughout. The frozen splits are reproducible from the random seed; we also release `delta_upload/results/M01/<encoder>_fold[0-4].csv` per encoder per fold.

### B7. Did you discuss the experiments compute infrastructure?
**Yes.** Appendix F (*Reproducibility*) — single A40/A100 GPU, ~3 minutes per fold for AraBERT v2, ~15-20 minutes total for the 5-fold benchmark per encoder.

### B8. Did you report the number of times you ran each experiment with different random seeds?
**Yes (with caveat).** All five encoders use seeds 42-46 (one per fold via `random_state + fold`). XLM-R fold 1 was retrained at seed 43 after the original seed-43 run collapsed (gold F1 = 0.009); the retrained value (0.5499) is in `experiments/E04_xlmr_fold1_retrain/results/retrain_fold1_seed_43/`. The other 4 XLM-R folds use the original seeds (42, 44, 45, 46). This is documented in §6 footnote and Appendix F.

### B9. Did you provide error bars or other significance information?
**Yes.** Table 1 reports mean ± std over 5 folds and bootstrap 95% CI on best-fold predictions (1,000 resamples). Pairwise paired-bootstrap tests with Benjamini–Hochberg FDR correction (10,000 resamples on fold-level differences) reported in §6 and verifiable via `experiments/E04_xlmr_fold1_retrain/scripts/pairwise_significance.py`.

### B10. Did you describe what data was used for each experiment?
**Yes.** §6 and Appendix F — the 5-fold CV uses the full 8,594-instance `train_t1t2.csv`, evaluated on the 438-instance held-out Gold. Appendix I (Controlled 12-Class Ablation) uses subsets restricted to the 12 classes with T1 training support.

## Section C — For experiments using human subjects

### C1. Did you describe the annotator population?
**Yes.** Appendix D (*Human Evaluation Protocol*). Two independent native-Arabic-speaking annotators not involved in taxonomy design; both have prior NLP annotation experience.

### C2. Did you provide the full annotation interface?
**Partially.** The annotation protocol and the evaluation criteria (Layer 1 + Layer 2, bilingual EN/AR) are described in the paper's dual-annotator validation appendix. The annotation workbook templates and per-instance annotator responses are reserved for a dedicated annotation-methodology paper and are not included in this release.

### C3. Did you report inter-annotator agreement?
**Yes.** §1, §3, §6, and Appendix D — Gwet's AC1 = 0.97 on T2 synthetic core-three, AC1 = 0.83 on T1 real core-three, on 147 stratified instances. Computation script and per-question breakdown in `experiments/iaa_tier1_AB.py`.

### C4. Did you describe how annotators were compensated?
**Yes.** Appendix P (*Ethics Statement*).

## Section D — For experiments using AI assistants / LLMs

### D1. Did you disclose any use of LLMs?
**Yes.** §4.6 (*Cross-Vendor ETCA Audit*), Appendix E (*ETCA Generation Prompt and Audit Rubric*), Appendix P (*Ethics Statement*).
- T2 synthetic data generated by GPT-4o (3-shot)
- Audit performed by Claude Sonnet 4 (cross-vendor design to mitigate self-preference bias)
- Verbatim prompts released at `raqeeb/data/prompts/etca_audit_prompt.txt` and `raqeeb/data/prompts/t2_generation_prompt.txt`

### D2. Did you describe how the LLMs were prompted?
**Yes.** Appendix E describes the prompt design; the verbatim prompts are released as plain-text files in the toolkit.

### D3. Did you discuss the impact of LLM-generated data?
**Yes.** §7 (*Limitations*) — synthetic instances alone are substantially less sample-efficient than validated real data; the mixed T1+T2 pool outperforms either alone.

## Anonymization

- Author identifiers removed from `paper_v2.tex` (`\author{Anonymous}`, `[review]` mode)
- Repository hosted at `https://anonymous.4open.science/r/raqeeb-anon`
- LICENSE notice does not contain author names in the review version
- No funding acknowledgements, supervisor names, or institutional identifiers in the body

## Verification commands

```bash
# Verify pairwise significance claims
python experiments/E04_xlmr_fold1_retrain/scripts/pairwise_significance.py

# Verify the IAA AC1 on Tier 1 / Tier 2 (147 paired instances)
python experiments/iaa_tier1_AB.py

# Verify XLM-R 5-fold mean ± std with the retrained fold 1
python experiments/E04_xlmr_fold1_retrain/scripts/compute_5fold_stats.py
```
