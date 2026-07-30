# Interview walkthrough

## 30-second overview

InspectIQ is an advisory ML ranking system for a supplied OSHA inspection candidate batch. I built a leakage-safe chronological workflow from cached source data through feature lineage, validation, local MLflow, deterministic batch scoring, a Streamlit review UI, monitoring, Docker, CI, and release checks. The selected Random Forest captured 36 positives in the top 10% of labelled 2022 validation rows versus 19 for the baseline; 2023 has no outcomes in the workflow.

## 2-minute overview

I first validated that `activity_nr`, `open_date`, and a serious/willful/repeat label were usable. Incomplete violation retrieval remained unknown, not negative. I used 2020–2021 for training, 2022 for validation, and held 2023 as an unlabelled candidate period. Historical industry features only use prior records. Eight models were compared to a training-only industry-rate baseline; a Random Forest was selected on ranking metrics. Calibration was studied but rejected because neither calibrated method qualified. The resulting score ranks candidates for reviewers; it is uncalibrated and cannot trigger enforcement. I then added local MLflow, deterministic artifacts, Streamlit, monitoring, and CI/local release validation.

## 5-minute technical walkthrough

Data is cached and manifest-validated, then the complete labelled subset becomes an immutable snapshot. Chronological splits prevent future data from entering earlier features or evaluation. The baseline establishes a review-capacity comparison. Features encode inspection descriptors and strictly prior industry history; establishment history is omitted because the key is not defensible. Model selection prioritizes Recall@10% before precision, lift, PR-AUC, Brier, and simplicity. The selected model scored 2023 only after it was frozen. Monitoring compares feature and score populations, while future outcome templates make later evaluation auditable. CI uses synthetic fixtures because runtime artifacts are intentionally ignored.

## Architecture, governance, and trade-offs

Explain the diagrams in [architecture](architecture.md): cached source → snapshot → chronology/features → selection/calibration → MLflow → frozen ranking → dashboard/monitoring → release checks. The key trade-off is useful prioritization versus conservative claims: the project preserves unknowns, avoids protected-attribute claims, excludes a weak establishment key, and prefers no calibration over misleading probability language.

## Interview questions and honest answers

1. **Why not random train/test split?** Time-ordered validation better simulates scoring later candidates and blocks future information leakage.
2. **Why Recall@10%?** Review capacity is finite; Recall@10% measures positives found in the top 10% budget.
3. **Why compare against a baseline?** It shows whether modelling adds value over a transparent training-only industry-rate rule.
4. **Why Random Forest?** It provided the selected 2022 validation ranking result among eight candidates while remaining practical for tabular mixed features.
5. **Why not accuracy?** A fixed review queue needs ranking concentration, not a thresholded aggregate classification score.
6. **Why retain an uncalibrated model?** Sigmoid and isotonic did not meet the meaningful-improvement policy; selecting them would misrepresent probability quality.
7. **Why are scores not probabilities?** The final selected output is uncalibrated, and historical selection bias further limits probability interpretation.
8. **How was leakage prevented?** Historical features are strictly prior; same-day, future, validation, and candidate feedback paths are blocked.
9. **Why exclude unknown outcomes?** Treating incomplete retrieval as negative would create false labels and bias the target.
10. **Why no establishment-level features?** Available fields did not provide a defensible stable establishment key, so the feature was omitted.
11. **What does PSI measure?** It summarizes distribution shift between reference and current populations; it is not a performance metric.
12. **Why is score drift not performance drift?** Without current outcomes, output changes cannot prove better or worse correctness.
13. **How are frozen predictions evaluated later?** Hashes and manifests preserve the ranked output, then complete future outcomes can be joined under an auditable evaluation process.
14. **Why are artifacts ignored?** They can be large, local, sensitive, and reproducible; source control keeps code/config/reports rather than silently committing runtime state.
15. **How does CI work without model artifacts?** CI validates code, contracts, documentation, and synthetic temporary fixtures; local mode validates real frozen artifacts.
16. **How does Docker receive artifacts?** Compose mounts host `data`, `artifacts`, and `reports` read-only; the image does not download or generate them.
17. **How is human review enforced?** The product provides review status, rationale, override, and escalation fields; code has no automatic enforcement path.
18. **What are the fairness limitations?** No protected demographic attributes are available, and review-exposure diagnostics are descriptive rather than outcome-fairness conclusions.
19. **What would trigger retraining?** Complete outcome evidence plus governed review of data quality, shift, validation, harms, and operational need—not drift alone.
20. **What would you change with more time?** Add governed out-of-time evaluation once outcomes are complete, improve coverage, assess lawful fairness data, and test alternative transparent ranking models.
21. **What was the hardest engineering problem?** Maintaining artifact integrity across resumable caching, deterministic ranking, and release checks while keeping unknowns safe.
22. **What mistake did you avoid?** I avoided describing uncalibrated scores as probabilities or treating missing violation retrieval as negative.

## Closing framing

The project is strongest as a disciplined ML systems demonstration: validated retrospective evidence, explicit chronology and lineage, useful review tooling, and clear boundaries around outcomes, fairness, and enforcement.
