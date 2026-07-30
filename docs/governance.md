# Governance summary

## Advisory-only purpose

InspectIQ ranks supplied candidates for human review. It does not establish a violation, create an enforcement target, or automatically initiate inspection or enforcement.

## Human-review workflow

Reviewers receive an advisory score, rank, explanations, and historical evidence. They record status, decision, reasoning, notes, review time, override, and escalation requirement. A reviewer may override model priority; model output is not a mandate.

## Unknown outcomes and frozen predictions

Incomplete outcome retrieval remains unknown rather than negative. The 2023 candidate batch is frozen and unlabelled in this workflow. Future outcome evaluation must join complete labels under a new, auditable process; frozen prediction artifacts and hashes protect against silent reranking.

## Monitoring and incident response

Pipeline `PASS` means the monitoring computation completed its configured integrity checks. Operational health is separately `HEALTHY`, `WARNING`, or `CRITICAL`. A `WARNING` can reflect expected temporal accumulation or score-distribution change and is not a current performance claim. Data drift measures feature/population change; score drift measures output distribution change; performance drift needs complete outcome labels.

Review-exposure diagnostics are descriptive and do not prove discrimination. No protected demographic attributes are available for outcome-fairness analysis. Monitoring artifacts, templates, and ranking outputs carry hashes for auditability.

## Retraining and operational stop conditions

Retraining should follow a documented review of data quality, outcome completeness, population change, validation evidence, and operational harms—not a single drift metric. Stop or restrict use when artifact integrity fails, required review controls are unavailable, inputs are out of contract, monitoring is critical without a documented disposition, or an operator proposes autonomous enforcement.

This document is governance guidance, not legal or compliance certification.
