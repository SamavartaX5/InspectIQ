# Data card

## Dataset summary and scope

InspectIQ uses 3,000 retrieved California OSHA inspections across 2020–2024 for feasibility and data-foundation work. Complete inspection-level labels are available for 2,100 rows: 552 positive and 1,548 negative. The remaining 900 outcomes are unknown and excluded. This is not a sample of every California workplace or all OSHA activity.

## Source endpoints and entity

The source is the U.S. Department of Labor OSHA inspection and violation endpoints. The inspection entity is `activity_nr`; violation records join through the same key. `open_date` determines chronology.

## Sampling, coverage, and early-year limitation

The default feasibility selection is year-balanced across the requested period, targeting approximately equal rows per calendar year where data exists. Earlier downloader behaviour could concentrate a bounded chronological fetch in the earliest dates; the current selection strategy addresses this by retrieving years separately and retaining deterministic chronological order. It remains a bounded, California-only operational sample rather than population coverage.

## Label construction and quality controls

A row is positive when it has at least one non-deleted Serious, Willful, or Repeat violation. Deleted violation rows are excluded. Missing/incomplete outcome retrieval and unknown violation types are safely excluded rather than labelled negative. Validation checks include identifier uniqueness, required columns, valid dates, categorical handling, and manifest/hash integrity.

## Chronology, fields, and leakage

Complete labels are split into 2020–2021 training (1,200), 2022 validation (600), and a 2023 candidate period (300) that has no target in the workflow. Candidate dates and `activity_nr` are excluded from model features. Retained features include inspection descriptors, establishment-size proxy, open month, and prior industry counts/rates/status. Establishment history is omitted because no defensible key is available.

## Bias, privacy, storage, and use

Historical inspection data reflects prior inspection and recording processes, so selection bias remains. The project has no protected demographic attributes for outcome-fairness evaluation. Generated raw/processed data and artifacts are ignored from Git to limit accidental sharing and keep large, reproducible operational state out of source control. Appropriate use is advisory candidate prioritization with review; inappropriate use includes asserting liability, autonomous enforcement, or generalizing to every workplace.

## Reproducibility and caching

Day 0 caches are manifest-backed and resumable. Compatible inspection snapshots and completed violation batches are reused; incomplete retrieval remains unknown. Reproduction commands and their network behaviour appear in [reproducibility](reproducibility.md).
