# Business Context — MVP First Run

## Business Overview
- **Company**: iNova Pharma Pte Ltd
- **Industry**: FMCG / OTC healthcare
- **Scale**: Multi-market portfolio; pilot will focus on available internal CSV data only.

## Problem Statement
Build a lightweight end-to-end pipeline that, for a hypothetical launch input (`launch month`, `category`, `market`, `brand`):
1) retrieves up to 3 similar historical SKU launches, and
2) produces a simple 12-month projected sales trend.

Also produce a short paragraph explaining the projection logic and evidence used.

## Current Limitations
- Data is split across multiple internal extracts with different SKU key names.
- No single reusable workflow currently exists for launch similarity + projection.
- External market/seasonality sources are not yet integrated in a stable way.

## Success Criteria
- Criterion 1: Pipeline runs end-to-end without manual intervention.
- Criterion 2: Returns similar-SKU set (max 3) and a 12-month projection output.
- Criterion 3: Includes a concise evidence-based narrative.

## Constraints
- **Timeline**: MVP now; iterate after first successful run.
- **Budget**: Reuse existing project stack and local data files.
- **Technical**: Prefer simple, robust transforms and clear assumptions over complex modeling in v1.

## Stakeholders
| Role | Name | Interest | Decision Authority |
|------|------|----------|--------------------|
| Business Lead | TBD | Usable launch projection output for decision support | Approver |
| Technical Lead | TBD | Reliable and maintainable MVP pipeline | Approver |
| Reviewer | TBD | Surface risks and weak assumptions | Contributor |
| Output Engineer | TBD | Keep output easy to consume (chart + paragraph) | Contributor |
