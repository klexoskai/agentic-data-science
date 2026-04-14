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
- Criterion 4: Produces concrete files in `outputs/` and runnable app code in `pipeline/`.

## Constraints
- **Timeline**: MVP now; iterate after first successful run.
- **Budget**: Reuse existing project stack and local data files.
- **Technical**: Prefer simple, robust transforms and clear assumptions over complex modeling in v1.

## Input Contract (Strict)
- Future runs must treat this file (`inputs/sample/context.md`) and `inputs/sample/data_sources.md` as the only business requirements source of truth.
- If requirements are missing from these two files, the pipeline should use conservative defaults instead of inventing scope.
- Any scope expansion (for example external data enrichment) must be added explicitly to these two files first.

## Tech Stack and Fallback
- Preferred stack: Python, pandas, Plotly, Dash.
- Preferred outputs: interactive Dash frontend + chart artifact + markdown summary.
- Fallback behavior if Dash/frontend generation fails: still generate CSV projections and markdown report in `outputs/`.

## Stakeholders
| Role | Name | Interest | Decision Authority |
|------|------|----------|--------------------|
| Business Lead | TBD | Usable launch projection output for decision support | Approver |
| Technical Lead | TBD | Reliable and maintainable MVP pipeline | Approver |
| Reviewer | TBD | Surface risks and weak assumptions | Contributor |
| Output Engineer | TBD | Keep output easy to consume (chart + paragraph) | Contributor |
