# Business Context — FMCG SKU Performance Analysis

## Business Overview
- **Company**: NusaGoods Pte Ltd
- **Industry**: Fast-Moving Consumer Goods (FMCG) — personal care & household cleaning
- **Scale**: ~USD 120M annual revenue, 350+ active SKUs, distribution across Singapore, Malaysia, Indonesia, Thailand, and the Philippines. 8,000 retail touchpoints plus growing e-commerce (Shopee, Lazada, TikTok Shop).

## Problem Statement
The commercial team needs to identify **top- and bottom-performing SKUs** across markets and channels, and build a **predictive framework** that estimates the likelihood of success for new product launches before committing to production runs. Today, launch decisions rely on gut feel and anecdotal comparisons to past products; roughly 40 % of new SKUs are delisted within 18 months.

## Current Limitations
- Performance tracking is done in Excel pivot tables refreshed manually once a month.
- No statistical rigour — "top performer" is simply sorted by revenue, ignoring margin, velocity, or trend.
- New-launch success factors have never been formally modelled; the team debates endlessly whether packaging, price point, or marketing spend matters most.
- Data lives in three separate systems with no automated joins.

## Success Criteria
- **SC-1**: A repeatable, automated SKU scorecard that ranks SKUs on a composite metric (revenue × margin × velocity trend) refreshable weekly.
- **SC-2**: A classification model predicting new-launch survival (still active after 12 months) with ≥ 75 % AUC on held-out data.
- **SC-3**: Clear, interpretable feature-importance output so the commercial team can act on the top 5 launch-success drivers.
- **SC-4**: Dashboard or report consumable by non-technical stakeholders (Sales Directors, Country Managers).

## Constraints
- **Timeline**: MVP in 4 weeks, production version in 8 weeks.
- **Budget**: No additional SaaS spend — must use existing Python stack + cloud notebooks.
- **Technical**: Data must remain within the company's AWS VPC (ap-southeast-1). Models must be explainable (no black-box-only outputs).

## Stakeholders
| Role | Name | Interest | Decision Authority |
|------|------|----------|--------------------|
| VP Commercial | Sarah Tan | Actionable SKU insights for quarterly reviews | Approver |
| Regional Sales Director | Rizal Ahmad | Country-level SKU rankings, launch recommendations | Contributor |
| Head of Data | Wei Lin | Technical architecture, data governance | Approver |
| Brand Manager | Priya Menon | New-launch success drivers, packaging insights | Informed |
