# Data Sources — FMCG SKU Performance Analysis

## Data Source 1: SKU Master Data
- **Location**: `s3://nusagoods-data/master/sku_master.csv`
- **Format**: CSV (UTF-8, comma-delimited)
- **Refresh Frequency**: Weekly (every Monday 02:00 SGT)
- **Date Range**: All-time (includes delisted SKUs flagged as inactive)
- **Row Count (approx.)**: 1,200 rows (350 active + 850 historical)

### Schema / Column Descriptions
| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| sku_id | string | Unique product identifier | `NG-PC-00412` |
| sku_name | string | Human-readable product name | `FreshClean Bodywash 500ml Citrus` |
| category | string | Product category | `Personal Care` |
| subcategory | string | Product subcategory | `Body Wash` |
| brand | string | Brand name | `FreshClean` |
| launch_date | date | Date SKU first shipped | `2022-03-15` |
| delist_date | date | Date SKU was delisted (NULL if active) | `2024-01-10` |
| status | string | Active / Inactive | `Active` |
| pack_size_ml | int | Pack size in ml | `500` |
| rrp_usd | float | Recommended retail price (USD) | `4.50` |
| cogs_usd | float | Cost of goods sold per unit (USD) | `1.80` |
| GP% | float | **Ambiguous** — believed to be gross profit margin as a decimal (0.60 = 60 %) but some rows appear to contain absolute gross profit in USD | `0.60` or `2.70` |
| packaging_type | string | Bottle / Sachet / Tube / Pouch | `Bottle` |
| target_demo | string | Target demographic segment | `Urban Female 25-40` |

### Known Issues / Quirks
- `GP%` column is inconsistent — roughly 15 % of rows contain absolute dollar values instead of percentages. Needs cleaning logic.
- `launch_date` has 23 NULL values for legacy SKUs imported from an older system.
- `target_demo` free-text field has inconsistent formatting ("Urban Female 25-40" vs "F25-40 Urban").

---

## Data Source 2: Sales Transaction Data
- **Location**: `s3://nusagoods-data/transactions/sales_daily/`
- **Format**: Partitioned Parquet files (`year=YYYY/month=MM/day=DD/`)
- **Refresh Frequency**: Daily (T+1, available by 06:00 SGT)
- **Date Range**: 2021-01-01 to present
- **Row Count (approx.)**: ~45 million rows

### Schema / Column Descriptions
| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| txn_date | date | Transaction date | `2024-06-15` |
| sku_id | string | Foreign key to SKU master | `NG-PC-00412` |
| store_id | string | Retail outlet identifier | `SG-FP-0087` |
| country | string | ISO 3166-1 alpha-2 country code | `SG` |
| channel | string | Retail / E-commerce / Wholesale | `Retail` |
| qty_sold | int | Units sold | `12` |
| net_revenue_usd | float | Net revenue after trade discounts (USD) | `48.60` |
| promo_flag | bool | Whether the SKU was on promotion that day | `true` |
| promo_type | string | Promo mechanic (NULL if no promo) | `Buy 2 Get 1 Free` |

### Known Issues / Quirks
- E-commerce channel data only available from 2022-07 onward.
- `promo_type` has 38 unique string values with overlapping meanings (e.g., "BOGO" vs "Buy 1 Get 1" vs "B1G1").
- Returns are recorded as negative `qty_sold`; not separated into a distinct returns table.

---

## Data Source 3: Marketing Spend Data
- **Location**: `s3://nusagoods-data/marketing/spend_by_campaign.xlsx`
- **Format**: Excel (.xlsx), single sheet named "CampaignSpend"
- **Refresh Frequency**: Monthly (finance closes books by the 10th)
- **Date Range**: 2022-01 to present
- **Row Count (approx.)**: 820 rows

### Schema / Column Descriptions
| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| campaign_id | string | Unique campaign identifier | `MKT-2024-Q2-001` |
| campaign_name | string | Descriptive campaign name | `Summer Fresh Launch SG` |
| sku_ids | string | Pipe-delimited list of SKUs in campaign | `NG-PC-00412|NG-PC-00415` |
| country | string | Target market | `SG` |
| start_date | date | Campaign start | `2024-06-01` |
| end_date | date | Campaign end | `2024-06-30` |
| channel | string | Media channel (TV / Digital / In-Store / Influencer) | `Digital` |
| spend_usd | float | Total campaign spend in USD | `15000.00` |
| impressions | int | Estimated impressions (NULL for In-Store) | `2500000` |
| attributed_revenue | float | Revenue attributed to campaign (NULL if not measured) | `42000.00` |

### Known Issues / Quirks
- `sku_ids` is a pipe-delimited string — needs parsing into a proper one-to-many relationship.
- `attributed_revenue` is NULL for ~60 % of rows (attribution model only covers Digital channel).
- Spend figures for In-Store campaigns are estimates, not actuals.

---

## Data Relationships
```
SKU Master (sku_id) ─────► Sales Transactions (sku_id)
SKU Master (sku_id) ─────► Marketing Spend (sku_ids — pipe-delimited, one-to-many)
Sales Transactions (country) ──► Marketing Spend (country)
Sales Transactions (txn_date) overlaps Marketing Spend (start_date..end_date)
```

**Granularity mismatches:**
- Sales data is daily × store × SKU level.
- Marketing data is monthly × campaign × SKU-group level.
- Joining requires date-range matching and SKU list explosion.

## Data Quality Notes
- Overall completeness is ~92 % across all three sources.
- Survivorship bias: delisted SKUs have less sales history (by definition), which could bias the launch-success model if not handled.
- Currency is standardised to USD, but exchange rates are applied at different cadences (daily for sales, monthly average for marketing).
- No explicit store-level attributes (format, size, traffic) — may limit store-level analysis.
