# Data Sources Documentation

## Data Source 1: {Source Name}
- **Location**: {File path, database connection, API endpoint}
- **Format**: {CSV, Excel, Parquet, API JSON, SQL database}
- **Refresh Frequency**: {Real-time, daily, weekly, monthly, one-off extract}
- **Date Range**: {Start — End, or "rolling N months"}
- **Row Count (approx.)**: {Number of records}

### Schema / Column Descriptions
| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| {col}  | {type} | {description} | {example}   |

### Known Issues / Quirks
- {e.g., "GP%" header is ambiguous — could mean gross profit margin or gross profit amount}
- {e.g., NULLs in region column for online-only orders}

---

## Data Source 2: {Source Name}
_(Repeat the structure above for each additional source.)_

---

## Data Relationships
{Describe how the sources connect — join keys, foreign keys, temporal alignment, granularity mismatches.}

```
Source A (SKU_ID) ──► Source B (product_code)
Source B (date, store_id) ──► Source C (campaign_date, store_id)
```

## Data Quality Notes
- {Overall data quality observations}
- {Any known biases, survivorship bias, missing segments}
- {Data freshness concerns}
