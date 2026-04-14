# Data Sources Documentation

## Data Source 1: Launch Tracker (`launch_tracker25_matched.csv`)
- **Location**: `data/launch_tracker25_matched.csv`
- **Format**: CSV
- **Refresh Frequency**: Ad-hoc extract
- **Date Range**: Launch planning months (current tracker period)
- **Row Count (approx.)**: ~60 rows

### Schema / Column Descriptions
| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| SKU Code | string/int | Primary launch SKU key for integration | `105905` |
| SKU Launch Month | string/date | Planned launch month | `2024-07` |
| Category | string | Product category | `Oral Care` |
| Market Specific | string | Market/country label | `Australia` |
| Brand | string | Brand | `Betadine` |

### Known Issues / Quirks
- Column names include spaces and punctuation; normalize before modeling.
- Some matching fields are text-based and may require fuzzy matching.

---

## Data Source 2: Internal Actuals (COPA) (`copa.csv`)
- **Location**: `data/copa.csv`
- **Format**: CSV
- **Refresh Frequency**: Ad-hoc extract
- **Date Range**: Multi-year historical actuals
- **Row Count (approx.)**: ~15k rows

### Schema / Column Descriptions
| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| PRODUCT_ID | string/int | Product key used for join | `105904` |
| PERIOD | date | Accounting period | `2025-10-01` |
| COUNTRY_CODE | string | Country/entity code | `AU09` |
| SALES_QUANTITY | float | Actual quantity sold | `24` |
| REVENUE_GOODS | float | Revenue component | `497.7600` |

### Known Issues / Quirks
- Figures are chunky (wide schema, many adjustment fields).
- Requires careful metric-definition rules before aggregation.

---

## Data Source 3: TM1 Sales Pivot (`tm1_qty_sales_pivot.csv`)
- **Location**: `data/tm1_qty_sales_pivot.csv`
- **Format**: CSV
- **Refresh Frequency**: Ad-hoc extract
- **Date Range**: Recent monthly actuals
- **Row Count (approx.)**: ~1.6k rows

### Schema / Column Descriptions
| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| sku_id | string/int | Join key for SKU actual performance | `101643` |
| period | string/date | Month | `2024-01` |
| quantity | string/float | Actual quantity | `"1,212"` |
| net_sales | string/float | Actual net sales | `"-7,187"` |

### Known Issues / Quirks
- Numeric values are stored as strings with separators.
- Includes unnamed export index column.

---

## Data Source 4: P&L Data (`pnl2425_volume_extracts_matched.csv`)
- **Location**: `data/pnl2425_volume_extracts_matched.csv`
- **Format**: CSV
- **Refresh Frequency**: Ad-hoc extract
- **Date Range**: FY24-FY25 planning windows
- **Row Count (approx.)**: ~700 rows

### Schema / Column Descriptions
| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| matched_SKU_ID | string/int | Join key to SKU-level actuals/planning | `105905` |
| project_name | string | Launch project identifier | `Rusty` |
| Market | string | Market/country | `Australia` |
| forecast_volume_y1 | float | Internal forecast volume (Y1) | `37468.0` |
| forecast_net_sales_y1 | float | Internal forecast net sales (Y1) | `345707.49` |

### Known Issues / Quirks
- Missing `matched_SKU_ID` in some rows.
- Wide forecast schema (`_y1`, `_y2`, `_y3`) needs reshaping.

---

## Data Relationships
```
Launch Tracker (SKU Code, SKU Launch Month, Category, Market Specific, Brand)
    ├──► COPA (PRODUCT_ID) for historical actual performance signals
    ├──► TM1 Pivot (sku_id) for monthly realized quantity/net sales curves
    └──► P&L (matched_SKU_ID) for prior internal forecast-vs-actual comparisons
```

## Data Quality Notes
- Core internal joins require a canonical SKU mapping layer across `SKU Code`, `PRODUCT_ID`, `sku_id`, and `matched_SKU_ID`.
- Month-level projection requires strict period normalization (`YYYY-MM` and date alignment across files).
- For first run, defer IQVIA/Euromonitor/WHO/ANP/SLOB integration to keep pipeline lightweight and stable.
