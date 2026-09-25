# NPT report in Power BI

The report charts where drilling time went: NPT hours by cause, per wellbore and over time,
with the tagged remarks behind each bar.

## Data flow

```
DDR XML -> oilfield.npt (parse, tag causes) -> data/npt/*.parquet
        -> python -m oilfield.bi.export     -> data/powerbi/*.csv  (or the gold tables on Databricks)
        -> Power BI model (load_npt.pq, measures.dax) -> report pages
```

## Model

| table | grain | keys |
|---|---|---|
| `fact_npt` | one NPT activity | `wellbore_key`, `date_key`, `cause_key` |
| `fact_activity` | one reported activity (productive or not) | `wellbore_key`, `date_key` |
| `dim_wellbore` | wellbore (`F-12`, `F-1C`, ...) | `wellbore_key` |
| `dim_cause` | NPT cause, with display label and sort order | `cause_key` |
| `dim_date` | calendar day | `date_key` (yyyymmdd) |

Both fact tables relate many-to-one to the dimensions; `fact_activity` gives the denominator for
**NPT %**. Sort `dim_cause[cause_label]` by `dim_cause[sort_order]`.

## Pages

1. **Overview**: cards for NPT Hours, NPT %, NPT Events, Top Cause; stacked bar of NPT Hours by
   wellbore and cause; line of NPT Hours 7d by date.
2. **Causes**: NPT Hours by cause (bar), Avg Hours per Event by cause, Low-Confidence NPT Hours
   so reviewers can see how much of the split rests on uncertain tags.
3. **Events**: table of `fact_npt` rows (date, wellbore, cause, hours, confidence, remark) with
   wellbore, cause and date slicers; drill-through target from the other pages.

## Setup

1. `python -m oilfield.bi.export` (after `oilfield.npt.tag`).
2. In Power BI Desktop create a text parameter `NptFolder` with the absolute path of `data/powerbi`.
3. Add a blank query, paste `load_npt.pq`, name it `LoadTable`, and create one query per table:
   `= LoadTable("fact_npt")`, and so on.
4. Create the relationships listed above and the measures in `measures.dax`.

On Databricks, skip the CSVs: connect Power BI to the SQL warehouse and load the same tables
from the gold schema.
