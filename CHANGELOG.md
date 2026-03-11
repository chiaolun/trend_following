# Change Log

- **2026-03-11 11:15 UTC** — Switched data source from parquet/CSV to DuckDB (`csi_data.duckdb`). Replaced `contracts.csv` metadata with `spec` table, replaced `csi_data.parquet` with `dailies` table. Prices are now in market format (no `mult` adjustment needed). Multiplier derived from `contract_size`/`units`. Symbol mapping handles spec "2" suffixes (e.g. BO2 → BO). Filtered `_Spot.csv` duplicates. Removed contract blacklist and complex filtering (DuckDB is pre-curated to 39 contracts).
