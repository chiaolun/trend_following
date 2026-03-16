-- Load CSI futures data into DuckDB
-- Run: duckdb csi_data.duckdb < load_data.sql

CREATE OR REPLACE TABLE spec AS
WITH raw AS (
    SELECT
        lower(replace(replace(replace(trim(column0[:position(':' in column0) - 1]), chr(13), ''), chr(9), ''), ' ', '_')) AS key,
        replace(replace(trim(column0[position(':' in column0) + 1:]), chr(13), ''), chr(9), '') AS value,
        regexp_extract(filename, '[^/]+$') AS origin_filename
    FROM read_csv('data/*.Specs.txt', header=false, sep='\x01', filename=true)
    WHERE trim(column0) != ''
)
PIVOT raw ON key USING first(value) GROUP BY origin_filename;

CREATE OR REPLACE TABLE dailies AS
SELECT
    strptime(Date::VARCHAR, '%Y%m%d')::DATE AS date,
    NumericDeliveryMonth AS numeric_delivery_month,
    strptime(ExpiryDate::VARCHAR, '%Y%m%d')::DATE AS expiry_date,
    Open AS open, High AS high, Low AS low, Close AS close,
    Volume AS volume, OI AS oi,
    TotalVolume AS total_volume, TotalOI AS total_oi,
    regexp_extract(filename, '[^/]+$') AS origin_filename,
    regexp_extract(filename, '([A-Za-z]+)[^/]*$', 1) AS symbol
FROM read_csv('data/*.csv', header=true, filename=true);
