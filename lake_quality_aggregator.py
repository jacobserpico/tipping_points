"""
Author: @Jacob Serpico
"""

from __future__ import annotations
from pathlib import Path
from io import StringIO
import re
import numpy as np
import pandas as pd

# File paths
data_dir = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data")
agg_path = data_dir / "aggregate_water_quality" / "aggregate_water_quality__1990-2024.csv"
wp_path = data_dir / "Water_Parameters.csv"

# Helper functions
def read_csv(path):
    for enc in ("utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin-1", "utf-8"):
        try:
            return pd.read_csv(path, sep=None, engine="python", dtype=str, encoding=enc)
        except Exception:
            continue
    txt = path.read_bytes().decode("latin-1", errors="replace")
    return pd.read_csv(StringIO(txt), sep=None, engine="python", dtype=str)


def normalize_colname(name):
    s = re.sub(r"\.+$", "", str(name)).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def normalize_columns(df):
    return df.rename(columns={c: normalize_colname(c) for c in df.columns})


def to_float_series(s):
    s2 = s.fillna("").astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")


def parse_dt(date_col, time_col = None):
    # Series with a stable index
    sdate = pd.Series(date_col, copy=True)
    sdate = sdate.fillna("").astype(str).str.strip()

    if time_col is not None:
        stime = pd.Series(time_col, index=sdate.index).fillna(
            "").astype(str).str.strip()
        has_time = sdate.str.contains(r"\d{1,2}:\d{2}")
        combined = sdate.where(has_time, (sdate + " " + stime).str.strip())
    else:
        combined = sdate

    dt = pd.to_datetime(combined, errors="coerce", utc=False)
    if isinstance(dt, pd.DatetimeIndex):
        dt = pd.Series(dt, index=sdate.index)
    return dt


def iso3_to_name(iso3):
    m = {
        "US": "United States of America",
        "CA": "Canada",
        "CAN": "Canada",
        "MEX": "Mexico",
        "URY": "Uruguay",
        "BRA": "Brazil",
        "ARG": "Argentina",
        "CHL": "Chile",
        "GBR": "United Kingdom",
        "DEU": "Germany",
        "FRA": "France",
        "ESP": "Spain",
        "ITA": "Italy",
        "CHN": "China",
        "IND": "India",
        "AUS": "Australia",
        "GRC": "Greece",
        "HRV": "Croatia",
        "AUT": "Austria",
        "NLD": "Netherlands",
        "IRL": "Ireland",
        "SWE": "Sweden",
        "EST": "Estonia",
        "NOR": "Norway",
        "FIN": "Finland",
    }

    s = str(iso3 or "").strip().upper()
    return m.get(s, iso3)


# Load my existing aggregate data
agg = read_csv(agg_path)
agg = normalize_columns(agg)

agg["latitude"] = to_float_series(
    agg.get("latitude",  pd.Series(index=agg.index)))
agg["longitude"] = to_float_series(
    agg.get("longitude", pd.Series(index=agg.index)))
agg_dt = parse_dt(agg.get("sample_date", pd.Series(index=agg.index)),
                  agg.get("sample_time", pd.Series(index=agg.index)))
agg["dt_r15"] = agg_dt.dt.round("15min")
agg["lat_r4"] = agg["latitude"].round(4)
agg["lon_r4"] = agg["longitude"].round(4)

# Load water parameters
wp = read_csv(wp_path)
wp = normalize_columns(wp)

need = ["latitude", "longitude", "sample_date", "country", "lakename"]
missing = [c for c in need if c not in wp.columns]
if missing:
    raise SystemExit(
        f"Water parameters dataset is missing: {missing}")

wp["latitude"] = to_float_series(wp["latitude"])
wp["longitude"] = to_float_series(wp["longitude"])
wp["country_name"] = wp["country"].map(iso3_to_name)

wp_dt = parse_dt(wp["sample_date"])
wp["sample_date_only"] = wp_dt.dt.date.astype(str)
wp["sample_time_only"] = wp_dt.dt.time.astype(str)
wp["dt_r15"] = wp_dt.dt.round("15min")
wp["lat_r4"] = wp["latitude"].round(4)
wp["lon_r4"] = wp["longitude"].round(4)

# Trim to aggregate's year span
if agg["dt_r15"].notna().any():
    y0, y1 = int(agg["dt_r15"].dt.year.min()), int(agg["dt_r15"].dt.year.max())
    wp = wp[wp_dt.dt.year.between(y0, y1, inclusive="both")]

# Map measurement columns to GEMStat codes
COLMAP = {
    "dissolvedphosphorus_ug_l": ("TDP",  "ug/L", "Dissolved Phosphorus"),
    "dissolvednitrogen_ug_l":  ("TDN",  "ug/L", "Dissolved Nitrogen"),
    "chlorophyll_a_ug_l":      ("CHLA", "ug/L", "Chlorophyll-a"),
    "temperature_c":           ("TEMP", "C",    "Temperature"),
}
present_cols = [c for c in COLMAP if c in wp.columns]
if not present_cols:
    raise SystemExit(
        "No recognized measurement columns found in Water Parameters dataset.")

id_cols = ["latitude", "longitude", "country_name", "lakename",
           "sample_date_only", "sample_time_only", "dt_r15", "lat_r4", "lon_r4"]
long = wp[id_cols + present_cols].melt(
    id_vars=id_cols, value_vars=present_cols, var_name="wp_field", value_name="value_raw"
)
long = long[long["value_raw"].astype(str).str.strip().ne("")]
if long.empty:
    raise SystemExit(
        "Water Parameters dataset contains no non-empty values for the selected variables.")

long["parameter_code"] = long["wp_field"].map(lambda c: COLMAP[c][0])
long["unit"] = long["wp_field"].map(lambda c: COLMAP[c][1])
long["parameter_long_name"] = long["wp_field"].map(lambda c: COLMAP[c][2])

v = long["value_raw"].astype(str).str.replace(",", ".", regex=False)
long["value"] = pd.to_numeric(v, errors="coerce")
long = long.dropna(subset=["value"])

# Match to the existing stations in the aggregate dataset
pos_map = (
    agg.dropna(subset=["lat_r4", "lon_r4"])[
        ["lat_r4", "lon_r4", "gems_station_number"]]
    .drop_duplicates()
)
long = long.merge(pos_map, on=["lat_r4", "lon_r4"], how="left")

unmatched = long["gems_station_number"].isna()
if unmatched.any():
    def make_ext(row):
        cn = str(row["country_name"] or "").upper().replace(" ", "")
        return f"EXT_{cn}_{row['lat_r4']:.4f}_{row['lon_r4']:.4f}"
    long.loc[unmatched, "gems_station_number"] = long.loc[unmatched].apply(
        make_ext, axis=1)

# Check and remove duplicates
agg_key = agg[["parameter_code", "lat_r4", "lon_r4", "dt_r15"]].copy()
agg_key["parameter_code"] = agg_key["parameter_code"].astype(str).str.upper()
agg_key = agg_key.dropna(
    subset=["parameter_code", "lat_r4", "lon_r4", "dt_r15"]).drop_duplicates()

long["parameter_code"] = long["parameter_code"].astype(str).str.upper()
incoming_key = long[["parameter_code", "lat_r4", "lon_r4", "dt_r15"]].copy()

dup_mask = incoming_key.merge(
    agg_key,
    on=["parameter_code", "lat_r4", "lon_r4", "dt_r15"],
    how="left",
    indicator=True
)["_merge"].eq("both").values

n_dups = int(dup_mask.sum())
if n_dups:
    print(
        f"Duplicate observations matching existing aggregate (dropped): {n_dups:,}")
long = long.loc[~dup_mask].copy()

before = len(long)
long = long.drop_duplicates(subset=[
                            "gems_station_number", "parameter_code", "dt_r15", "lat_r4", "lon_r4", "value"])
after = len(long)
if before != after:
    print(
        f"Internal duplicates within Water Parameters (dropped): {before - after:,}")

if long.empty:
    raise SystemExit("Nothing to append after duplicate filtering.")

# Shape dataset for analysis
out_cols = [
    "gems_station_number", "country_name", "water_type", "lake_name",
    "latitude", "longitude", "sample_date", "sample_time", "depth",
    "parameter_code", "parameter_long_name",
    "analysis_method_code", "value_flags", "value", "unit", "data_quality",
    "source_category"
]

out_rows = pd.DataFrame({
    "gems_station_number": long["gems_station_number"].astype(str),
    "country_name": long["country_name"],
    "water_type": "Lake station",
    "lake_name": long["lakename"],
    "latitude": long["latitude"],
    "longitude": long["longitude"],
    "sample_date": long["sample_date_only"],
    "sample_time": long["sample_time_only"],
    "depth": np.nan,
    "parameter_code": long["parameter_code"],
    "parameter_long_name":long["parameter_long_name"],
    "analysis_method_code": "",
    "value_flags": "",
    "value": long["value"],
    "unit": long["unit"],
    "data_quality": "",
    "source_category": "water_parameters"
})

for c in out_cols:
    if c not in out_rows.columns:
        out_rows[c] = pd.NA
out_rows = out_rows[out_cols]

# Append
combined = pd.concat([agg[out_cols], out_rows], ignore_index=True)

sort_cols = [c for c in ["country_name", "water_type", "gems_station_number",
                         "sample_date", "parameter_code"] if c in combined.columns]
if sort_cols:
    combined = combined.sort_values(
        sort_cols, kind="mergesort").reset_index(drop=True)

out_path = agg_path.with_name(agg_path.stem + "__with_water_parameters.csv")
combined.to_csv(out_path, index=False)

# Print useful summary :-)
n_added = len(out_rows)
n_total = len(combined)
n_new_entities = out_rows["gems_station_number"].str.startswith("EXT_").sum()
n_reused = n_added - n_new_entities

print(f"Wrote: {out_path}")
print(
    f"Appended rows: {n_added:,} (matched existing station IDs: {n_reused:,}; new synthetic IDs: {n_new_entities:,})")
print(f"Duplicates dropped: {n_dups:,}")
print(
    f"Combined rows: {n_total:,}, Entities: {combined['gems_station_number'].nunique():,}")
