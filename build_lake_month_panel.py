"""
Author: @ Jacob Serpico
"""

from pathlib import Path
from io import StringIO
import re
import numpy as np
import pandas as pd

# File paths
data_dir = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data")
agg_path = data_dir / "aggregate_water_quality" / "aggregate_water_quality__1990-2024__with_water_parameters.csv"
land_use_path = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data/land_use_data/v1/lake_landuse_allyears.csv")
out_dir = data_dir / "panels"
out_dir.mkdir(parents=True, exist_ok=True)

# Time window
start_year, end_year = 1992, 2022

# Land use settings
default_radius = 1500 # which radius becomes the 'prop_agri'
include_grassland = True # how to derive prop_agri if it's not present
radii_limiter = None  # [500,1000,1500,3000] or None for "all present"

def read_any_csv(p):
    try:
        return pd.read_csv(p, dtype=str, low_memory=False)
    except Exception:
        pass
    for enc in ("utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin-1", "utf-8"):
        try:
            return pd.read_csv(p, sep=None, engine="python", dtype=str)
        except Exception:
            continue
    txt = p.read_bytes().decode("latin-1", errors="replace")
    return pd.read_csv(StringIO(txt), sep=None, engine="python", dtype=str)

def norm_name(s):
    s = re.sub(r"\.+$", "", str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def norm_cols(df):
    return df.rename(columns={c: norm_name(c) for c in df.columns})

def to_float(s):
    s2 = s.fillna("").astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")

def attach_entity_id(df):
    d = df.copy()
    if "gems_station_number" in d.columns and d["gems_station_number"].notna().any():
        d["entity_id"] = d["gems_station_number"].astype(str).str.strip()
        if "lake_name" in d.columns:
            d["lake_name"] = d["lake_name"].astype(str).str.strip()
    else:
        d["entity_id"] = d["lake_name"].astype(str).str.strip()
    return d

def norm_unit(u):
    s = str(u or "").strip().lower().replace("µ", "u")
    s = re.sub(r"\s+", "", s)
    if s in {"m", "meter", "metre", "meters", "metres"}:
        return "m"
    if s == "cm":
        return "cm"
    if "mg/l" in s:
        return "mg/L"
    if "ug/l" in s or "μg/l" in s:
        return "ug/L"
    if s in {"c", "degc", "°c"}:
        return "C"
    if "ntu" in s:
        return "NTU"
    return s or ""

def standardize(df, code):
    d = df[df["parameter_code"].str.upper().eq(code)].copy()
    if d.empty:
        return d
    d["unit_std"] = d["unit"].map(norm_unit)
    if code == "TP":  # ug/L
        d = d[d["unit_std"].isin(["ug/L", "mg/L"])]
        mg = d["unit_std"].eq("mg/L")
        d.loc[mg, "value"] = d.loc[mg, "value"] * 1000.0
    elif code == "TN":  # mg/L
        d = d[d["unit_std"].isin(["mg/L", "ug/L"])]
        ug = d["unit_std"].eq("ug/L")
        d.loc[ug, "value"] = d.loc[ug, "value"] / 1000.0
    elif code == "TRANS":  # m
        d = d[d["unit_std"].isin(["m", "cm"])]
        cm = d["unit_std"].eq("cm")
        d.loc[cm, "value"] = d.loc[cm, "value"] / 100.0
        m_is_cm = d["unit_std"].eq("m") & d["value"].between(30, 250)
        d.loc[m_is_cm, "value"] = d.loc[m_is_cm, "value"] / 100.0
        d = d[(d["value"] >= 0) & (d["value"] <= 30)]
    elif code == "CHLA":  # ug/L
        d = d[d["unit_std"].eq("ug/L")]
    elif code == "TEMP":  # C
        d = d[d["unit_std"].eq("C")]
    return d

def tsi_components(sd_m=None, chla_ugL=None, tp_ugL=None):
    out = {}
    if (sd_m is not None) and np.isfinite(sd_m) and sd_m > 0:
        out["TSI_SD"] = 60.0 - 14.41*np.log(sd_m)
    if (chla_ugL is not None) and np.isfinite(chla_ugL) and chla_ugL > 0:
        out["TSI_CHLA"] = 9.81*np.log(chla_ugL) + 30.6
    if (tp_ugL is not None) and np.isfinite(tp_ugL) and tp_ugL > 0:
        out["TSI_TP"] = 14.42*np.log(tp_ugL) + 4.15
    return out

# Aggreagte data loader
agg = read_any_csv(agg_path)
agg = norm_cols(agg)

need = {"gems_station_number", "lake_name", "latitude",
        "longitude", "parameter_code", "unit", "value", "sample_date"}
missing = [c for c in need if c not in agg.columns]
if missing:
    raise SystemExit(f"Aggregate missing columns: {missing}")

agg["latitude"] = to_float(agg["latitude"])
agg["longitude"] = to_float(agg["longitude"])
agg["value"] = to_float(agg["value"])
agg["sample_date"] = pd.to_datetime(
    agg["sample_date"], errors="coerce", utc=False)
agg = agg.dropna(subset=["sample_date", "value"])
agg = attach_entity_id(agg)
agg["year"] = agg["sample_date"].dt.year
agg["month"] = agg["sample_date"].dt.month

parts = [standardize(agg, code)
         for code in ["TP", "TN", "TRANS", "CHLA", "TEMP"]]
clean = pd.concat([p for p in parts if not p.empty], ignore_index=True)

# Monthly medians per entity
mo = (clean[(clean["year"] >= start_year) & (clean["year"] <= end_year)]
      .groupby(["entity_id", "year", "month", "parameter_code"], observed=True)["value"]
      .median().reset_index())

wide = (mo.pivot_table(index=["entity_id", "year", "month"], columns="parameter_code",
                       values="value", aggfunc="median")
        .reset_index())

pos = (agg.groupby("entity_id", as_index=False)
          .agg(latitude=("latitude", "median"),
               longitude=("longitude", "median"),
               lake_name=("lake_name", lambda s: s.dropna().mode().iloc[0] if s.dropna().size else "")))

panel = wide.merge(pos, on="entity_id", how="left")

for c in ["TP", "TN", "TRANS", "CHLA", "TEMP"]:
    if c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

def read_landuse_csv_strict(p):
    try:
        return pd.read_csv(p, sep=",", dtype=str, low_memory=False, quotechar='"', encoding="utf-8")
    except Exception:
        pass
    try:
        return pd.read_csv(p, sep=",", dtype=str, engine="python", quotechar='"', encoding="utf-8")
    except Exception:
        pass
    print("warn: skipping malformed rows in land_use CSV (on_bad_lines='skip')")
    return pd.read_csv(p, sep=",", dtype=str, engine="python", quotechar='"',
                       encoding="latin-1", on_bad_lines="skip")

lu = read_landuse_csv_strict(land_use_path)
lu = norm_cols(lu)

if "lake_id" not in lu.columns or "year" not in lu.columns:
    raise SystemExit("Land-use file needs lake_id and year.")

if "buffer_m" in lu.columns:
    lu["buffer_m"] = pd.to_numeric(lu["buffer_m"], errors="coerce")
    if radii_limiter:
        lu = lu[lu["buffer_m"].isin([float(r) for r in radii_limiter])]

if "prop_agri" not in lu.columns:
    frac_cols = [c for c in lu.columns if c.startswith("frac_")]
    if not frac_cols:
        raise SystemExit(
            "Land-use file lacks prop_agri and frac_* to derive it.")
    lu[frac_cols] = lu[frac_cols].apply(pd.to_numeric, errors="coerce")
    crops = lu["frac_crops"] if "frac_crops" in lu.columns else 0.0
    if include_grassland and "frac_grassland" in lu.columns:
        prop_agri = crops + lu["frac_grassland"]
    else:
        prop_agri = crops
    lu["prop_agri"] = prop_agri.clip(0, 1)

lu["year"] = pd.to_numeric(lu["year"], errors="coerce").astype("Int64")

if "buffer_m" in lu.columns:
    piv = (lu.pivot_table(index=["lake_id", "year"], columns="buffer_m",
                          values="prop_agri", aggfunc="mean")
           .reset_index())
    ren = {}
    for c in piv.columns:
        if isinstance(c, (int, float)) and np.isfinite(c):
            ren[c] = f"prop_agri_r{int(round(float(c)))}"
    piv = piv.rename(columns=ren)
else:
    piv = lu[["lake_id", "year", "prop_agri"]].copy()

if "buffer_m" in lu.columns:
    pref_col = f"prop_agri_r{int(default_radius)}"
    if pref_col not in piv.columns:
        rcols = [c for c in piv.columns if str(c).startswith("prop_agri_r")]
        if not rcols:
            raise SystemExit(
                "No prop_agri_r* columns could be created from LANDUSE.")
        pref_col = sorted(rcols, key=lambda s: abs(
            int(s.split("r")[1]) - default_radius))[0]
    piv = piv.rename(columns={pref_col: "prop_agri"})

panel["lake_id"] = panel["entity_id"]
panel = panel.merge(piv, on=["lake_id", "year"], how="left")

# TSI
def tsi_components(sd_m=None, chla_ugL=None, tp_ugL=None):
    out = {}
    if (sd_m is not None) and np.isfinite(sd_m) and sd_m > 0:
        out["TSI_SD"] = 60.0 - 14.41*np.log(sd_m)
    if (chla_ugL is not None) and np.isfinite(chla_ugL) and chla_ugL > 0:
        out["TSI_CHLA"] = 9.81*np.log(chla_ugL) + 30.6
    if (tp_ugL is not None) and np.isfinite(tp_ugL) and tp_ugL > 0:
        out["TSI_TP"] = 14.42*np.log(tp_ugL) + 4.15
    return out

def compute_tsi(row):
    comps = tsi_components(sd_m=row.get(
        "TRANS"), chla_ugL=row.get("CHLA"), tp_ugL=row.get("TP"))
    if not comps:
        return np.nan
    return float(np.median(list(comps.values())))

panel["TSI"] = panel.apply(compute_tsi, axis=1)
panel = panel[(panel["year"] >= start_year) & (panel["year"] <= end_year)]
panel = panel.sort_values(["entity_id", "year", "month"])

out = out_dir / f"lake_month_panel__{start_year}-{end_year}.csv"
keep_cols = ["entity_id", "lake_name", "latitude", "longitude", "year", "month", "lake_id",
             "TP", "TN", "TRANS", "CHLA", "TEMP", "TSI", "prop_agri"]

rcols = [c for c in panel.columns if c.startswith("prop_agri_r")]
keep_cols += sorted(rcols)
panel[keep_cols].to_csv(out, index=False)

nlakes = panel["entity_id"].nunique()
print(f"Wrote: {out} | rows={len(panel):,} | lakes={nlakes:,}")
if rcols:
    radii = []
    for c in rcols:
        try:
            radii.append(int(c.split("r")[1]))
        except:
            pass
    if radii:
        x = panel[["year"]+sorted(rcols)].melt(id_vars=["year"],
                                               var_name="col", value_name="v")
        x["buffer_m_round"] = x["col"].str.replace(
            "prop_agri_r", "", regex=False).astype(int)
        s = x.groupby("buffer_m_round", as_index=False).size().rename(
            columns={"size": "rows"})
        print("Land-use radii found (rounded to nearest meter):")
        print(s.to_string(index=False))
print("Columns present:", ["prop_agri"] + sorted(rcols)[:5])
