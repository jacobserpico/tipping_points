"""
Author: @Jacob Serpico
"""

from pathlib import Path
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter
from numpy import RankWarning

# File paths
data_dir = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data")
agg_path = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data/aggregate_water_quality/aggregate_water_quality__1990-2024__with_water_parameters.csv")

fig_dir = Path("/Users/Jake/Desktop")
fig_dir.mkdir(parents=True, exist_ok=True)

# Time specificaation
start_year = 1990
end_year = 2024

# Variables of interest
variables = {
    "phosphorus": ["TDP", "TP"],
    "secchi_depth": ["TRANS"],
    "turbidity": ["TURB"],
    "nitrogen": ["TDN", "TN"],
    "temperature": ["TEMP"],
}

# Colors
colors = {
    "phosphorus": "#000000",
    "secchi_depth": "#000000",
    "turbidity": "#000000",
    "nitrogen": "#000000",
    "temperature": "#000000",
}

# Aggregation frequency
# M = monthly (default). "Y" for yearly. Yearly just makes things less noisy.
agg_freq_by_var = {
    # "temperature": "Y"
}

# Aggregate by lake if available, else station, then take global median
station_first = True

# if True, group by lake (preferred) else by station
weight_by_lake = False
min_years_per_station = 1
surface_max_depth_m = None # 2.0
# String match on water_type
water_type_keep = None #["lake"]

# Detrending
detrend = False
detrend_per_station = False

min_points_per_group = 3

# Trophic state index proxy thresholds
thresholds = {
    "phosphorus": 24.0, # µg/L
    "secchi_depth": 2.0, # m
}

# Plot number of samples
plot_counts = False

# Style parameters
line_lw = 1.0
iqr_lw = 0.5
iqr_color = "#989a9e"
# shade_pos = "#f3b0a8"
shade_pos = "#A8A8A8"
# shade_neg = "#A8A8A8"
shade_neg = "#4998F5"
residual_pos = "#f3b0a8"
residual_neg = "#A8A8A8"
facecolor = "#ffffff"

# Helper functions
def read_aggregate(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["sample_date"] = pd.to_datetime(
        df["sample_date"], errors="coerce", utc=False)

    v = df["value"].fillna("").astype(str).str.replace(",", ".", regex=False)
    df["value"] = pd.to_numeric(v, errors="coerce")

    df["parameter_code"] = df["parameter_code"].astype(
        str).str.strip().str.upper()
    df["unit"] = df["unit"].astype(str).fillna("").str.strip()

    m = (df["sample_date"].dt.year >= start_year) & (
        df["sample_date"].dt.year <= end_year)
    df = df[m].dropna(subset=["sample_date", "value"])
    return df

def normalize_unit(u):
    s = str(u).strip().lower()
    s = s.replace("µ", "u")
    s = re.sub(r"\s+", "", s)
    if s in ("m", "meter", "metre", "meters", "metres"):
        return "m"
    if s in ("cm", "centimeter", "centimetre", "centimeters", "centimetres"):
        return "cm"
    if "mg/l" in s or "mgperlitre" in s or "mgperl" in s:
        return "mg/L"
    if "ug/l" in s or "μg/l" in s or "microg/l" in s or "mcg/l" in s:
        return "ug/L"
    if "ntu" in s:
        return "NTU"
    if s in ("c", "degc", "°c"):
        return "C"
    if s in ("ph",):
        return "pH"
    if "%" in s or "percent" in s or "perc" in s:
        return "%"
    return s or ""

def standardize_units(df, var_key):
    df = df.copy()
    df["unit_std"] = df["unit"].map(normalize_unit)
    y_unit = None

    if var_key == "phosphorus":
        df = df[df["unit_std"].isin(["ug/L", "mg/L"])]
        mg = df["unit_std"] == "mg/L"
        df.loc[mg, "value"] = df.loc[mg, "value"] * 1000.0
        y_unit = "µg/L"

    elif var_key == "secchi_depth":
        df["unit_std"] = df["unit_std"].fillna("")
        df = df[df["unit_std"].isin(["m", "cm"])]
        cm = df["unit_std"] == "cm"
        df.loc[cm, "value"] = df.loc[cm, "value"] / 100.0
        m_is_cm = (df["unit_std"] == "m") & (df["value"].between(30, 250))
        df.loc[m_is_cm, "value"] = df.loc[m_is_cm, "value"] / 100.0
        df = df[(df["value"] >= 0) & (df["value"] <= 30)]
        y_unit = "m"

    elif var_key == "turbidity":
        df = df[df["unit_std"].isin(["NTU"])]
        y_unit = "NTU"

    elif var_key == "temperature":
        df = df[df["unit_std"].isin(["C"])]
        y_unit = "°C"

    elif var_key == "ph":
        y_unit = "pH"

    else:
        if "unit_std" in df.columns and df["unit_std"].notna().any():
            y_unit = df["unit_std"].mode(dropna=True).iloc[0]
        else:
            y_unit = ""

    return df, y_unit

def select_variable(df, codes):
    return df[df["parameter_code"].isin([c.upper() for c in codes])]

def attach_entity_id(df):
    d = df.copy()
    if "lake_name" in d.columns and d["lake_name"].notna().any():
        d["lake_id"] = d["lake_name"].fillna("").astype(str).str.strip()
        missing = d["lake_id"].eq("")
        if missing.any():
            d.loc[missing, "lake_id"] = d.loc[missing,
                                              "gems_station_number"].astype(str).str.strip()
    else:
        d["lake_id"] = d["gems_station_number"].astype(str).str.strip()
    return d


def _entity_col(df):
    return "lake_id" if ("lake_id" in df.columns and df["lake_id"].notna().any()) else "gems_station_number"


def monthwise_interpolate(df, per_entity=True, max_gap=2, require_min_obs=3, use_log=True):
    d = df.copy()
    d["year"] = d["sample_date"].dt.year
    d["month"] = d["sample_date"].dt.month

    ent = _entity_col(d) if per_entity else None
    group_cols = ([ent] if ent else []) + ["year", "month"]

    # Collapse to one value per (entity, year, month)
    g = (
        d.groupby(group_cols, sort=False)["value"]
        .median()
        .rename("value")
        .reset_index()
    )

    # Full year index
    years = np.arange(start_year, end_year + 1)

    parts = []
    by_cols = ([ent] if ent else []) + ["month"]
    for _, sub in g.groupby(by_cols, sort=False):
        # Need a minimum number of observed years to interpolate
        if sub["value"].notna().sum() < require_min_obs:
            # Still return the observed points if no fill
            sub2 = sub.copy()
            parts.append(sub2)
            continue

        s = sub.set_index("year")["value"].sort_index().reindex(years)

        if use_log:
            s_pos = s.where(s > 0)
            s_work = np.log(s_pos)
        else:
            s_work = s

        s_filled = s_work.interpolate(
            method="index",
            limit=max_gap,
            limit_direction="both"
        )

        if use_log:
            s_filled = np.exp(s_filled)

        sub2 = pd.DataFrame({
            "year": years,
            "value": s_filled.values,
        })

        sub2["month"] = sub["month"].iloc[0]
        if ent:
            sub2[ent] = sub[ent].iloc[0]

        parts.append(sub2)

    filled = pd.concat(parts, ignore_index=True)

    filled["sample_date"] = pd.to_datetime(dict(
        year=filled["year"], month=filled["month"], day=1
    ))

    keep = ["sample_date", "value"]
    if ent:
        keep.append(ent)
    return filled[keep].sort_values(["sample_date"] + ([ent] if ent else []))


def filter_common(df):
    d = df.copy()

    if water_type_keep and "water_type" in d.columns:
        toks = {t.strip().casefold() for t in water_type_keep}
        d = d[d["water_type"].astype(str).str.casefold().apply(
            lambda s: any(tok in s for tok in toks))]

    if surface_max_depth_m is not None and "depth" in d.columns:
        d["depth"] = pd.to_numeric(d["depth"], errors="coerce")
        d = d[(d["depth"].isna()) | (d["depth"] <= float(surface_max_depth_m))]
    return d


def filter_by_station_years(df, min_years):
    if not min_years:
        return df
    ent = _entity_col(df)
    yrs = df.assign(_y=df["sample_date"].dt.year)
    keep_ids = (
        yrs.groupby(ent)["_y"]
        .nunique()
        .loc[lambda s: s >= min_years]
        .index
    )
    return df[df[ent].isin(keep_ids)]


def linear_residuals(x_year, y):
    mask = np.isfinite(x_year) & np.isfinite(y)
    if mask.sum() < min_points_per_group or np.unique(x_year[mask]).size < 2:
        r = y.copy().astype(float)
        if mask.any():
            r[mask] = r[mask] - np.nanmean(r[mask])
        r[~mask] = np.nan
        return r
    xc = x_year.astype(float)
    xc = xc - np.nanmean(xc[mask])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RankWarning)
        a, b = np.polyfit(xc[mask], y[mask], 1)
    yhat = a * xc + b
    return y - yhat

def apply_detrend(df, freq="M", per_station=True):
    d = df.copy()
    d["year"] = d["sample_date"].dt.year
    ent = _entity_col(d) if per_station else None

    if freq == "M":
        d["month"] = d["sample_date"].dt.month
        keys = [ent, "month"] if ent else ["month"]
    else:  # yearly
        keys = [ent] if ent else []

    if keys:
        parts = []
        for _, g in d.groupby(keys, sort=False):
            x = g["year"].to_numpy()
            y = g["value"].to_numpy(dtype=float)
            r = linear_residuals(x, y)
            g2 = g.copy()
            g2["value"] = r
            parts.append(g2)
        d = pd.concat(parts, ignore_index=False)
    else:
        x = d["year"].to_numpy()
        y = d["value"].to_numpy(dtype=float)
        d["value"] = linear_residuals(x, y)

    return d.sort_index()

def aggregate_stats(df, freq="M", station_first=False):
    if freq == "Y":
        key = df["sample_date"].dt.to_period("Y").dt.to_timestamp()
    else:
        key = df["sample_date"].values.astype("datetime64[M]")

    # When we aggregate first by entity
    if station_first:
        ent = _entity_col(df)
        g1 = df.groupby([df[ent], key])["value"].median()
        g = g1.groupby(level=1)
        out = pd.DataFrame({
            "median": g.median(),
            "q25": g.quantile(0.25),
            "q75": g.quantile(0.75),
            "n": g.size()  # number of entities contributing
        })
    else:
        g = df.groupby(key)["value"]
        out = pd.DataFrame({
            "median": g.median(),
            "q25": g.quantile(0.25),
            "q75": g.quantile(0.75),
            "n": g.count()
        })

    if freq == "Y":
        idx = pd.period_range(
            f"{start_year}", f"{end_year}", freq="Y").to_timestamp()
    else:
        idx = pd.period_range(f"{start_year}-01",
                              f"{end_year}-12", freq="M").to_timestamp()
    out = out.reindex(idx)
    return out

def compute_counts(df, freq="M"):
    # n_entities and n_samples per period
    if freq == "Y":
        key = df["sample_date"].dt.to_period("Y").dt.to_timestamp()
    else:
        key = df["sample_date"].values.astype("datetime64[M]")
    ent = _entity_col(df)
    n_ent = df.groupby(key)[ent].nunique()
    n_sa = df.groupby(key)["value"].size()
    if freq == "Y":
        idx = pd.period_range(
            f"{start_year}", f"{end_year}", freq="Y").to_timestamp()
    else:
        idx = pd.period_range(f"{start_year}-01",
                              f"{end_year}-12", freq="M").to_timestamp()
    out = pd.DataFrame({"n_entities": n_ent, "n_samples": n_sa}).reindex(idx)
    return out

def format_axes(ax):
    ax.grid(False)
    ax.set_facecolor(facecolor)
    ax.tick_params(axis="both", which="both", direction="in", length=4,
                   top=False, right=False, left=True, bottom=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(pd.Timestamp(f"{start_year}-01-01"),
                pd.Timestamp(f"{end_year}-12-31"))
    ax.xaxis.set_major_locator(YearLocator(base=2))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.set_xlabel("Year")

def plot_series(ts, var_key, color, y_label_text, thr=None, freq="M", shade_residuals=False):
    fig = plt.figure(figsize=(10, 5), dpi=600)
    ax = fig.add_subplot(111)

    # Inner quartile range lines
    ax.plot(ts.index, ts["q25"], color=iqr_color,
            lw=iqr_lw, alpha=0.9, zorder=1)
    ax.plot(ts.index, ts["q75"], color=iqr_color,
            lw=iqr_lw, alpha=0.9, zorder=1)

    x = ts.index.to_pydatetime()
    y = ts["median"].to_numpy(dtype=float)
    y_ma = np.ma.masked_invalid(y)

    # Threshold shades
    if thr is not None and not shade_residuals and ts["median"].notna().any():
        above = y_ma >= thr
        below = y_ma < thr
        ax.fill_between(x, thr, y_ma, where=above, interpolate=True,
                        facecolor=shade_pos, edgecolor="none", alpha=0.6, zorder=0)
        ax.fill_between(x, thr, y_ma, where=below, interpolate=True,
                        facecolor=shade_neg, edgecolor="none", alpha=0.6, zorder=0)
        ax.axhline(thr, lw=1, color="#000000", alpha=0.9,
                   zorder=1, linestyle="dashed")

    # Residual shading near 0
    if shade_residuals and ts["median"].notna().any():
        base = 0.0
        above0 = y_ma >= base
        below0 = y_ma < base
        ax.fill_between(x, base, y_ma, where=above0, interpolate=True,
                        facecolor=residual_pos, edgecolor="none", alpha=0.6, zorder=0)
        ax.fill_between(x, base, y_ma, where=below0, interpolate=True,
                        facecolor=residual_neg, edgecolor="none", alpha=0.6, zorder=0)
        ax.axhline(base, lw=1, color="#000000", alpha=0.9,
                   zorder=1, linestyle="dashed")

    # Median
    ax.plot(ts.index, ts["median"], color=color,
            lw=line_lw, zorder=2, alpha=0.6)

    format_axes(ax)
    ax.set_ylabel(y_label_text)

    tag = "yearly" if freq == "Y" else "monthly"
    if detrend:
        tag += "_detrended"
    out_png = fig_dir / f"{var_key}__{tag}__{start_year}-{end_year}.png"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote: {out_png}")


def plot_counts_series(counts, var_key, freq="M"):
    fig = plt.figure(figsize=(10, 3.2), dpi=600)
    ax = fig.add_subplot(111)
    ax.plot(counts.index, counts["n_entities"],
            color="#000000", lw=1.0, alpha=0.8)
    format_axes(ax)
    ax.set_ylabel("Contributing entities (n)")
    tag = "yearly" if freq == "Y" else "monthly"
    out_png = fig_dir / \
        f"{var_key}__{tag}__counts__{start_year}-{end_year}.png"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote: {out_png}")

# Run it
if __name__ == "__main__":
    df = read_aggregate(agg_path)

    for key, codes in variables.items():
        sub = select_variable(df, codes)
        if sub.empty:
            print(f"skip {key}: no data")
            continue

        sub, unit_label = standardize_units(sub, key)
        if sub.empty:
            print(f"skip {key}: no usable data after unit filtering")
            continue

        sub = filter_common(sub)
        if sub.empty:
            print(f"skip {key}: no data after water_type/surface filters")
            continue

        sub = attach_entity_id(sub)
        sub = filter_by_station_years(sub, min_years_per_station)

        sub = monthwise_interpolate(
            sub,
            per_entity=True,
            max_gap=2,
            require_min_obs=3,
            use_log=(key in ["phosphorus", "turbidity",
             "nitrogen"])
        )

        sub = filter_by_station_years(sub, min_years_per_station)
        if sub.empty:
            print(
                f"skip {key}: no entities meet min_years_per_station={min_years_per_station}")
            continue

        # De-trend
        freq = agg_freq_by_var.get(key, "M")
        sub_dt = sub
        if detrend:
            sub_dt = apply_detrend(
                sub, freq=freq, per_station=detrend_per_station)
            thr_to_use = None
            ylab = (f"{key.replace('_', ' ').title()} "
                    f"({'residuals' if not unit_label else unit_label + ' residuals'})")
            shade_resid = True
        else:
            thr_to_use = float(
                thresholds[key]) if key in thresholds and thresholds[key] is not None else None
            ylab = f"{key.replace('_', ' ').title()} ({unit_label})" if unit_label else f"{key.replace('_', ' ').title()}"
            shade_resid = False

        # Aggregate
        ts = aggregate_stats(
            sub_dt, freq=freq, station_first=True if weight_by_lake else False)
        color = colors.get(key, "#0d6efd")
        plot_series(ts, key, color, ylab, thr=thr_to_use,
                    freq=freq, shade_residuals=shade_resid)

        # Count plots
        if plot_counts:
            cnt = compute_counts(sub, freq=freq)
            plot_counts_series(cnt, key, freq=freq)