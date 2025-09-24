"""
Author: @Jacob Serpico
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# File paths
data_dir = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data")
agg_csv = data_dir / "aggregate_water_quality" / "aggregate_water_quality__1990-2024.csv"
land_use_csv = data_dir / "land_use_data" / "lake_landuse_allyears.csv"
figure_dir = Path("/Users/Jake/Desktop")
figure_dir.mkdir(parents=True, exist_ok=True)
save_fig = True

# Specifications
font_size = 15
title_font_size = 22
x_rotation = 55
y_rotation = 0

start_year = 1992
end_year = 2022
period_freq = "Y"

water_type_keep = None
surface_max_depth_m = None
min_years_per_entity = 1

variables = {
    "TP":     ["TDP"],
    "TN":     ["TDN"],
    "TRANS":  ["TRANS"],
    "TURB":   ["TURB"],
    "TEMP":   ["TEMP"],
}

friendly_labels = {
    "TP": "Phosphorus",
    "TN": "Nitrogen",
    "TRANS": "Secchi Depth",
    "TURB": "Turbidity",
    "TEMP": "Temperature",
    "TEMP_ds": "Temp Residuals",
    "prop_agri": "Agriculture",
    "latitude": "Latitude",
    "longitude": "Longitude",
}

top_n = None
minimum_value = None
show_upper_half = False
annotate = True
annotate_fontsize = 14
color_map = "vlag"
decimals = 2
focus = None

add_agriculture = True
grassland = False  # crops and grassland if True; crops only if False

land_cover_radius = 3000

# Residualize on these controls
residualize_on = ["latitude", "longitude"]

# Variables to residualize
residualize_variables = [
                    "prop_agri", 
                    "TP", 
                    "TN", 
                    "TRANS", 
                    "TURB", 
                    "TEMP"
                    ]

create_csv = False

# Helper functions
def normalize_unit(u):
    s = str(u or "").strip().lower().replace("µ", "u")
    s = re.sub(r"\s+", "", s)
    if s in {"m", "meter", "metre", "meters", "metres"}:
        return "m"
    if s in {"cm", "centimeter", "centimetre", "centimeters", "centimetres"}:
        return "cm"
    if "mg/l" in s:
        return "mg/L"
    if "ug/l" in s or "μg/l" in s or "microg/l" in s or "mcg/l" in s:
        return "ug/L"
    if "ntu" in s:
        return "NTU"
    if s in {"c", "degc", "°c"}:
        return "C"
    return s or ""

def standardize_units_minimal(df, var_key):
    d = df.copy()
    d["unit_std"] = d["unit"].map(normalize_unit)
    if var_key == "TP":
        d = d[d["unit_std"].isin(["ug/L", "mg/L"])]
        mg = d["unit_std"].eq("mg/L")
        d.loc[mg, "value"] = d.loc[mg, "value"] * 1000.0
    elif var_key == "TN":
        d = d[d["unit_std"].isin(["ug/L", "mg/L"])]
        ug = d["unit_std"].eq("ug/L")
        d.loc[ug, "value"] = d.loc[ug, "value"] / 1000.0
    elif var_key == "TRANS":
        d = d[d["unit_std"].isin(["m", "cm"])]
        cm = d["unit_std"].eq("cm")
        d.loc[cm, "value"] = d.loc[cm, "value"] / 100.0
        m_is_cm = d["unit_std"].eq("m") & d["value"].between(30, 250)
        d.loc[m_is_cm, "value"] = d.loc[m_is_cm, "value"] / 100.0
        d = d[(d["value"] >= 0) & (d["value"] <= 30)]
    elif var_key == "TURB":
        d = d[d["unit_std"].eq("NTU")]
    elif var_key == "TEMP":
        d = d[d["unit_std"].eq("C")]
    return d

def attach_entity_id(df):
    d = df.copy()
    if "lake_name" in d.columns and d["lake_name"].notna().any():
        d["entity_id"] = d["lake_name"].fillna("").astype(str).str.strip()
        missing = d["entity_id"].eq("")
        d.loc[missing, "entity_id"] = d.loc[missing,
                                            "gems_station_number"].astype(str).str.strip()
    else:
        d["entity_id"] = d["gems_station_number"].astype(str).str.strip()
    return d


def mode_str(x):
    s = pd.Series(x).dropna().astype(str)
    if s.empty:
        return np.nan
    return s.mode().iloc[0]

# Data 
def load_aggregate_table(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["sample_date"] = pd.to_datetime(
        df["sample_date"], errors="coerce", utc=False)
    v = df["value"].fillna("").astype(str).str.replace(",", ".", regex=False)
    df["value"] = pd.to_numeric(v, errors="coerce")
    for c in ["latitude", "longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["sample_date", "value"])
    m = (df["sample_date"].dt.year >= start_year) & (
        df["sample_date"].dt.year <= end_year)
    return df[m]

def apply_basic_filters(df):
    d = df.copy()
    if water_type_keep and "water_type" in d.columns:
        toks = {t.strip().casefold() for t in water_type_keep}
        d = d[d["water_type"].astype(str).str.casefold().apply(
            lambda s: any(t in s for t in toks))]
    if surface_max_depth_m is not None and "depth" in d.columns:
        d["depth"] = pd.to_numeric(d["depth"], errors="coerce")
        d = d[(d["depth"].isna()) | (d["depth"] <= float(surface_max_depth_m))]
    return d

def build_wide(df, freq="Y"):
    d = attach_entity_id(df)
    d = apply_basic_filters(d)
    keep_codes = np.unique([c for codes in variables.values() for c in codes])
    d = d[d["parameter_code"].str.upper().isin(keep_codes)]

    parts = []
    for var_key, codes in variables.items():
        sub = d[d["parameter_code"].str.upper().isin([c.upper()
                                                      for c in codes])]
        if sub.empty:
            continue
        parts.append(standardize_units_minimal(sub, var_key))
    if not parts:
        return pd.DataFrame()

    d = pd.concat(parts, ignore_index=True)
    period = d["sample_date"].dt.to_period(freq).dt.to_timestamp()
    d = d.assign(period=period)

    if min_years_per_entity:
        yrs = d.groupby("entity_id")["period"].apply(
            lambda s: s.dt.year.nunique())
        keep_ids = yrs[yrs >= int(min_years_per_entity)].index
        d = d[d["entity_id"].isin(keep_ids)]

    g = (d.groupby(["entity_id", "period", "parameter_code"], observed=True)["value"]
         .median().reset_index())
    code_to_key = {c.upper(): k for k, codes in variables.items()
                   for c in codes}
    g["var"] = g["parameter_code"].str.upper().map(code_to_key)

    wide = (g.pivot_table(index=["entity_id", "period"], columns="var",
                          values="value", aggfunc="median").reset_index())
    wide["year"] = wide["period"].dt.year
    pos = (df.groupby("entity_id", as_index=False)
             .agg(latitude=("latitude", "median"),
                  longitude=("longitude", "median")))
    wide = wide.merge(pos, on="entity_id", how="left")
    return wide

def read_land_use(path):
    try:
        lu = pd.read_csv(path, dtype=str, sep=",", quotechar='"',
                         escapechar="\\", low_memory=False)
    except pd.errors.ParserError:
        lu = pd.read_csv(
            path,
            dtype=str,
            sep=",",
            quotechar='"',
            escapechar="\\",
            engine="python",
            on_bad_lines="skip"
        )

    lu.columns = [c.strip() for c in lu.columns]

    for c in ("year", "buffer_m"):
        if c in lu.columns:
            lu[c] = pd.to_numeric(lu[c], errors="coerce")

    if "prop_agri" not in lu.columns:
        frac_cols = [c for c in lu.columns if c.startswith("frac_")]
        if frac_cols:
            lu[frac_cols] = lu[frac_cols].apply(pd.to_numeric, errors="coerce")
            crops = lu["frac_crops"] if "frac_crops" in lu.columns else 0.0
            if grassland and "frac_grassland" in lu.columns:
                prop = crops + lu["frac_grassland"]
            else:
                prop = crops
            lu["prop_agri"] = pd.to_numeric(prop, errors="coerce").clip(0, 1)
    return lu

def pick_radius_table(lu):
    if "buffer_m" not in lu.columns or lu["buffer_m"].dropna().empty:
        return lu[["lake_id", "year", "prop_agri"]].dropna(subset=["lake_id", "year"])
    avail = sorted(lu["buffer_m"].dropna().astype(float).unique())
    target = float(land_cover_radius) if land_cover_radius in avail else min(
        avail, key=lambda r: abs(r - land_cover_radius))
    sub = lu[lu["buffer_m"].astype(float).eq(target)].copy()
    return sub[["lake_id", "year", "prop_agri"]].dropna(subset=["lake_id", "year"])

def residualize_columns(df, targets, controls):
    if not targets or not controls:
        return df.copy()
    out = df.copy()
    X = out[controls].copy()
    X = X.assign(const=1.0)
    cols = ["const"] + controls
    for c in cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for t in targets:
        if t not in out.columns:
            continue
        y = pd.to_numeric(out[t], errors="coerce")
        m = X[cols].notna().all(1) & y.notna()
        if m.sum() < 10:
            out[t] = np.nan
            continue
        beta, _, _, _ = np.linalg.lstsq(
            X.loc[m, cols].values, y.loc[m].values, rcond=None)
        yhat = (X[cols].values @ beta)
        out[t] = y - yhat
    return out

def friendly(name):
    return friendly_labels.get(name, name)


def prepare_corr_matrix(num, method, top_n, minimum_value, focus):
    corr = num.corr(method=method)
    if focus and focus in corr.columns:
        s = corr[focus].drop(labels=[focus]).abs()
        if minimum_value is not None:
            s = s[s >= float(minimum_value)]
        if top_n is not None:
            s = s.sort_values(ascending=False).head(top_n)
        keep = [focus] + s.index.tolist()
        sub = corr.loc[keep, keep]
    else:
        abs_corr = corr.abs().mask(np.eye(len(corr), dtype=bool))
        pairs = abs_corr.unstack().dropna().sort_values(ascending=False)
        if minimum_value is not None:
            pairs = pairs[pairs >= float(minimum_value)]
        if top_n is not None:
            pairs = pairs.head(top_n)
        if pairs.empty:
            sub = corr
        else:
            keep = pd.Index(pairs.index.get_level_values(0).tolist() +
                            pairs.index.get_level_values(1).tolist()).unique()
            sub = corr.loc[keep, keep]
    sub = sub.rename(index=friendly, columns=friendly)
    return sub

def correlation_heatmap(
    df,
    top_n=None,
    method="pearson",
    minimum_value=None,
    show_upper_half=False,
    cmap="vlag",
    annotate=True,
    decimals=2,
    annotate_fontsize=None,
    exclude=None,
    exclude_regex=None,
    focus=None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    num = df.select_dtypes("number").copy()
    for c in ["year"]:
        if c in num.columns:
            num = num.drop(columns=[c])
    if exclude:
        num = num.drop(
            columns=[c for c in exclude if c in num.columns], errors="ignore")
    if exclude_regex:
        keep_cols = [c for c in num.columns if not re.search(exclude_regex, c)]
        num = num[keep_cols]
    num = num.loc[:, ~num.isna().all(axis=0)]
    if num.shape[1] < 2:
        ax.text(0.5, 0.5, "Not enough numeric columns",
                ha="center", va="center")
        return ax

    sub = prepare_corr_matrix(num, method, top_n, minimum_value, focus)

    n = sub.shape[0]
    annot_fs_auto = 10 if n <= 10 else (9 if n <= 16 else 8)
    annot_fs = annotate_fontsize if annotate_fontsize is not None else annot_fs_auto
    mask = None if show_upper_half else np.triu(np.ones_like(sub, dtype=bool))

    sns.heatmap(
        sub, mask=mask, cmap=cmap, center=0, annot=annotate,
        fmt=f".{decimals}f" if annotate else "",
        annot_kws={"size": annot_fs} if annotate else None,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.75}, ax=ax,
    )

    ax.set_title(f"{method.title()} Correlation", pad=12,
                 fontsize=title_font_size, weight="bold")
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True,
                   left=True, right=False, labelleft=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation,
                       ha="right", va="top", fontsize=font_size)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=y_rotation,
                       ha="right", va="center", fontsize=font_size)
    return ax

# Main
def main():
    sns.set_context("talk")
    sns.set_style("white")
    df = load_aggregate_table(agg_csv)
    df = attach_entity_id(df)
    if "gems_station_number" in df.columns:
        ent2lake = (df.groupby("entity_id", as_index=False)
                      .agg(lake_id=("gems_station_number", mode_str)))
    else:
        ent2lake = pd.DataFrame(columns=["entity_id", "lake_id"])

    wide = build_wide(df, freq=period_freq)
    if wide.empty:
        raise SystemExit(
            "Not enough variables present to compute correlations.")

    if add_agriculture and land_use_csv is not None and land_use_csv.exists():
        lu = read_land_use(land_use_csv)
        lu = lu[(lu["year"] >= start_year) & (lu["year"] <= end_year)]
        if "prop_agri" in lu.columns and "lake_id" in lu.columns:
            lu_sel = pick_radius_table(lu)
            wide = wide.merge(ent2lake, on="entity_id", how="left")
            wide = wide.merge(lu_sel, on=["lake_id", "year"], how="left")
        else:
            print(
                "Note: land_use_csv present but missing lake_id/prop_agri; skipping agri merge.")
    else:
        print("Note: agriculture merge skipped (file missing or disabled).")

    for c in ["prop_agri", "latitude", "longitude"]:
        if c in wide.columns:
            wide[c] = pd.to_numeric(wide[c], errors="coerce")

    keep_cols = [k for k in variables.keys() if k in wide.columns]
    extra_cols = [c for c in ["prop_agri",
                              "latitude", "longitude"] if c in wide.columns]
    cols_for_corr = keep_cols + extra_cols
    if len(cols_for_corr) < 2:
        raise SystemExit(
            "Not enough variables present to compute correlations.")
    corr_df_raw = wide[["entity_id", "period", "year"] + cols_for_corr].copy()

    if create_csv:
        pear = corr_df_raw[cols_for_corr].corr(method="pearson")
        spear = corr_df_raw[cols_for_corr].corr(method="spearman")
        pear.to_csv(
            figure_dir / f"corr_matrix_raw_{period_freq.lower()}_{start_year}-{end_year}__pearson.csv")
        spear.to_csv(
            figure_dir / f"corr_matrix_raw_{period_freq.lower()}_{start_year}-{end_year}__spearman.csv")
        print("Wrote raw correlation CSVs (pearson/spearman)")

    did_resid = False
    corr_df_plot = corr_df_raw.copy()
    if residualize_on and any(v in corr_df_plot.columns for v in residualize_variables):
        targets = [v for v in residualize_variables if v in corr_df_plot.columns]
        controls = [c for c in residualize_on if c in corr_df_plot.columns]
        if targets and controls:
            corr_df_resid = residualize_columns(
                corr_df_plot, targets, controls)
            did_resid = True
            if create_csv:
                pear_r = corr_df_resid[cols_for_corr].corr(method="pearson")
                spear_r = corr_df_resid[cols_for_corr].corr(method="spearman")
                pear_r.to_csv(
                    figure_dir / f"corr_matrix_resid_{period_freq.lower()}_{start_year}-{end_year}__pearson.csv")
                spear_r.to_csv(
                    figure_dir / f"corr_matrix_resid_{period_freq.lower()}_{start_year}-{end_year}__spearman.csv")
                print("Wrote residualized correlation CSVs (pearson/spearman)")
            corr_df_plot = corr_df_resid

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    correlation_heatmap(
        corr_df_plot[cols_for_corr], top_n=top_n, method="pearson", minimum_value=minimum_value,
        show_upper_half=show_upper_half, cmap=color_map, annotate=annotate, decimals=decimals,
        annotate_fontsize=annotate_fontsize, focus=focus, ax=axes[0]
    )
    correlation_heatmap(
        corr_df_plot[cols_for_corr], top_n=top_n, method="spearman", minimum_value=minimum_value,
        show_upper_half=show_upper_half, cmap=color_map, annotate=annotate, decimals=decimals,
        annotate_fontsize=annotate_fontsize, focus=focus, ax=axes[1]
    )

    tag = "resid" if did_resid else "raw"
    if save_fig:
        out = figure_dir / \
            f"corr_panel_{tag}_{period_freq.lower()}_{start_year}-{end_year}.png"
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.05)
        print(f"Wrote: {out}")
    plt.close(fig)

if __name__ == "__main__":
    main()
