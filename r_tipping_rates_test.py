"""
Author: @ Jacob Serpico
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

data_dir = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data")
panel_dir = data_dir / "panels" / "lake_month_panel__1992-2022.csv"
analysis_dir = data_dir / "analysis" / "gilarranz_replication"
candidates = analysis_dir / "candidate_changes__monthly.csv"
output_dir = analysis_dir / "r_tipping"
output_dir.mkdir(parents=True, exist_ok=True)

horizon = 6   # horizon in months
temperature_window = 3   # months
aggregeate_rate_window_year = 3   # years for AG slope
minimum_months = 24

output_csv = output_dir / "hazard_glm_terms.csv"

def robust_read_csv(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", **kw)

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def rolling_ls_slope(y, t):
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    if len(y) < 2:
        return np.nan
    t0 = t - t.mean()
    den = (t0**2).sum()
    if den == 0:
        return np.nan
    return float((t0 * (y - y.mean())).sum() / den)

# Load panel
panel = robust_read_csv(panel_dir, dtype=str, low_memory=False)
need = {"lake_id", "year", "month", "TEMP"}
miss = [c for c in need if c not in panel.columns]
if miss:
    raise SystemExit(f"Panel missing columns: {miss}")
panel["year"] = to_num(panel["year"])
panel["month"] = to_num(panel["month"])
panel["date_num"] = (panel["year"]*12 + panel["month"]).astype("Int64")

# TEMP_ds if absent. De-mean by lake and month
if "TEMP_ds" not in panel.columns:
    panel = panel.sort_values(["lake_id", "year", "month"]).copy()
    panel["TEMP"] = to_num(panel["TEMP"])
    panel["TEMP_ds"] = panel["TEMP"] - \
        panel.groupby(["lake_id", "month"])["TEMP"].transform("mean")
    print("note: TEMP_ds computed by lake and month demeaning in r_tipping_hazard_glm.py")

# Exposure
has_ag = "prop_agri" in panel.columns
if has_ag:
    panel["prop_agri"] = to_num(panel["prop_agri"])

# Tipping events (prefer monthly candidates)
tips = None
if candidates.exists():
    ev = robust_read_csv(candidates, dtype=str)
    for c in ["lake_id", "year", "month", "is_tipping"]:
        if c not in ev.columns:
            ev[c] = np.nan
    ev["year"] = to_num(ev["year"])
    ev["month"] = to_num(ev["month"])
    ev["date_num"] = (ev["year"]*12 + ev["month"]).astype("Int64")
    ev["is_tipping"] = ev["is_tipping"].astype(
        str).str.lower().isin(["1", "true", "t", "yes", "y"])
    tips = ev[ev["is_tipping"]][["lake_id", "date_num"]].dropna().copy()

# Fallback if no event list
if tips is None or tips.empty:
    raise SystemExit(
        "No candidate tipping events found (candidate_changes__monthly.csv).")

evt = tips.copy()
evt["y"] = 1
event_map = {(r.lake_id, int(r.date_num)) : 1 for r in evt.itertuples(index=False)}

# Build design matrix Z
rows = []
for gid, d in panel.groupby("lake_id", sort=False):
    if len(d) < minimum_months:
        continue
    d = d.sort_values("date_num").copy()

    # TEMP predictors (monthly)
    d["TEMP_rate_3m"] = d["TEMP_ds"].rolling(temperature_window).apply(
        lambda s: rolling_ls_slope(s.values, np.arange(len(s))), raw=False)
    d["TEMP_dtheta"] = d["TEMP_ds"].rolling(temperature_window).apply(
        lambda s: np.nanmax(s)-np.nanmin(s), raw=False)

    # AGRI predictors (yearly mapped to months)
    if has_ag:
        agy = (d[["year", "prop_agri"]].dropna()
               .groupby("year", as_index=False)["prop_agri"].mean()
               .sort_values("year"))
        if len(agy) >= 2:
            # rolling 3 year slope (per year), map to months
            if len(agy) >= aggregeate_rate_window_year:
                agy["AG_rate_year"] = agy["prop_agri"].rolling(aggregeate_rate_window_year).apply(
                    lambda s: rolling_ls_slope(s.values, np.arange(len(s))), raw=False)
            else:
                agy["AG_rate_year"] = agy["prop_agri"].diff(1)
            agy["AG_rate_3y_mo"] = agy["AG_rate_year"] / 12.0

            # AG level (use yearly mean. could also use 3 yr mean)
            agy["AG_level"] = agy["prop_agri"]

            d = d.merge(agy[["year", "AG_rate_3y_mo", "AG_level"]],
                        on="year", how="left")

    for _, r in d.iterrows():
        dn = int(r["date_num"]) if pd.notna(r["date_num"]) else None
        if dn is None:
            continue
        y = 0
        for h in range(1, horizon+1):
            if (gid, dn+h) in event_map:
                y = 1
                break
        rows.append({
            "lake_id": gid,
            "date_num": dn,
            "y": y,
            "TEMP_rate_3m": r.get("TEMP_rate_3m", np.nan),
            "TEMP_dtheta":  r.get("TEMP_dtheta",  np.nan),
            "AG_rate_3y_mo": r.get("AG_rate_3y_mo", np.nan) if has_ag else np.nan,
            "AG_level":      r.get("AG_level",      np.nan) if has_ag else np.nan,
        })

Z = pd.DataFrame(rows)

# Drop rows with all predictors missing
preds = ["TEMP_rate_3m", "TEMP_dtheta"]
if has_ag:
    preds += ["AG_rate_3y_mo", "AG_level"]
Z = Z.dropna(subset=["y"] + preds, how="any").copy()

# Interactions
Z["TEMP_r_x_dtheta"] = Z["TEMP_rate_3m"] * Z["TEMP_dtheta"]
if has_ag:
    Z["AG_r_x_dtheta"] = Z["AG_rate_3y_mo"] * Z["TEMP_dtheta"]
    Z["AGlev_x_dtheta"] = Z["AG_level"] * Z["TEMP_dtheta"]

# Standardize (z) to stabilize scale
def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    m = s.mean()
    sd = s.std(ddof=1)
    return (s-m)/sd if (sd and np.isfinite(sd) and sd != 0) else pd.Series(np.nan, index=s.index)

cols = ["TEMP_rate_3m", "TEMP_dtheta", "TEMP_r_x_dtheta"]
if has_ag:
    cols += ["AG_rate_3y_mo", "AG_level", "AG_r_x_dtheta", "AGlev_x_dtheta"]

for c in cols:
    Z[c+"_z"] = zscore(Z[c])

# Fit GLM (Binomial) with robust SEs
use = [c+"_z" for c in cols]
d = Z[["y"] + use].replace([np.inf, -np.inf], np.nan).dropna()
y = d["y"].astype(int).values
X = sm.add_constant(d[use], has_constant="add")
res = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC1")

coefs = res.params
ses = res.bse
out = pd.DataFrame({
    "term": coefs.index,
    "odds_ratio": np.exp(coefs.values),
    "or_95lo": np.exp(coefs.values - 1.96*ses.values),
    "or_95hi": np.exp(coefs.values + 1.96*ses.values),
    "p_value": res.pvalues.values
})
out.to_csv(output_csv, index=False)

print(f"Wrote: {output_csv}")
print("Significant (p<0.05):")
sig = out[(out["term"] != "const") & (
    out["p_value"] < 0.05)].sort_values("p_value")
if len(sig):
    print(sig.to_string(index=False))
else:
    print("  none")
print("\nKey terms:")
keep = [t for t in out["term"] if any(
    k in t for k in ["TEMP_rate_3m", "TEMP_dtheta", "AG_rate_3y_mo", "AG_level"])]
print(out[out["term"].isin(keep)].sort_values("term").to_string(index=False))
