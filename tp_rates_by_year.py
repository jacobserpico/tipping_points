"""
Author: @ Jacob Serpico
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd


# File paths and parameters
data_dir = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data")
panel_dir = data_dir / "panels" / "lake_month_panel__1992-2022.csv"
candidate_changes_csv = data_dir / "analysis" / \
    "gilarranz_replication" / "candidate_changes__monthly.csv"
output_dir = data_dir / "analysis" / "gilarranz_replication"
output_dir.mkdir(parents=True, exist_ok=True)

start_year, end_year = 1992, 2022
minimum_segments = 24
pre_window = 60

def tnum(y, m): 
    return int(y)*12 + int(m)

def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    xm = xr - xr.mean()
    ym = yr - yr.mean()
    den = (np.sqrt((xm**2).sum()) * np.sqrt((ym**2).sum()))
    if den == 0:
        return np.nan, np.nan
    r = float((xm*ym).sum() / den)
    n = len(x)
    p = math.erfc(abs(r) * math.sqrt((n-2)/(1-r*r)) /
                  math.sqrt(2.0)) if (n > 3 and abs(r) < 1) else np.nan
    return r, p

panel = pd.read_csv(panel_dir, low_memory=False)
for c in ["year", "month"]:
    panel[c] = pd.to_numeric(panel[c], errors="coerce")
panel = panel.dropna(subset=["lake_id", "year", "month"])
panel = panel[(panel["year"] >= start_year) & (panel["year"] <= end_year)].copy()
panel["tnum"] = panel["year"].astype(int)*12 + panel["month"].astype(int)

# Lake coverage bounds
cov = (panel.groupby("lake_id", as_index=False)
            .agg(first_tnum=("tnum", "min"), last_tnum=("tnum", "max")))

# Load changes
ev = pd.read_csv(candidate_changes_csv)
need = {"lake_id", "year", "month", "is_regimeshift", "is_tipping"}
miss = [c for c in need if c not in ev.columns]
if miss:
    raise SystemExit(f"{candidate_changes_csv.name} missing columns: {miss}")
for c in ["year", "month"]:
    ev[c] = pd.to_numeric(ev[c], errors="coerce")

# Tag RS/TP per year, per lake (unique lakes per year)
ev_rs = (ev[ev["is_regimeshift"] == True]
         .dropna(subset=["year"])
         .groupby(["year", "lake_id"], as_index=False).size())
ev_tp = (ev[ev["is_tipping"] == True]
         .dropna(subset=["year"])
         .groupby(["year", "lake_id"], as_index=False).size())

rows = []
for Y in range(start_year, end_year+1):
    y_start = tnum(Y, 1)
    y_end = tnum(Y, 12)

    # Eligibility windows
    # RS need pre/post segments of length minimum_segments around any CP in Y
    rs_elig_mask = (cov["first_tnum"] <= y_start -
                    minimum_segments) & (cov["last_tnum"] >= y_end + minimum_segments)

    # TP need 60 month pre-window and minimum_segments post
    tp_elig_mask = (cov["first_tnum"] <= y_start -
                    pre_window) & (cov["last_tnum"] >= y_end + minimum_segments)

    rs_elig_lakes = set(cov.loc[rs_elig_mask, "lake_id"].astype(str))
    tp_elig_lakes = set(cov.loc[tp_elig_mask, "lake_id"].astype(str))

    # Lakes with >1 RS/TP in year Y (unique lakes)
    lakes_rs_in_y = set(ev_rs.loc[ev_rs["year"] == Y, "lake_id"].astype(str))
    lakes_tp_in_y = set(ev_tp.loc[ev_tp["year"] == Y, "lake_id"].astype(str))

    n_rs_elig = len(rs_elig_lakes)
    n_tp_elig = len(tp_elig_lakes)
    n_rs_lakes = len(lakes_rs_in_y & rs_elig_lakes)
    n_tp_lakes = len(lakes_tp_in_y & tp_elig_lakes)

    rs_rate = (n_rs_lakes / n_rs_elig) if n_rs_elig > 0 else np.nan
    tp_rate = (n_tp_lakes / n_tp_elig) if n_tp_elig > 0 else np.nan

    rows.append({
        "year": Y,
        "n_lakes_eligible_RS": n_rs_elig,
        "n_lakes_eligible_TP": n_tp_elig,
        "n_lakes_with_RS": n_rs_lakes,
        "n_lakes_with_TP": n_tp_lakes,
        "rs_rate": rs_rate,
        "tp_rate": tp_rate
    })

out = pd.DataFrame(rows)
out_path = output_dir / "rs_tp_rates_by_year.csv"
out.to_csv(out_path, index=False)
print(f"Wrote: {out_path}")

# Spearman
use_rs = out["n_lakes_eligible_RS"] > 0
use_tp = out["n_lakes_eligible_TP"] > 0
yrs_rs = out.loc[use_rs, "year"].to_numpy()
yrs_tp = out.loc[use_tp, "year"].to_numpy()
r_rs, p_rs = spearman(yrs_rs, out.loc[use_rs, "rs_rate"].to_numpy())
r_tp, p_tp = spearman(yrs_tp, out.loc[use_tp, "tp_rate"].to_numpy())

print(
    f"Regime Shift Rate vs Year (usable years={use_rs.sum()}): rho={r_rs if np.isfinite(r_rs) else np.nan:.3f}, p={p_rs if np.isfinite(p_rs) else np.nan:.3f}")
print(
    f"Tipping Point Rate vs Year (usable years={use_tp.sum()}): rho={r_tp if np.isfinite(r_tp) else np.nan:.3f}, p={p_tp if np.isfinite(p_tp) else np.nan:.3f}")

print("\nHead:")
print(out.head(8).to_string(index=False))
