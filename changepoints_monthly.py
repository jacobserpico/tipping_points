"""
Author: @ Jacob Serpico
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd

data_dir = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data")
panel_dir = data_dir / "panels" / "lake_month_panel__1992-2022.csv"
output_dir = data_dir / "analysis" / "gilarranz_replication"
output_dir.mkdir(parents=True, exist_ok=True)

start_year, end_year = 1992, 2022
minimum_months = 96
variation_window = 12
half_window_fill = 6
edge_exclusion = 24
maximum_changepoints = 3
minimum_segments = 24
bootstraps = 500
alpha = 0.05
np.random.seed(42)

def detrend_by_month(tsi, years, months):
    df = pd.DataFrame({"TSI": tsi, "year": years, "month": months})
    resid = np.full(len(df), np.nan, float)
    for m in range(1, 13):
        d = df[df["month"] == m].copy()
        if d.empty:
            continue
        x = d["year"].to_numpy(float)
        y = d["TSI"].to_numpy(float)
        k = d.index.to_numpy(int)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            if ok.any():
                resid[k[ok]] = y[ok] - np.nanmean(y[ok])
            continue
        X = np.column_stack([np.ones(ok.sum()), x[ok] - x[ok].mean()])
        b0, b1 = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        yhat = b0 + b1 * (x - x[ok].mean())
        resid[k] = y - yhat
    return resid

def bootstrap_gapfill_mean(resid, tnum, halfwin=half_window_fill, B=bootstraps):
    n = len(resid)
    draws = np.full((B, n), np.nan, float)
    obs_mask = np.isfinite(resid)
    draws[:, obs_mask] = resid[obs_mask]
    for b in range(B):
        fill = resid.copy()
        for i in range(n):
            if np.isfinite(fill[i]):
                continue
            lo = tnum[i] - halfwin
            hi = tnum[i] + halfwin
            w = (tnum >= lo) & (tnum <= hi) & np.isfinite(resid)
            if w.sum() == 0:
                w = np.isfinite(resid)
            picks = resid[w]
            fill[i] = np.random.choice(picks, 1)[0] if picks.size else np.nan
        draws[b] = fill
    return np.nanmean(draws, axis=0)

def rolling_var(y, win):
    return pd.Series(y).rolling(win, min_periods=max(3, win//2)).var().to_numpy()

def spearman_rho(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    xr = pd.Series(x[m]).rank().to_numpy()
    yr = pd.Series(y[m]).rank().to_numpy()
    xm = xr - xr.mean()
    ym = yr - yr.mean()
    den = (np.sqrt((xm**2).sum()) * np.sqrt((ym**2).sum()))
    if den == 0:
        return np.nan
    return float((xm*ym).sum()/den)

def vartrend_pre_cp(resid_filled, tnum, cp_idx, win=variation_window, B=bootstraps):
    pre_y = resid_filled[:cp_idx+1].copy()
    pre_t = tnum[:cp_idx+1].copy()
    rhos = []
    for _ in range(B):
        fill = pre_y.copy()
        for i in range(len(fill)):
            if np.isfinite(fill[i]):
                continue
            lo = pre_t[i] - half_window_fill
            hi = pre_t[i] + half_window_fill
            w = (pre_t >= lo) & (pre_t <= hi) & np.isfinite(pre_y)
            if w.sum() == 0:
                w = np.isfinite(pre_y)
            picks = pre_y[w]
            fill[i] = np.random.choice(picks, 1)[0] if picks.size else np.nan
        v = rolling_var(fill, win)
        rhos.append(spearman_rho(pre_t, v))
    rhos = np.asarray(rhos, float)
    rhos = rhos[np.isfinite(rhos)]
    if not rhos.size:
        return np.nan, np.nan
    return float(rhos.mean()), float(rhos.std(ddof=1)) if rhos.size > 1 else np.nan

def mean_shift_cps(y, maximum_changepoints=maximum_changepoints, minimum_segments=minimum_segments):
    idxs = []

    def seg_cost(a, b, arr):
        s = arr[a:b]
        m = np.nanmean(s)
        return float(np.nansum((s-m)**2))

    def best_split(a, b, arr):
        best = None
        bestc = np.inf
        for k in range(a+minimum_segments, b-minimum_segments+1):
            c = seg_cost(a, k, arr) + seg_cost(k, b, arr)
            if c < bestc:
                bestc = c
                best = k
        return best, bestc

    def split_rec(a, b, arr, depth):
        if depth == 0 or b-a < 2*minimum_segments:
            return
        k, c = best_split(a, b, arr)
        if k is None:
            return
        idxs.append(k)
        split_rec(a, k, arr, depth-1)
        split_rec(k, b, arr, depth-1)
    arr = np.asarray(y, float)
    split_rec(0, len(arr), arr, maximum_changepoints)
    idxs = sorted(set(int(i) for i in idxs))
    return idxs[:maximum_changepoints]

def abruptness(y, k):
    yb = y[:k]
    ya = y[k:]
    if len(yb) < 3 or len(ya) < 3:
        return np.nan
    js = np.nanmean(ya) - np.nanmean(yb)
    sb = float(np.nanstd(yb, ddof=1))
    sa = float(np.nanstd(ya, ddof=1))
    den = 0.5*(sb+sa) if (np.isfinite(sb) and np.isfinite(sa)) else np.nan
    if not np.isfinite(den) or den == 0:
        return np.nan
    return abs(js)/den

def welch_t_p(yb, ya):
    b = pd.Series(yb).dropna()
    a = pd.Series(ya).dropna()
    if len(b) < 3 or len(a) < 3:
        return np.nan
    mb, ma = b.mean(), a.mean()
    vb, va = b.var(ddof=1), a.var(ddof=1)
    nb, na = len(b), len(a)
    se = math.sqrt(vb/nb + va/na)
    if se == 0:
        return np.nan
    t = (ma-mb)/se
    return math.erfc(abs(t)/math.sqrt(2.0))

panel = pd.read_csv(panel_dir, low_memory=False)
panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
panel["month"] = pd.to_numeric(panel["month"], errors="coerce")
panel = panel.dropna(subset=["lake_id", "year", "month"])
panel = panel[(panel["year"] >= start_year) & (panel["year"] <= end_year)]
panel["tnum"] = panel["year"].astype(int)*12 + panel["month"].astype(int)

rows = []
tp_counts = []

for lake_id, g in panel.groupby("lake_id", sort=False):
    g = g.sort_values(["year", "month"]).copy()
    tsi = pd.to_numeric(g["TSI"], errors="coerce").to_numpy()
    if np.isfinite(tsi).sum() < minimum_months:
        continue
    years = g["year"].to_numpy(int)
    months = g["month"].to_numpy(int)
    tnum = g["tnum"].to_numpy(int)

    resid = detrend_by_month(tsi, years, months)
    filled = bootstrap_gapfill_mean(
        resid, tnum, halfwin=half_window_fill, B=bootstraps)

    cps = mean_shift_cps(filled, maximum_changepoints=maximum_changepoints, minimum_segments=minimum_segments)
    cps = [k for k in cps if not (
        k < edge_exclusion or (len(filled)-k) < edge_exclusion)]

    n_tp = 0
    for k in cps:
        yb = filled[:k]
        ya = filled[k:]
        a = abruptness(filled, k)
        pmean = welch_t_p(yb, ya)
        is_rs = (np.isfinite(a) and a > 1.0) and (
            np.isfinite(pmean) and pmean < 0.05)

        is_tp = False
        if is_rs:
            m, s = vartrend_pre_cp(filled, tnum, k, win=variation_window, B=bootstraps)
            if np.isfinite(m) and np.isfinite(s) and s > 0:
                t = m/(s/np.sqrt(bootstraps))
                p = math.erfc(abs(t)/math.sqrt(2.0))
                is_tp = (m > 0) and (p < alpha)

        rows.append({
            "lake_id": lake_id,
            "year": int(g.iloc[k]["year"]),
            "month": int(g.iloc[k]["month"]),
            "is_regimeshift": bool(is_rs),
            "is_tipping": bool(is_tp),
            "abruptness": float(a) if np.isfinite(a) else np.nan,
            "welch_p": float(pmean) if np.isfinite(pmean) else np.nan
        })
        if is_tp:
            n_tp += 1

    tp_counts.append({"lake_id": lake_id, "n_tp": int(n_tp)})

ev = pd.DataFrame(rows)
ev_path = output_dir / "candidate_changes__monthly.csv"
ev.to_csv(ev_path, index=False)

tp = pd.DataFrame(tp_counts)
tp["tipped"] = (tp["n_tp"] > 0).astype(int)
tp_path = output_dir / "candidate_tipping__lake_summary.csv"
tp.to_csv(tp_path, index=False)

if len(ev):
    ann = (ev.groupby(["year"], as_index=False)
             .agg(rs=("is_regimeshift", "sum"), tp=("is_tipping", "sum")))
    ann_path = output_dir / "rs_tp_counts_by_year.csv"
    ann.to_csv(ann_path, index=False)
else:
    ann = pd.DataFrame(columns=["year", "rs", "tp"])

# Terminal
print(f"Wrote: {ev_path}")
print(f"Wrote: {tp_path}")
if len(ann):
    print(f"Wrote: {ann_path}")
print("Diagnostics:")
print("Lakes with > one tipping:", int((tp["n_tp"] > 0).sum()))
print("Total candidate CPs:", len(ev))
print("Regime shifts:", int(ev["is_regimeshift"].sum()))
print("Tipping events:", int(ev["is_tipping"].sum()))
if len(ann):
    tail = ann.sort_values("year").tail(10)
    print("Last 10 years (rs, tp):")
    print(tail.to_string(index=False))
ex = ev[ev["is_tipping"] == True].head(
    8)[["lake_id", "year", "month", "abruptness", "welch_p"]]
if len(ex):
    print("Sample tipping events:")
    print(ex.to_string(index=False))
