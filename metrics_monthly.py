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
bootstrap_num = 500
alpha = 0.05
np.random.seed(42)

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
    return float((xm * ym).sum() / den)

def p_approx_from_t(tstat):
    if not np.isfinite(tstat):
        return np.nan
    z = abs(float(tstat))
    return math.erfc(z / math.sqrt(2.0))

def detrend_by_month(tsi, years, months):
    df = pd.DataFrame({"TSI": tsi, "year": years, "month": months})
    resid = np.full(len(df), np.nan, float)
    for m in range(1, 13):
        d = df[df["month"] == m].copy()
        if d.empty:
            continue
        x = d["year"].to_numpy(dtype=float)
        y = d["TSI"].to_numpy(dtype=float)
        k = d.index.to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            if ok.any():
                resid[k[ok]] = y[ok] - np.nanmean(y[ok])
            continue
        X = np.column_stack([np.ones(ok.sum()), x[ok]-x[ok].mean()])
        b0, b1 = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        yhat = b0 + b1*(x - x[ok].mean())
        resid[k] = y - yhat
    return resid

def rolling_var(y, win):
    return pd.Series(y).rolling(win, min_periods=max(3, win//2)).var().to_numpy()

def bootstrap_fill_centered(resid, tnum, halfwin=6, B=bootstrap_num):
    rhos = []
    n = len(resid)
    for _ in range(B):
        filled = resid.copy()
        for i in range(n):
            if np.isfinite(filled[i]):
                continue
            lo = tnum[i] - halfwin
            hi = tnum[i] + halfwin
            w = (tnum >= lo) & (tnum <= hi) & np.isfinite(resid)
            if w.sum() == 0:
                w = np.isfinite(resid)
            picks = resid[w]
            filled[i] = np.random.choice(picks, 1)[0] if picks.size else np.nan
        vts = rolling_var(filled, variation_window)
        rhos.append(spearman_rho(tnum, vts))
    r = np.array(rhos, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan, np.nan, 0, np.nan, np.nan
    mean_rho = float(np.nanmean(r))
    std_rho = float(np.nanstd(r, ddof=1)) if r.size > 1 else np.nan
    n_eff = int(r.size)
    t_stat = mean_rho / (std_rho / np.sqrt(n_eff)
                         ) if (std_rho and n_eff > 1) else np.nan
    p = p_approx_from_t(t_stat)
    return mean_rho, std_rho, n_eff, t_stat, p

panel = pd.read_csv(panel_dir, low_memory=False)
need = {"lake_id", "year", "month", "TSI"}
miss = [c for c in need if c not in panel.columns]
if miss:
    raise SystemExit(f"Panel missing columns: {miss}")

for c in ["year", "month", "TSI", "TEMP", "prop_agri"]:
    if c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

panel = panel.dropna(subset=["lake_id", "year", "month"])
panel = panel[(panel["year"] >= start_year) & (panel["year"] <= end_year)]
panel["tnum"] = panel["year"].astype(int)*12 + panel["month"].astype(int)

rows = []
for lake_id, g in panel.groupby("lake_id", sort=False):
    g = g.sort_values(["year", "month"]).copy()
    tnum = g["tnum"].to_numpy()
    tsi = g["TSI"].to_numpy(dtype=float)
    years = g["year"].to_numpy(dtype=int)
    months = g["month"].to_numpy(dtype=int)
    n_obs = np.isfinite(tsi).sum()
    if n_obs < minimum_months:
        continue

    miss_frac = 1.0 - (n_obs / float(len(tsi)))
    avg_tsi = float(np.nanmean(tsi))
    rho_tsi = spearman_rho(tnum, tsi)
    n = np.isfinite(tsi).sum()
    p_tsi = p_approx_from_t(
        rho_tsi * np.sqrt((n-2)/(1 - rho_tsi**2))) if (n and abs(rho_tsi) < 1) else np.nan

    resid = detrend_by_month(tsi, years, months)
    sd_resid = float(np.nanstd(resid, ddof=1))
    cv_resid = (sd_resid / avg_tsi) if (np.isfinite(avg_tsi)
                                        and avg_tsi != 0) else np.nan

    m_rho, s_rho, n_rho, t_rho, p_rho = bootstrap_fill_centered(
        resid, tnum, halfwin=6, B=bootstrap_num)

    rows.append({
        "lake_id": lake_id,
        "n_months_obs": int(n_obs),
        "miss_frac": float(miss_frac),
        "avg_TSI": avg_tsi,
        "trend_TSI_rho": rho_tsi,
        "trend_TSI_p_approx": p_tsi,
        "CV_resid": cv_resid,
        "vartrend_rho_mean": m_rho,
        "vartrend_rho_std": s_rho,
        "vartrend_rho_n": n_rho,
        "vartrend_t_stat": t_rho,
        "vartrend_p_approx": p_rho,
        "vartrend_sig_pos": bool(np.isfinite(m_rho) and np.isfinite(p_rho) and (m_rho > 0) and (p_rho < alpha))
    })

out = output_dir / "tsi_metrics__monthly.csv"
df = pd.DataFrame(rows)
df.to_csv(out, index=False)

print(f"Wrote {out} (lakes with > {minimum_months} months: {len(df):,})")
if len(df):
    print("Summary (monthly metrics):")
    print("Mean(avg_TSI) =", round(float(df["avg_TSI"].mean()), 3))
    print("Mean(CV_resid) =", round(float(df["CV_resid"].mean()), 3))
    sig = df["vartrend_sig_pos"].sum()
    print(
        f"  sig positive variance-trend (@alpha={alpha}): {sig} / {len(df)} ({100*sig/len(df):.1f}%)")
    ex = df.sort_values("vartrend_rho_mean", ascending=False).head(
        5)[["lake_id", "vartrend_rho_mean", "vartrend_p_approx"]]
    print("Top 5 vartrend_rho_mean:")
    print(ex.to_string(index=False))
