"""
Author: @ Jacob Serpico
"""

from pathlib import Path
import sys
import time
import re
import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pyproj import CRS, Transformer
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# File paths
data_dir = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data")
water_csv = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data/aggregate_water_quality/aggregate_water_quality__1990-2024__with_water_parameters.csv")
land_csv = Path("/Users/Jake/Desktop/Everything/Research/tipping_points/data/land_use_data/GEMStat_downloads/e4c9c120c7273be007069e818811434c")
output_dir = data_dir / "land_use_data" / "lake_landuse_allyears.csv"

buffers = [500, 1000, 1500, 3000]
include_grassland = False
overwrite_csv = True

# Parallelization
year_workers = max(1, min(4, (os.cpu_count() or 4)//3))  # Processes (years in parallel)
lake_workers = 6  # Threads per process (lakes in parallel)

# Optional quick-run toggles
debug_run = False
debug_years = 1
debug_lakes = 50
print_statements = 500

latitude_candidates = ["LatitudeDecimalDegrees", "latitude", "lat", "Latitude"]
longitude_candidates = ["LongitudeDecimalDegrees", "longitude", "lon", "Longitude"]
id_candidates = ["gems_station_number", "station_id", "StationID", "lake_id"]
name_candidates = ["lake_name", "LakeName", "LAKE_NAME"]

# Land cover buckets
land_cover_classes = {
    10: "crops", 20: "crops", 30: "crops",
    40: "forestry", 50: "forestry", 60: "forestry", 70: "forestry", 80: "forestry", 90: "forestry", 100: "forestry",
    110: "grassland", 120: "grassland", 121: "grassland", 122: "grassland", 130: "grassland", 140: "grassland", 150: "grassland",
    160: "wetland", 170: "wetland",
    190: "built",
    200: "bare",
    210: "water",
    220: "bare",
}
buckets = ["crops", "forestry", "grassland",
           "built", "wetland", "water", "bare"]

# Helper functions

def flush(msg):
    print(msg, flush=True)

def find_column(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def laea_crs(lat, lon):
    return CRS.from_proj4(f"+proj=laea +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")

def parse_year_from_string(s):
    yrs = re.findall(r'(?:19|20)\d{2}', s)
    return int(yrs[-1]) if yrs else None


def circle_bbox_deg(lat0, lon0, r_m):
    dlat = r_m / 111000.0
    clat = max(0.05, np.cos(np.deg2rad(lat0)))
    dlon = r_m / (111000.0 * clat)
    return lat0 - dlat, lat0 + dlat, lon0 - dlon, lon0 + dlon

def read_lake_points(water_csv):
    flush(f"[{time.strftime('%H:%M:%S')}] Reading water_csv: {water_csv}")
    if not water_csv.exists():
        raise SystemExit(f"File not found: {water_csv}")
    df = pd.read_csv(water_csv, dtype=str, low_memory=False)
    latc = find_column(df, latitude_candidates)
    lonc = find_column(df, longitude_candidates)
    idc = find_column(df, id_candidates) or "point_id"
    namec = find_column(df, name_candidates)
    if not latc or not lonc:
        raise SystemExit("Latitude/longitude columns not found in water_csv.")
    pts = df[[idc, latc, lonc]].dropna().copy()
    pts.columns = ["lake_id", "lat", "lon"]
    pts["lake_name"] = df[namec] if namec and namec in df.columns else ""
    pts["lat"] = pd.to_numeric(pts["lat"], errors="coerce")
    pts["lon"] = pd.to_numeric(pts["lon"], errors="coerce")
    pts = pts.dropna(subset=["lat", "lon"]).drop_duplicates(
        subset=["lake_id", "lat", "lon"])
    gdf = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(
        pts["lon"], pts["lat"]), crs=4326)
    # return simple columns to reduce pickling weight
    out = gdf[["lake_id", "lake_name", "geometry"]].copy()
    out["lat"] = out.geometry.y
    out["lon"] = out.geometry.x
    flush(f"[{time.strftime('%H:%M:%S')}] Lakes loaded: {len(out):,}")
    return out[["lake_id", "lake_name", "lat", "lon"]].reset_index(drop=True)

def load_nc_by_years(root):
    flush(f"[scan] land_csv: {root}")
    files = sorted(Path(root).rglob("*.nc*"))
    flush(f"[scan] NetCDF files found: {len(files)}")
    nc = {}
    for p in files:
        y = parse_year_from_string(p.name) or parse_year_from_string(str(p))
        if y is None:
            try:
                ds = xr.open_dataset(p, decode_times=True)
                if "time" in ds.coords and ds["time"].size > 0:
                    y = pd.to_datetime(ds["time"].values[0]).year
                ds.close()
            except Exception:
                y = None
        if y is None:
            flush(f"[warn] could not infer year for {p.name}")
            continue
        nc.setdefault(y, []).append(p)
    if nc:
        counts = [(y, len(v)) for y, v in sorted(nc.items())]
        flush("[scan] per-year NetCDF counts: " +
              ", ".join([f"{y}:{n}" for y, n in counts]))
    return nc

def open_year_ds(paths, verbose=False):
    if not paths:
        return None, None
    p0 = paths[0]
    ds = xr.open_dataset(p0, decode_times=False)
    int_like = [k for k in ds.data_vars if ds[k].dtype.kind in "iu"]
    var = "lccs_class" if "lccs_class" in ds.data_vars else (
        int_like[0] if int_like else list(ds.data_vars)[0])
    if verbose:
        flush(
            f"  Using var='{var}' from {p0.name}; dims={list(ds[var].dims)} dtype={ds[var].dtype}")
    return ds, var

def get_lat_lon_dims(ds, var):
    da = ds[var]
    dims = list(da.dims)
    latname = [d for d in dims if "lat" in d.lower()][0]
    lonname = [d for d in dims if "lon" in d.lower()][0]

    other = [d for d in dims if d not in [latname, lonname]]
    if other:
        da = da.isel({d: 0 for d in other})

    lat = np.array(ds[latname].values).astype(float)
    lon = np.array(ds[lonname].values).astype(float)

    if (lon.min() >= 0) and (lon.max() > 180):
        lon2 = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(lon2)
        lon = lon2[order]
        da = da.isel({lonname: order})

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        da = da.isel({latname: slice(None, None, -1)})
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        da = da.isel({lonname: slice(None, None, -1)})

    da = da.squeeze(drop=True)
    return da, lat, lon, latname, lonname

# Parallel workers
def lake_rows_for_year(lakes_pack, year, da, lat, lon, latname, lonname):
    lids = lakes_pack["lake_id"]
    names = lakes_pack["lake_name"]
    lats = lakes_pack["lat"]
    lons = lakes_pack["lon"]

    r_max = max(buffers)
    out_rows = []

    def work(i):
        lat0 = float(lats[i])
        lon0 = float(lons[i])
        # bbox in degrees for windowing
        minlat, maxlat, minlon, maxlon = circle_bbox_deg(lat0, lon0, r_max)
        iy0 = max(0, np.searchsorted(lat, minlat, side="left"))
        iy1 = min(len(lat), np.searchsorted(lat, maxlat, side="right"))
        ix0 = max(0, np.searchsorted(lon, minlon, side="left"))
        ix1 = min(len(lon), np.searchsorted(lon, maxlon, side="right"))
        if iy0 >= iy1 or ix0 >= ix1:
            rows = []
            for r in buffers:
                rows.append({
                    "lake_id": lids[i], "lake_name": names[i], "year": int(year),
                    "buffer_m": int(r), "total_pixels": 0,
                    "prop_agri": 0.0, "prop_agri_land": 0.0,
                    **{f"frac_{b}": 0.0 for b in buckets},
                    **{f"n_{b}": 0 for b in buckets},
                })
            return rows

        win = da.isel({latname: slice(iy0, iy1),
                      lonname: slice(ix0, ix1)}).values
        win = np.squeeze(win)
        while win.ndim > 2:
            win = win[0]
        if win.ndim != 2:
            win = np.atleast_2d(win)

        lat_sub = lat[iy0:iy1]
        lon_sub = lon[ix0:ix1]
        xs, ys = np.meshgrid(lon_sub, lat_sub)

        # vectorized LAEA distances
        laea = laea_crs(lat0, lon0)
        transf = Transformer.from_crs(4326, laea, always_xy=True)
        x, y = transf.transform(xs, ys)
        x0, y0 = transf.transform(lon0, lat0)
        dx = x - x0
        dy = y - y0
        dist2 = dx*dx + dy*dy

        rows = []
        for r in buffers:
            mask = dist2 <= (r*r)
            vals = win[mask]
            tot = int(mask.sum())

            if tot == 0 or vals.size == 0:
                row = {
                    "lake_id": lids[i], "lake_name": names[i], "year": int(year),
                    "buffer_m": int(r), "total_pixels": 0,
                    "prop_agri": 0.0, "prop_agri_land": 0.0,
                }
                for b in buckets:
                    row[f"frac_{b}"] = 0.0
                    row[f"n_{b}"] = 0
                rows.append(row)
                continue

            u, c = np.unique(vals, return_counts=True)
            counts = {b: 0 for b in buckets}
            for uu, cc in zip(u, c):
                b = land_cover_classes.get(int(uu))
                if b:
                    counts[b] += int(cc)

            fr = {b: (counts[b] / tot) for b in buckets}
            num = fr["crops"] + \
                (fr["grassland"] if include_grassland else 0.0)
            denom_land = max(1e-12, 1.0 - fr["water"] - fr["wetland"])
            prop_agri = num
            prop_agri_land = num / denom_land

            row = {
                "lake_id": lids[i],
                "lake_name": names[i],
                "year": int(year),
                "buffer_m": int(r),
                "total_pixels": int(tot),
                "prop_agri": float(prop_agri),
                "prop_agri_land": float(min(max(prop_agri_land, 0.0), 1.0)),
            }
            for b in buckets:
                row[f"frac_{b}"] = fr[b]
                row[f"n_{b}"] = counts[b]
            rows.append(row)

        return rows

    with ThreadPoolExecutor(max_workers=lake_workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(lids))]
        done = 0
        for f in as_completed(futs):
            out_rows.extend(f.result())
            done += 1
            if (done % print_statements) == 0:
                flush(f"    progress: {done}/{len(lids)} lakes")
    return out_rows


def process_one_year(year, paths, lakes_pack, verbose_first):
    # child process entry
    ds, var = open_year_ds(paths, verbose=verbose_first)
    if ds is None:
        return year, []
    da, lat, lon, latname, lonname = get_lat_lon_dims(ds, var)
    rows = lake_rows_for_year(
        lakes_pack, year, da, lat, lon, latname, lonname)
    ds.close()
    return year, rows

# Main
def main():
    flush(
        f"Land_use_dataset_creator start [{time.strftime('%Y-%m-%d %H:%M:%S')}].")
    flush(f"data_dir={data_dir}")
    flush(f"water_csv={water_csv}")
    flush(f"land_csv={land_csv}")
    flush(
        f"Buffers={buffers}. Include_grassland={include_grassland}")
    flush(
        f"debug_run={debug_run}. year_workers={year_workers} | lake_workers={lake_workers}")

    lakes = read_lake_points(water_csv)
    if debug_run:
        lakes = lakes.head(debug_lakes).copy()
        flush(f"[debug] restricting to first {len(lakes)} lakes")

    nc = load_nc_by_years(land_csv)
    if not nc:
        flush("No NetCDF years found. Exiting.")
        return

    years = sorted(nc.keys())
    if debug_run:
        years = years[:debug_years]
        flush(f"[debug] restricting to years: {years}")

    if overwrite_csv and output_dir.exists():
        try:
            output_dir.unlink()
        except Exception as e:
            flush(f"Could not remove existing output_dir ({output_dir}): {e}")

    # pack lakes as simple arrays to minimize pickling cost
    lakes_pack = {
        "lake_id": lakes["lake_id"].tolist(),
        "lake_name": lakes["lake_name"].tolist(),
        "lat": lakes["lat"].tolist(),
        "lon": lakes["lon"].tolist(),
    }

    flush(f"[{time.strftime('%H:%M:%S')}] Launching {len(years)} year task(s) with {year_workers} process(es)")
    all_rows = []
    with ProcessPoolExecutor(max_workers=year_workers) as pex:
        futs = []
        first = True
        for y in years:
            futs.append(pex.submit(process_one_year,
                        y, nc[y], lakes_pack, first))
            first = False
        for f in as_completed(futs):
            try:
                y, rows = f.result()
                flush(
                    f"[{time.strftime('%H:%M:%S')}] Year {y}: completed; rows={len(rows):,}")
                all_rows.extend(rows)
            except Exception as e:
                flush(f"[warn] year task failed: {repr(e)}")

    out = pd.DataFrame(all_rows)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir, index=False)
    flush(f"Wrote: {output_dir}")
    if len(out):
        flush(
            f"Rows: {len(out):,} | Lakes: {out['lake_id'].nunique():,} | Years: {out['year'].nunique():,} | Radii: {sorted(out['buffer_m'].unique())}")
        keep = [c for c in out.columns if c.startswith('prop_agri')][:6]
        flush("Sample columns: " + ", ".join(keep))
    flush("=== land_use_dataset_creator done ===")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
