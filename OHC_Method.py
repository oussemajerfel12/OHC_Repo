#!/usr/bin/env python3
"""
compute_ohc.py
Robust OHC computation script adapted from the notebook.

Usage examples:
    python compute_ohc.py \
      --mask_woa blue-cloud-dataspace/MEI/INGV/INPUT/BATHYMETRY/gebco_2019_mask_1_8_edited_final_woa13.nc \
      --mask_local blue-cloud-dataspace/MEI/INGV/INPUT/BATHYMETRY/gebco_2019_mask_1_8_edited_final.nc \
      --temp blue-cloud-dataspace/MEI/INGV/INPUT/CLIMATOLOGY/Temperature_sliding_climatology_WP.nc \
      --outdir output_ohc \
      --year_min 1980 --year_max 2014
"""

import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Physical constants
RHO = 1030.0
CP = 3980.0

# default target grid (same as in your notebook)
DX = DY = 0.125
LONR = np.arange(-5.625, 36.5 + DX, DX)
LATR = np.arange(30.0, 46.0 + DY, DY)
DEPTHR = np.array([
    0., 5., 10., 15., 20., 25., 30., 35., 40., 45.,
    50., 55., 60., 65., 70., 75., 80., 85., 90., 95.,
    100., 125., 150., 175., 200., 225., 250., 275., 300., 325.,
    350., 375., 400., 425., 450., 475., 500., 550., 600., 650.,
    700., 750., 800., 850., 900., 950., 1000., 1050., 1100., 1150.,
    1200., 1250., 1300., 1350., 1400., 1450., 1500., 1550., 1600., 1650.,
    1700., 1750., 1800., 1850., 1900., 1950., 2000.
])

# -------------------------
# Helpers
# -------------------------
def find_coord(ds, candidates):
    for c in candidates:
        if c in ds.coords:
            return c
    return None

def find_var(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    # fallback: first numeric var
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.number):
            return v
    return None

def compute_dz(depth_vals):
    dv = np.asarray(depth_vals)
    if dv.size == 0:
        return np.array([1.0])
    if dv.size == 1:
        return np.array([dv[0] if dv[0] > 0 else 1.0])
    edges = np.zeros(dv.size + 1)
    edges[1:-1] = 0.5 * (dv[:-1] + dv[1:])
    edges[0] = dv[0] - (edges[1] - dv[0])
    edges[-1] = dv[-1] + (dv[-1] - edges[-2])
    dz = edges[1:] - edges[:-1]
    dz[dz <= 0] = np.abs(np.diff(np.concatenate(([0.0], dv))))[0]
    return dz

def open_ds(path):
    return xr.open_dataset(path, decode_times=True, mask_and_scale=True)

def compute_grid_area_from_deltas(delta_lon, delta_lat, deg_to_m=111319.5):
    """
    If user has delta_lon/delta_lat arrays (degrees), convert to m^2 (approx) using deg_to_m.
    delta_lon/delta_lat may be 2D arrays or 3D; this function expects pre-masked horizontal sizes.
    """
    # deg multiplier roughly meters per degree at equator; fine approximation for small med area
    # Accept arrays in degrees and return m^2 (elementwise product)
    return (delta_lon * deg_to_m) * (delta_lat * deg_to_m)

# -------------------------
# Core compute function
# -------------------------
def compute_ohc(ds_mask_woa13, ds_mask_local, ds_temp,
                lonr=LONR, latr=LATR, depthr=DEPTHR,
                rho=RHO, cp=CP, year_min=None, year_max=None, area_da=None):
    """
    Compute OHC and return:
      - ohc_grid: xarray DataArray (time?, lat, lon) J m^-2
      - ohc_ts: xarray DataArray (time) area-weighted mean J m^-2 (or simple mean if no area provided)
    """

    # 1) detect temperature variable & coordinates
    temp_candidates = ["t_an", "temperature", "thetao", "TEMP", "temp", "t"]
    temp_var = find_var(ds_temp, temp_candidates)
    if temp_var is None:
        raise ValueError("No numeric temperature variable detected in ds_temp")
    temp_da = ds_temp[temp_var]
    lon_name = find_coord(ds_temp, ["lon", "longitude", "LONGITUDE", "LONG"])
    lat_name = find_coord(ds_temp, ["lat", "latitude", "LATITUDE", "LAT"])
    depth_name = find_coord(ds_temp, ["depth", "z", "DEPTH", "depths"])
    time_exists = "time" in ds_temp.coords

    print("[compute_ohc] temp_var:", temp_var)
    print("[compute_ohc] coords detected:", {"lon": lon_name, "lat": lat_name, "depth": depth_name, "time": "yes" if time_exists else "no"})

    # 2) optional time subset on original ds_temp (safer than on interpolated)
    if time_exists and (year_min is not None or year_max is not None):
        try:
            # try datetime-like selection
            years = temp_da["time"].dt.year
            sel_mask = np.ones(len(years), dtype=bool)
            if year_min is not None:
                sel_mask &= (years >= int(year_min))
            if year_max is not None:
                sel_mask &= (years <= int(year_max))
            ds_temp = ds_temp.isel(time=np.where(sel_mask)[0])
            temp_da = ds_temp[temp_var]
            print(f"[compute_ohc] time-subset original ds_temp -> {len(temp_da.time)} steps")
        except Exception:
            # fallback if time is numeric years
            try:
                tvals = temp_da["time"].values.astype(int)
                sel_mask = np.ones_like(tvals, dtype=bool)
                if year_min is not None:
                    sel_mask &= (tvals >= int(year_min))
                if year_max is not None:
                    sel_mask &= (tvals <= int(year_max))
                ds_temp = ds_temp.isel(time=np.where(sel_mask)[0])
                temp_da = ds_temp[temp_var]
                print(f"[compute_ohc] time-subset fallback -> {len(temp_da.time)} steps")
            except Exception:
                print("[compute_ohc] could not subset time by year - proceeding without time subset")

    # 3) interpolation planning
    interp_kwargs = {}
    if lon_name is not None:
        interp_kwargs["lon"] = lonr
    if lat_name is not None:
        interp_kwargs["lat"] = latr
    if depth_name is not None and depthr is not None:
        interp_kwargs["depth"] = depthr
    print("[compute_ohc] interp target dims:", list(interp_kwargs.keys()))

    # 4) attempt to make base_mask & local mask on target grid
    base_mask = None
    local_mask = None
    try:
        base_mask = ds_mask_woa13["mask"]
        if interp_kwargs:
            base_mask = base_mask.interp(method="nearest", **interp_kwargs)
    except Exception as e:
        print("[compute_ohc] base_mask interp failed:", e)
    try:
        local_mask = ds_mask_local["mask"]
        if interp_kwargs:
            local_mask = local_mask.interp(method="nearest", **interp_kwargs)
    except Exception as e:
        print("[compute_ohc] local_mask interp failed:", e)

    # 5) attempt to interpolate full temperature dataset (fast path)
    temp_interp = temp_da
    try:
        if interp_kwargs:
            # try linear then nearest fallback
            try:
                temp_interp = temp_da.interp(method="linear", **interp_kwargs)
            except Exception:
                temp_interp = temp_da.interp(method="nearest", **interp_kwargs)
        print("[compute_ohc] attempted interpolation; resulting time shape:",
              (temp_interp.time.size if "time" in temp_interp.coords else "no-time"))
    except Exception as e:
        print("[compute_ohc] full-interp failed -> will attempt per-slice interpolation. error:", e)
        temp_interp = temp_da  # keep original; will fallback later

    # 6) prepare combined mask (logical AND of masks if available)
    if (local_mask is not None) and (base_mask is not None):
        combined_mask = xr.where((local_mask == 1) & (base_mask == 1), 1, 0)
    elif local_mask is not None:
        combined_mask = xr.where(local_mask == 1, 1, 0)
    elif base_mask is not None:
        combined_mask = xr.where(base_mask == 1, 1, 0)
    else:
        # fallback: ones on the horizontal dims present in temp_interp or temp_da
        print("[compute_ohc] no mask found: building full-ones mask on target grid")
        sample = temp_interp.isel(time=0) if ("time" in temp_interp.coords and temp_interp.time.size>0) else temp_da
        # create mask with lat/lon/depth dims where present
        coords = {}
        if "lat" in sample.coords:
            coords["lat"] = sample["lat"].values
        else:
            coords["lat"] = latr
        if "lon" in sample.coords:
            coords["lon"] = sample["lon"].values
        else:
            coords["lon"] = lonr
        if "depth" in sample.coords:
            coords["depth"] = sample["depth"].values
        combined_mask = xr.DataArray(np.ones((len(coords["lat"]), len(coords["lon"]), len(coords.get("depth", [0]))), dtype=int),
                                     coords={"lat": coords["lat"], "lon": coords["lon"], "depth": coords.get("depth", [0])},
                                     dims=("lat","lon","depth") if "depth" in sample.coords else ("lat","lon"))

    # 7) dz array (depth thickness)
    if "depth" in temp_interp.coords:
        depth_vals = temp_interp["depth"].values
    elif "depth" in ds_temp.coords:
        depth_vals = ds_temp["depth"].values
    else:
        depth_vals = depthr
    dz_vals = compute_dz(depth_vals)
    dz_da = xr.DataArray(dz_vals, coords={"depth": depth_vals}, dims=("depth",))

    # 8) Prepare time-loop: choose between fast path and per-slice fallback
    has_time_after_interp = ("time" in temp_interp.coords) and int(getattr(temp_interp, "time").size) > 0
    has_time_original = ("time" in ds_temp.coords) and int(getattr(ds_temp, "time").size) > 0

    ohc_list = []
    if has_time_after_interp:
        print("[compute_ohc] using fully-interpolated dataset (fast path).")
        ntime = int(temp_interp.time.size)
        for i in range(ntime):
            slice_da = temp_interp.isel(time=i)
            # apply mask: align dims safely
            mask_for_slice = combined_mask
            if ("time" in combined_mask.dims) and ("time" not in slice_da.dims):
                mask_for_slice = combined_mask.isel(time=0)
            masked = slice_da.where(mask_for_slice == 1)
            if "depth" in masked.dims:
                integrand = masked * dz_da
                integrand = integrand.fillna(0.0)
                vertical_sum = integrand.sum(dim="depth")
            else:
                vertical_sum = masked.fillna(0.0)
            ohc_grid_t = vertical_sum * rho * cp
            ohc_list.append(ohc_grid_t)
            if (i % 5) == 0:
                print(f"  processed step {i+1}/{ntime}")
        ohc_grid = xr.concat(ohc_list, dim="time")
        ohc_grid = ohc_grid.assign_coords(time=temp_interp.time)
    elif has_time_original:
        # fallback: iterate original time axis and interpolate each slice to target grid
        print("[compute_ohc] temp_interp had no time; falling back to per-slice interpolation from original ds_temp.")
        # determine original temperature variable again
        temp_var_name = temp_var
        ntime = int(ds_temp.time.size)
        for i in range(ntime):
            try:
                slice_orig = ds_temp[temp_var_name].isel(time=i)
            except Exception:
                # fallback to first numeric var if indexing fails
                slice_orig = ds_temp[list(ds_temp.data_vars)[0]].isel(time=i)
            # interpolate spatially (and depth) for single slice
            try:
                if interp_kwargs:
                    try:
                        slice_on_grid = slice_orig.interp(method="linear", **interp_kwargs)
                    except Exception:
                        slice_on_grid = slice_orig.interp(method="nearest", **interp_kwargs)
                else:
                    slice_on_grid = slice_orig
            except Exception as e:
                print("  slice interpolation failed, skipping slice", i, "error:", e)
                continue
            mask_for_slice = combined_mask
            if ("time" in combined_mask.dims) and ("time" not in slice_on_grid.dims):
                mask_for_slice = combined_mask.isel(time=0)
            masked = slice_on_grid.where(mask_for_slice == 1)
            if "depth" in masked.dims:
                integrand = masked * dz_da
                integrand = integrand.fillna(0.0)
                vertical_sum = integrand.sum(dim="depth")
            else:
                vertical_sum = masked.fillna(0.0)
            ohc_grid_t = vertical_sum * rho * cp
            ohc_list.append(ohc_grid_t)
            if (i % 5) == 0:
                print(f"  processed orig slice {i+1}/{ntime}")
        if len(ohc_list) == 0:
            raise RuntimeError("No valid time slices processed in fallback path.")
        ohc_grid = xr.concat(ohc_list, dim="time")
        ohc_grid = ohc_grid.assign_coords(time=ds_temp.time[:ohc_grid.sizes["time"]])
    else:
        # no time anywhere: single snapshot
        print("[compute_ohc] no time axis found: treating dataset as single snapshot.")
        if "time" in temp_interp.coords and temp_interp.time.size == 0:
            # empty, but no original time -> use temp_da as a single snapshot
            slice_da = temp_da
        else:
            slice_da = temp_interp if ("time" not in temp_interp.coords) else temp_interp.isel(time=0)
        mask_for_slice = combined_mask
        if ("time" in combined_mask.dims) and ("time" not in slice_da.dims):
            mask_for_slice = combined_mask.isel(time=0)
        masked = slice_da.where(mask_for_slice == 1)
        if "depth" in masked.dims:
            integrand = masked * dz_da
            integrand = integrand.fillna(0.0)
            vertical_sum = integrand.sum(dim="depth")
        else:
            vertical_sum = masked.fillna(0.0)
        ohc_grid = vertical_sum * rho * cp

    # compute spatial (area-weighted if area_da provided) mean time series
    ohc_ts = None
    if "time" in ohc_grid.coords:
        spatial_dims = [d for d in ("lat","lon") if d in ohc_grid.dims]
        if spatial_dims:
            if area_da is not None and set(spatial_dims).issubset(set(area_da.dims)):
                numer = (ohc_grid * area_da).sum(dim=spatial_dims)
                denom = area_da.sum(dim=spatial_dims)
                ohc_ts = numer / denom
            else:
                ohc_ts = ohc_grid.mean(dim=spatial_dims)
        else:
            # e.g., gridded data collapsed to no spatial dims
            ohc_ts = ohc_grid

    return ohc_grid, ohc_ts

# -------------------------
# Save outputs
# -------------------------
def save_outputs(ohc_grid, ohc_ts, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # NetCDF output (ohc_grid)
    out_nc = outdir / "OHC_result.nc"
    try:
        ohc_grid.to_netcdf(out_nc)
        print("[save_outputs] saved:", out_nc)
    except Exception as e:
        print("[save_outputs] failed to save netcdf:", e)

    # plots: first time, last time, timeseries
    try:
        if "time" in ohc_grid.coords:
            ohc_grid.isel(time=0).plot()
            plt.title("OHC - first time step")
            plt.savefig(outdir / "OHC_first.png", dpi=150, bbox_inches="tight")
            plt.close()

            ohc_grid.isel(time=-1).plot()
            plt.title("OHC - last time step")
            plt.savefig(outdir / "OHC_last.png", dpi=150, bbox_inches="tight")
            plt.close()

            if ohc_ts is not None:
                plt.figure(figsize=(10,4))
                # ensure ohc_ts is plottable
                if isinstance(ohc_ts, xr.DataArray):
                    ohc_ts.plot()
                else:
                    plt.plot(ohc_ts)
                plt.title("OHC time series (spatial mean)")
                plt.grid(True)
                plt.savefig(outdir / "OHC_timeseries.png", dpi=150, bbox_inches="tight")
                plt.close()
        else:
            # single snapshot
            ohc_grid.plot()
            plt.title("OHC (single snapshot)")
            plt.savefig(outdir / "OHC_snapshot.png", dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print("[save_outputs] plotting failed:", e)


def main(args):
    ds_mask_woa = open_ds(args.mask_file_woa13)
    ds_mask_local = open_ds(args.mask_file)
    ds_temp = open_ds(args.temperature_file)

    area_da = None
    if args.use_area_from_mask:
        # attempt to build area_da from local mask file if it has delta_lon/delta_lat
        try:
            dl = ds_mask_local.get("delta_lon", None)
            dlat = ds_mask_local.get("delta_lat", None)
            if dl is not None and dlat is not None:
                # expect dl/dlat shape lat/lon (or lon/lat) - try to squeeze to 2D
                dl_vals = dl.values
                dlat_vals = dlat.values
                # attempt transpose or squeeze heuristics
                # final area in m^2
                area_vals = compute_grid_area_from_deltas(dl_vals, dlat_vals)
                # try to create DataArray matching lat/lon names
                lon_name = find_coord(ds_mask_local, ["lon", "longitude"])
                lat_name = find_coord(ds_mask_local, ["lat", "latitude"])
                coords = {}
                if lat_name in ds_mask_local.coords:
                    coords["lat"] = ds_mask_local[lat_name].values
                else:
                    coords["lat"] = latr
                if lon_name in ds_mask_local.coords:
                    coords["lon"] = ds_mask_local[lon_name].values
                else:
                    coords["lon"] = lonr
                # try to align shapes: if area_vals shape matches (lat, lon) use directly;
                # otherwise try transposes until match found
                if area_vals.shape == (len(coords["lat"]), len(coords["lon"])):
                    area_da = xr.DataArray(area_vals, coords=coords, dims=("lat","lon"))
                    print("[main] area_da built from delta arrays.")
                elif area_vals.shape == (len(coords["lon"]), len(coords["lat"])):
                    area_da = xr.DataArray(area_vals.T, coords=coords, dims=("lat","lon"))
                    print("[main] area_da built (transposed) from delta arrays.")
                else:
                    print("[main] area shape mismatch; skipping area_da creation.")
        except Exception as e:
            print("[main] failed to compute area_da from mask:", e)

    ohc_grid, ohc_ts = compute_ohc(ds_mask_woa, ds_mask_local, ds_temp,
                                   lonr=LONR, latr=LATR, depthr=DEPTHR,
                                   rho=RHO, cp=CP,
                                   year_min=args.year_min, year_max=args.year_max, area_da=area_da)

    save_outputs(ohc_grid, ohc_ts, args.outdir)
    print("Done.")


if __name__ == "__main__":
    class Args:
        mask_file_woa13 = "blue-cloud-dataspace/MEI/INGV/INPUT/BATHYMETRY/gebco_2019_mask_1_8_edited_final_woa13.nc"
        mask_file = "blue-cloud-dataspace/MEI/INGV/INPUT/BATHYMETRY/gebco_2019_mask_1_8_edited_final.nc"
        temperature_file = "blue-cloud-dataspace/MEI/INGV/INPUT/CLIMATOLOGY/Temperature_sliding_climatology_WP.nc"
        outdir = "output_ohc"            # <-- must match main() usage (outdir)
        year_min = 1980
        year_max = 2014
        use_area_from_mask = False      # <-- add this so main() can read it
    args = Args()
    main(args)