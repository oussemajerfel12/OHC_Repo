import argparse
from pathlib import Path
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os


# ==============================
# Constants
# ==============================
years = np.arange(1960, 2015)

RHO = 1030.0
CP = 3980.0

dx = dy = 0.125 
lonr = np.arange(-5.625, 36.5 + dx, dx)
latr = np.arange(30.0, 46.0 + dy, dy)
depthr = np.array([0., 5., 10., 15., 20., 25., 30., 35., 40., 45.,
                   50., 55., 60., 65., 70., 75., 80., 85., 90., 95.,
                   100., 125., 150., 175., 200., 225., 250., 275., 300., 325.,
                   350., 375., 400., 425., 450., 475., 500., 550., 600., 650.,
                   700., 750., 800., 850., 900., 950., 1000., 1050., 1100., 1150.,
                   1200., 1250., 1300., 1350., 1400., 1450., 1500., 1550., 1600., 1650.,
                   1700., 1750., 1800., 1850., 1900., 1950., 2000.])
deg_multiplier = 111319.5

lon_min = -5.625
lon_max = 36.5
lat_min = 30.0
lat_max = 46.0
depth_min = 0
depth_max = 20000
t_min = 1960
t_max = 2014
experiment = "WP"

# ==============================
# Functions
# ==============================

def get_mask_index(mask_file , lon_min, lon_max, lat_min, lat_max, depth_min, depth_max):
    with Dataset(mask_file, "r") as mask_data:
        lon_mask = mask_data["lon"][:]
        lat_mask = mask_data["lat"][:]
        depth_mask = mask_data["depth"][:]
    start_lon_idx = np.argmin(np.abs(lon_mask - lon_min))
    end_lon_idx   = np.argmin(np.abs(lon_mask - lon_max))
    start_lat_idx = np.argmin(np.abs(lat_mask - lat_min))
    end_lat_idx   = np.argmin(np.abs(lat_mask - lat_max))
    start_depth_idx = np.argmin(np.abs(depth_mask - depth_min))
    end_depth_idx   = np.argmin(np.abs(depth_mask - depth_max))
   
    return start_lon_idx, end_lon_idx, start_lat_idx , end_lat_idx , start_depth_idx, end_depth_idx

def load_temp(temp_file,mask_file):
    with Dataset(temp_file,"r") as temp_data:
        time = temp_data["time"][:]
        climatology_bounds = temp_data["climatology_bounds"][:]
        temperature = temp_data["Temperature"][:]
    temperature = np.transpose(temperature, (3, 2, 1, 0))

    start_lon_mask_index,end_lon_mask_index,start_lat_mask_index,end_lat_mask_index,start_depth_mask_index,end_depth_mask_index = get_mask_index(mask_file,lon_min,lon_max,lat_min,lat_max,depth_min,depth_max)

    temperature = temperature[start_lon_mask_index:end_lon_mask_index + 1,
                           start_lat_mask_index:end_lat_mask_index + 1,
                           start_depth_mask_index:end_depth_mask_index + 1, :]
    
    target_years = np.array([1960, 1970, 1980, 1990, 2000, 2010])
    time_indices = np.array([np.where(years == year)[0][0] if year in years else None for year in target_years])
    temperature_reference = np.expand_dims(np.nanmean(temperature[:, :, :, time_indices], axis=3), axis=3)
    temperature_filtered = np.ma.masked_array(temperature[:, :, :, time_indices], np.isnan(temperature[:, :, :, time_indices]))

    return temperature,temperature_reference,temperature_filtered,climatology_bounds






def load_mask(mask_file, lon_min, lon_max, lat_min, lat_max, depth_min, depth_max):
    with Dataset(mask_file, "r") as mask_data:
        lon_mask = mask_data["lon"][:]
        lat_mask = mask_data["lat"][:]
        depth_mask = mask_data["depth"][:]
        mask = mask_data["mask"][:].astype(bool)
        delta_lon = mask_data["delta_lon"][:]
        delta_lat = mask_data["delta_lat"][:]
        delta_depth = mask_data["delta_depth"][:]

    mask = np.transpose(mask, (2, 1, 0))
    delta_lon = np.transpose(delta_lon, (2, 1, 0))
    delta_lat = np.transpose(delta_lat, (2, 1, 0))
    delta_depth = np.transpose(delta_depth, (2, 1, 0))



    start_lon_idx = np.argmin(np.abs(lon_mask - lon_min))
    end_lon_idx   = np.argmin(np.abs(lon_mask - lon_max))


    start_lat_idx = np.argmin(np.abs(lat_mask - lat_min))
    end_lat_idx   = np.argmin(np.abs(lat_mask - lat_max))


    start_depth_idx = np.argmin(np.abs(depth_mask - depth_min))
    end_depth_idx   = np.argmin(np.abs(depth_mask - depth_max))

    lon_mask   = lon_mask[start_lon_idx:end_lon_idx+1]
    lat_mask   = lat_mask[start_lat_idx:end_lat_idx+1]
    depth_mask = depth_mask[start_depth_idx:end_depth_idx+1]

    mask = mask[start_lon_idx:end_lon_idx+1,
                start_lat_idx:end_lat_idx+1,
                start_depth_idx:end_depth_idx+1]

    delta_lon = delta_lon[start_lon_idx:end_lon_idx+1,
                          start_lat_idx:end_lat_idx+1,
                          start_depth_idx:end_depth_idx+1]

    delta_lat = delta_lat[start_lon_idx:end_lon_idx+1,
                          start_lat_idx:end_lat_idx+1,
                          start_depth_idx:end_depth_idx+1]

    delta_depth = delta_depth[start_lon_idx:end_lon_idx+1,
                              start_lat_idx:end_lat_idx+1,
                              start_depth_idx:end_depth_idx+1]

    print(f"Subset mask -> lon: {lon_min}-{lon_max}, lat: {lat_min}-{lat_max}, depth: {depth_min}-{depth_max}")
    print(f"Shapes -> mask: {mask.shape}, delta_lon: {delta_lon.shape}")

    return lon_mask, lat_mask, depth_mask, mask, delta_lon, delta_lat, delta_depth


def output_suffix(lon_min, lon_max, lat_min, lat_max, depth_min, depth_max):
    suffix = (
        f"{int(lon_min)}_{int(lon_max)}_"
        f"{int(lat_min)}_{int(lat_max)}_"
        f"{int(depth_min)}_{int(depth_max)}"
    )
    return suffix

def plot_mask_levels(lon_mask, lat_mask, mask, depth_mask, 
                     levels=None, cmap='jet', save_dir=None, show=True):
   
    if levels is None:
        levels = [0]
    elif isinstance(levels, int):
        levels = [levels]

    for level in levels:
        plt.figure(figsize=(6, 4))
        ax = plt.gca()

        ax.set_aspect(1 / np.cos(np.mean(lat_mask) * np.pi / 180))

        ax.tick_params(axis='both', labelsize=6)

        plt.pcolor(lon_mask, lat_mask, mask[:, :, level].astype(float).T,
                   cmap=cmap, shading='nearest')

        plt.title(f"Water mask at depth {depth_mask[level]} m")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/mask_level_{int(depth_mask[level])}.png", 
                        dpi=150, bbox_inches='tight')
            print(f"[plot_mask_levels] Saved: {save_dir}/mask_level_{int(depth_mask[level])}.png")

        if show:
            plt.show()
        else:
            plt.close()
    

def avg_temperature_anomaly_plt(temperature, temperature_reference, output_path, suffix):
    temperature_reference_4d = np.repeat(temperature_reference[:, :, :], temperature.shape[3], axis=3)
    temperature_diff = temperature - temperature_reference_4d
    temperature_profile = np.nanmean(temperature_diff, axis=(0, 1))

    plt.figure(figsize=(16, 6))
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=16)

    vmin, vmax = -1.0, 1.0
    contour_levels = np.linspace(vmin, vmax, 21)

    pcolor = ax.pcolor(np.arange(temperature_profile.shape[1]), -depthr, temperature_profile,
                       vmin=vmin, vmax=vmax, cmap='bwr', shading='auto')

    ax.contour(np.arange(temperature_profile.shape[1]), -depthr, temperature_profile,
               levels=contour_levels, colors='k', linewidths=0.4)

    cbar = plt.colorbar(pcolor, orientation="vertical", shrink=0.8)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('$\\circ{C}$', fontsize=20, labelpad=10)


    plt.title("Temperature Anomaly", fontsize=24)
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/TEMP_ANOMALY_{suffix}.png", bbox_inches='tight', dpi=200)
    #plt.show()






def plot_temperature_profile(temperature, depthr, years, output_path, suffix, vmin=13, vmax=16):
    temperature_profile = np.nanmean(temperature, axis=(0, 1))  

    plt.figure(figsize=(16, 6))
    ax = plt.gca()
    ax.tick_params("both", labelsize=16)

    contour_levels = np.arange(vmin, vmax + 0.1, 0.1)
    ax.contour(years, -depthr, temperature_profile, levels=contour_levels, colors="k", linewidths=0.4)
    pcolor = ax.pcolor(years, -depthr, temperature_profile, vmin=vmin, vmax=vmax, cmap="jet")
    
    cbar = plt.colorbar(pcolor, orientation="vertical", shrink=0.8)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('$\circ{C}$\n', fontsize=20, labelpad=-20, loc='top')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.label.set_rotation(0)

    plt.title("Temperature", fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_path}/TEMP_PROFILE_{suffix}.png", bbox_inches='tight', dpi=200)
    #plt.show()

def compute_temperature_anomaly_and_ohc(temperature, temperature_reference, mask,delta_lon, delta_lat, delta_depth,depthr, reference_density,specific_heat_capacity,bottom_depth):
    bottom_depth_index = np.argmin(np.abs(depthr - bottom_depth))
    
    temperature_reference_4d = np.repeat(temperature_reference, temperature.shape[3], axis=3)
    
    temperature_anomaly = temperature - temperature_reference_4d
    
    delta_depth_slice = delta_depth[:, :, :bottom_depth_index+1]
    mask_slice = mask[:, :, :bottom_depth_index+1]
    
    delta_depth_4d = delta_depth_slice[:, :, :, np.newaxis]
    mask_4d = mask_slice[:, :, :, np.newaxis]
    
    temp_anom_integrand = temperature_anomaly[:, :, :bottom_depth_index+1, :] * delta_depth_4d
    temp_anom_integrand = temp_anom_integrand * mask_4d
    temp_anom_integrand = np.nan_to_num(temp_anom_integrand, nan=0.0)
    
    vertical_temperature_anomaly_sum = np.sum(temp_anom_integrand, axis=2)
    
    temperature_anomaly_profile = np.nanmean(temperature_anomaly[:, :, :bottom_depth_index+1, :], axis=(0,1,2))
    
    horizontal_area = delta_lon[:, :, 0] * delta_lat[:, :, 0] * mask[:, :, 0]
    horizontal_area_total = np.nansum(horizontal_area)
    
    ocean_heat_content_profile = np.empty(temperature.shape[3], dtype=np.float64)
    for t in range(temperature.shape[3]):
        ocean_heat_content_profile[t] = (
            reference_density * specific_heat_capacity *
            np.nansum(vertical_temperature_anomaly_sum[:, :, t] * horizontal_area) /
            horizontal_area_total
        )
    
    return temperature_anomaly_profile, ocean_heat_content_profile, bottom_depth_index


def plot_temperature_anomaly_trend(years, temperature_anomaly_profile, bottom_depth,output_path,suffix,start_year=1993):
    plt.figure(figsize=(12, 4))
    plt.grid(True)
    
    plt.plot(years, temperature_anomaly_profile, "m-", label="VRE")
    
    year_index = np.argwhere(years == start_year)[0][0]
    
    fit_coeff = np.polyfit(years[year_index:], temperature_anomaly_profile[year_index:], 1)
    fit_function = np.poly1d(fit_coeff)
    
    profile_annual_trend = np.round(fit_function(years[-1]) - fit_function(start_year), 3)
    
    plt.plot(years[year_index:], fit_function(years[year_index:]), "r-", 
             label=f"VRE trend from {start_year}: {profile_annual_trend:.3f} °C")
    
    plt.xlabel("Years", fontsize=16)
    plt.ylabel("Temp. anomaly (°C)", fontsize=16)
    plt.title(f"Temperature anomaly 0-{bottom_depth} m", fontsize=20)
    plt.legend(fontsize=14)
    
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    plt.savefig(f"{output_path}/TEMP_ANOMALY_TREND_{suffix}.png", bbox_inches='tight', dpi=200)
    #plt.show()

def plot_ohc_anomaly(years, ocean_heat_content_profile, bottom_depth,output_path,suffix, start_year=1993):
    plt.figure(figsize=(12, 4))
    plt.grid(True)

    plt.plot(years, ocean_heat_content_profile, "m-", label="VRE")

    year_index = np.where(years == start_year)[0][0]

    fit_coeff = np.polyfit(years[year_index:], ocean_heat_content_profile[year_index:], 1)
    fit_function = np.poly1d(fit_coeff)

    profile_annual_trend = np.round((fit_function(years[-1]) - fit_function(start_year)) / (86400. * 365), 1)

    plt.plot(years[year_index:], fit_function(years[year_index:]), "r-", 
             label=f"VRE trend from {start_year}: {profile_annual_trend:.1f} ± 0.1 W m⁻²")

    plt.xlabel("Years", fontsize=16)
    plt.ylabel("OHC (J m⁻²)", fontsize=16)
    plt.title(f"Ocean Heat Content anomaly 0-{bottom_depth} m", fontsize=20)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.savefig(f"{output_path}/OHC_700_ANOMALY_{suffix}.png", bbox_inches='tight', dpi=200)
    #plt.show()

    
def save_ohc_temperature_nc(output_path,climatology_bounds,temperature_anomaly_profile, suffix, years, ocean_heat_content_profile):
    
    out_file = os.path.join(output_path, f'Ocean_Heat_Content_WP4_{suffix}.nc')
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    if os.path.isfile(out_file):
        try:
            os.remove(out_file)
            print(f"Removed existing file: {out_file}")
        except OSError as e:
            print(f"Error removing existing file: {e}")
    
    print(f"Creating new NetCDF file: {out_file}")
    out_data = Dataset(out_file, "w", format="NETCDF4")

    out_data.setncatts({"Conventions": "CF-1.6"})

    depth_dim = out_data.createDimension("depth", 1)
    time_dim = out_data.createDimension("time", len(years))
    nv_dim = out_data.createDimension("nv", climatology_bounds.shape[0])

    nctime = out_data.createVariable("time", "f8", ("time",))
    nctime.units = "days since 1900-01-01 00:00:00"
    nctime.standard_name = "time"
    nctime.long_name = "time"
    nctime.calendar = "standard"
    nctime.climatology = "climatology_bounds"

    ncdepth_bounds = out_data.createVariable("depth_bounds", "f8", ("nv", "depth"))
    ncdepth_bounds.units = "m"

    ncclimatology_bounds = out_data.createVariable("climatology_bounds", "f8", ("nv", "time"))
    ncclimatology_bounds.units = "days since 1900-01-01 00:00:00"

    ncVariable1 = out_data.createVariable("Ocean_Heat_Content", "f4", ("time", "depth"),
                                          fill_value=9.96921e36, zlib=True)
    ncVariable1.units = "J"
    ncVariable1.standard_name = "ocean_heat_content"
    ncVariable1.long_name = "Ocean Heat Content Anomaly"
    ncVariable1.cell_methods = "time: mean within years time: mean over years"

    ncVariable2 = out_data.createVariable("Temperature_anomaly", "f4", ("time", "depth"),
                                          fill_value=9.96921e36, zlib=True)
    ncVariable2.units = "degrees_C"
    ncVariable2.standard_name = "sea_water_temperature"
    ncVariable2.long_name = "Sea Water In Situ Temperature Anomaly"
    ncVariable2.cell_methods = "time: mean within years time: mean over years"

    ncVariable1[:, 0] = ocean_heat_content_profile
    ncVariable2[:, 0] = temperature_anomaly_profile
    nctime[:] = years

    out_data.close()
    print(f"NetCDF file saved: {out_file}")
    
def main():
    parser = argparse.ArgumentParser(description="Compute Ocean Heat Content and Temperature Anomaly.")
    parser.add_argument("--mask_file", type=str, required=True)
    parser.add_argument("--temperature_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="OceanHeatContent")
    
    args = parser.parse_args()

    mask_file = args.mask_file
    temperature_file = args.temperature_file
    outdir = args.outdir
    years = np.arange(1960, 2015)
    os.makedirs(outdir, exist_ok=True)


    suffix = output_suffix(lon_min, lon_max, lat_min, lat_max, depth_min, depth_max)

    lon_mask, lat_mask, depth_mask, mask, delta_lon, delta_lat, delta_depth = load_mask(
        mask_file, lon_min, lon_max, lat_min, lat_max, depth_min, depth_max
    )

    temperature, temperature_reference, temperature_filtered, climatology_bounds = load_temp(temperature_file, mask_file)
    year_indices = np.where((years >= t_min) & (years <= t_max))[0]
    temperature = temperature[:, :, :, year_indices]

    selected_years = years[year_indices]


    plot_temperature_profile(temperature, depthr, selected_years, output_path=outdir, suffix=suffix)

    avg_temperature_anomaly_plt (temperature,temperature_reference,output_path=outdir,suffix=suffix)

    temperature_anomaly_profile, ocean_heat_content_profile, bottom_depth_index = compute_temperature_anomaly_and_ohc(temperature,temperature_reference,mask,delta_lon,delta_lat,delta_depth,depthr,reference_density=RHO,specific_heat_capacity=CP,bottom_depth=depth_max)

    plot_temperature_anomaly_trend(selected_years, temperature_anomaly_profile, depthr[bottom_depth_index],start_year=1993, output_path=outdir, suffix=suffix)

    plot_ohc_anomaly(selected_years,ocean_heat_content_profile,bottom_depth=depthr[bottom_depth_index],start_year=1993,output_path=outdir,suffix=suffix)

    save_ohc_temperature_nc(outdir,climatology_bounds,temperature_anomaly_profile,suffix,selected_years,ocean_heat_content_profile)


    print(f"Plots saved in: {outdir}")

# ==============================
# Run main
# ==============================
if __name__ == "__main__":
    main()
