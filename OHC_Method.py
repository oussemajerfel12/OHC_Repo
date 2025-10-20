import argparse
from pathlib import Path
import xarray as xr 
import numpy as np  
import matplotlib.pyplot as plt

def load_data(mask_file_wo13, mask_file, temperature_file):
    ds_mask_file_wo13 = xr.open_dataset(mask_file_wo13)
    ds_mask = xr.open_dataset(mask_file)
    ds_temp = xr.open_dataset(temperature_file)
    print(ds_temp)

    return ds_mask_file_wo13,ds_mask,ds_temp

if __name__ == "__main__":
    mask_file_woa13 = "blue-cloud-dataspace/MEI/INGV/INPUT/BATHYMETRY/gebco_2019_mask_1_8_edited_final_woa13.nc"
    mask_file = "blue-cloud-dataspace/MEI/INGV/INPUT/BATHYMETRY/gebco_2019_mask_1_8_edited_final.nc"
    temperature_file = 'blue-cloud-dataspace/MEI/INGV/INPUT/CLIMATOLOGY/Temperature_sliding_climatology_WP.nc'

    load_data(mask_file_woa13,mask_file,temperature_file)


