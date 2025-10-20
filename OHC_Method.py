import argparse
from pathlib import path
import xarray as xr 
import numpy as np  
import matplotlib.pyplot as plt

def load_data(mask_file_wo13, mask_file, temperature_file):
    ds_mask_file_wo13 = xr.open_dataset(mask_file_wo13)
    ds_mask = xr.open_dataset(mask_file)
    ds_temp = xr.open_dataset(temperature_file)

    return ds_mask_file_wo13,ds_mask,ds_temp