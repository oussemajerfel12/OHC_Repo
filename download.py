import os 
import requests

ccp_data = os.environ.get('CCP_DATA','')

dirs = [
    os.path.join(ccp_data,"INPUT","BATHYMETRY"),
    os.path.join(ccp_data,"INPUT","CLIMATOLOGY"),
]

for dir in dirs :
    os.makedirs(dir,exist_ok=True)
    print(f"Created Directory : {dir}")

files_to_download = {
    "CLIMATOLOGY/Temperature_sliding_climatology_WP.nc": "https://data.d4science.net/7LLjQ",
    "BATHYMETRY/gebco_2019_mask_1_8_edited_final.nc": "https://data.d4science.net/NVr2L",
}

for relative_path, url in files_to_download.items():
    file_path = os.path.join(ccp_data, "INPUT", relative_path)
    print(f"Downloading {url} to {file_path} ...")
    response = requests.get(url, stream=True,timeout=60)
    response.raise_for_status()  
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {file_path}")