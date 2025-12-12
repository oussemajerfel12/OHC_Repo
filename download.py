import os
import sys
import yaml
import requests
import argparse

def download_file(url, dest_path):
    try:
        print(f"Downloading {url} -> {dest_path}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[SUCCESS] Downloaded: {dest_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        raise

def main(config_path, base_outdir, data_source):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    inputs = config.get("inputs", {})
    clim_list = inputs.get("clim", [])
    bathy_list = inputs.get("bathy", [])

    clim_dict = {list(item.keys())[0]: list(item.values())[0] for item in clim_list}
    bathy_dict = {list(item.keys())[0]: list(item.values())[0] for item in bathy_list}

    for bathy_id, meta in bathy_dict.items():
        name, url = meta["name"], meta["url"]
        dest_path = os.path.join(base_outdir, "INPUT", "BATHYMETRY", name)
        download_file(url, dest_path)

    if data_source:
        for ds_id in data_source:
            if ds_id not in clim_dict:
                raise ValueError(f"[ERROR] '{ds_id}' not found in config under 'clim'")
            meta = clim_dict[ds_id]
            name, url = meta["name"], meta["url"]
            dest_path = os.path.join(base_outdir, "INPUT", "CLIMATOLOGY", name)
            download_file(url, dest_path)
    else:
        print("No climatology dataset IDs provided (data_source empty).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default=os.environ.get("CCP_DATA", "/data"))
    parser.add_argument("--data_source", type=str)
    args = parser.parse_args()

    import json
    data_source = json.loads(args.data_source) if args.data_source else []
    main(args.config, args.outdir, data_source)
