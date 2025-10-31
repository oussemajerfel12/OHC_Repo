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

def main(config_path, base_outdir):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    inputs = config.get("inputs", {})
    if not inputs:
        print("No inputs found in config.yaml")
        return

    for category, items in inputs.items():
        for item in items:
            for key, meta in item.items():
                name = meta["name"]
                url = meta["url"]

                subdir = "CLIMATOLOGY" if category.lower().startswith("clim") else "BATHYMETRY"
                dest_path = os.path.join(base_outdir, "INPUT", subdir, name)

                download_file(url, dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default=os.environ.get("CCP_DATA", "/data"))
    args = parser.parse_args()

    main(args.config, args.outdir)
