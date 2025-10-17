import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys

def safe_copy(src_dir, dest_dir):
    src = Path(src_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"[WARN] Source path {src} does not exist, skipping.")
        return

    for item in src.iterdir():
        target = dest / item.name
        try:
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
            print(f"[INFO] Copied {item} → {target}")
        except Exception as e:
            print(f"[ERROR] Failed to copy {item}: {e}")

def main():
    parser = argparse.ArgumentParser(description="OHC output collection and copy")
    parser.add_argument("--repository", type=str, required=False, default="", help="Repository path")
    parser.add_argument("--data_path", type=str, required=False, help="Path to input data")
    parser.add_argument("--outputs_path", type=str, required=False, help="Path to output data")
    parser.add_argument("--data_source", type=str, required=True, help="Dataset ID")
    parser.add_argument("--id_output_type", type=str, required=True, help="Output type (e.g., ohc_timeseries)")
    parser.add_argument("--working_domain", type=str, required=True, help="Working domain")
    parser.add_argument("--start_time", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_time", type=str, required=False, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    start_time = datetime.strptime(args.start_time, "%Y-%m-%d")
    end_time = datetime.strptime(args.end_time, "%Y-%m-%d") if args.end_time else None

    outputs_dir = Path(args.outputs_path)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    inputs_file = outputs_dir / "inputs.txt"
    with open(inputs_file, "w") as f:
        f.write(f"repository: {args.repository}\n")
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"outputs_path: {outputs_dir}\n")
        f.write(f"data_source: {args.data_source}\n")
        f.write(f"id_output_type: {args.id_output_type}\n")
        f.write(f"working_domain: {args.working_domain}\n")
        f.write(f"start_time: {start_time.strftime('%Y-%m-%d')}\n")
        if end_time:
            f.write(f"end_time: {end_time.strftime('%Y-%m-%d')}\n")

    print(f"[INFO] Saved input parameters to {inputs_file}")

    source_path = Path(outputs_dir)
    copy_targets = [Path("/outputs"), Path("/ccp_data")]

    for target in copy_targets:
        print(f"[INFO] Copying files from {source_path} to {target}")
        safe_copy(source_path, target)

    print("[INFO] Copy completed successfully.")

if __name__ == "__main__":
    main()
