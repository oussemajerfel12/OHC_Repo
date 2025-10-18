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
        print(f"[WARN] Source path {src} does not exist, skipping copy.")
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
    parser = argparse.ArgumentParser(description="Collect and copy OHC analysis outputs.")
    parser.add_argument("--repository", type=str, required=True, help="Git repository used (for reference).")
    parser.add_argument("--data_path", type=str, required=True, help="Input data path.")
    parser.add_argument("--outputs_path", type=str, required=True, help="Path where notebook outputs are generated.")
    parser.add_argument("--data_source", type=str, required=True, help="Dataset ID (e.g. SDC_MED_DP2).")
    parser.add_argument("--id_output_type", type=str, required=True, help="Output type (e.g. OHC_plot).")
    parser.add_argument("--working_domain", type=str, required=True, help="Spatial/temporal domain info.")
    parser.add_argument("--start_time", type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_time", type=str, required=True, help="End date (YYYY-MM-DD).")

    args = parser.parse_args()

    try:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%d")
        end_time = datetime.strptime(args.end_time, "%Y-%m-%d")
    except ValueError:
        print("[ERROR] Invalid date format. Use YYYY-MM-DD.")
        sys.exit(1)

    outputs_dir = Path(args.outputs_path)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    inputs_file = outputs_dir / "run_parameters.txt"
    with open(inputs_file, "w") as f:
        f.write(f"Repository: {args.repository}\n")
        f.write(f"Data path: {args.data_path}\n")
        f.write(f"Outputs path: {outputs_dir}\n")
        f.write(f"Data source: {args.data_source}\n")
        f.write(f"Output type: {args.id_output_type}\n")
        f.write(f"Working domain: {args.working_domain}\n")
        f.write(f"Start time: {start_time.strftime('%Y-%m-%d')}\n")
        f.write(f"End time: {end_time.strftime('%Y-%m-%d')}\n")

    print(f"[INFO] Saved run parameters to {inputs_file}")


    copy_targets = [
        Path("/outputs"),               
        Path("/workspace/CCP/executions")  
    ]

    for target in copy_targets:
        print(f"[INFO] Copying results from {outputs_dir} to {target}")
        safe_copy(outputs_dir, target)

    print("[INFO] Copy completed successfully!")


if __name__ == "__main__":
    main()
