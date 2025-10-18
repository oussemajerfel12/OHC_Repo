import argparse
import os
import shutil
from pathlib import Path
import subprocess

def run(cmd, **kwargs):
    print("RUN:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Copy OHC outputs to CCP data folder")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--outputs_path", required=True)
    parser.add_argument("--data_source", required=True)
    parser.add_argument("--id_output_type", required=True)
    parser.add_argument("--working_domain", required=True)
    parser.add_argument("--start_time", required=True)
    parser.add_argument("--end_time", required=True)
    args = parser.parse_args()

    repo_name = "ohc"
    if not Path(repo_name).exists():
        run(["git", "clone", args.repository, repo_name])

    # Save parameters to a text file for traceability
    inputs_file = Path(args.outputs_path) / "inputs.txt"
    inputs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(inputs_file, "w") as f:
        f.write(f"repository: {args.repository}\n")
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"outputs_path: {args.outputs_path}\n")
        f.write(f"data_source: {args.data_source}\n")
        f.write(f"id_output_type: {args.id_output_type}\n")
        f.write(f"working_domain: {args.working_domain}\n")
        f.write(f"start_time: {args.start_time}\n")
        f.write(f"end_time: {args.end_time}\n")

    print(f"Parameters saved to: {inputs_file}")

    # Copy results from the real output path to CCP data
    src = Path(args.outputs_path)
    dst = Path("/ccp_data")
    dst.mkdir(parents=True, exist_ok=True)

    if src.exists():
        for file in src.glob("*"):
            shutil.copy(file, dst)
        print(f"Copied to CCP data: {dst}")
    else:
        print(f"Warning: Output folder {src} not found!")

    print("Method finished successfully.")

if __name__ == "__main__":
    main()
