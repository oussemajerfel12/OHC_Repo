import argparse
import os
import subprocess
import shutil
import sys
from pathlib import Path

def run(cmd, **kwargs):
    print("RUN:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Run OHC Notebook inside Blue-Cloud")
    parser.add_argument("--notebook_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--outputs_path", type=str, required=True)
    parser.add_argument("--data_source", type=str, required=False)
    parser.add_argument("--id_output_type", type=str, required=False)
    parser.add_argument("--working_domain", type=str, required=False)
    parser.add_argument("--start_time", type=str, required=False)
    parser.add_argument("--end_time", type=str, required=False)
    parser.add_argument("--central_storage", type=str, required=False)

    args = parser.parse_args()

    outputs_path = Path(args.outputs_path).resolve()
    outputs_path.mkdir(parents=True, exist_ok=True)
    print(f"Outputs directory: {outputs_path}")

    notebook_candidate = Path(args.notebook_path)
    if not notebook_candidate.is_absolute():
        notebook_path = (Path("ohc") / notebook_candidate).resolve()
    else:
        notebook_path = notebook_candidate.resolve()

    print(f"Resolved notebook path: {notebook_path}")
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    notebook_dir = notebook_path.parent

    with (outputs_path / "inputs.txt").open("w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        notebook_path.name,
        "--output", "OHC_executed.ipynb",
        "--ExecutePreprocessor.timeout=600",
        "--output-dir", str(outputs_path)
    ]

    print(f"Executing notebook (cwd = {notebook_dir})")
    run(cmd, cwd=str(notebook_dir))

    for file in notebook_dir.iterdir():
        if file.suffix.lower() in {".png", ".csv", ".nc"}:
            shutil.move(str(file), str(outputs_path / file.name))
            print(f"Moved {file.name} to outputs")

    print("Execution finished. All outputs in:", outputs_path)

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("Subprocess failed:", e)
        raise
