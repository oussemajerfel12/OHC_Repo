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

    parser.add_argument("--repository", type=str, required=True)
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

    # Prepare outputs directory
    outputs_path = Path(args.outputs_path).resolve()
    outputs_path.mkdir(parents=True, exist_ok=True)
    print(f" Outputs directory: {outputs_path}")

    # Clone repository into ./ohc (if not already exists)
    repo_dir = Path("ohc")
    if repo_dir.exists():
        print(f"Repository folder '{repo_dir}' already exists — skipping clone.")
    else:
        print(f"Cloning {args.repository} into {repo_dir} ...")
        run(["git", "clone", args.repository, str(repo_dir)])
    repo_dir = repo_dir.resolve()

    # Resolve notebook path (absolute or relative to repo)
    notebook_candidate = Path(args.notebook_path)
    if not notebook_candidate.is_absolute():
        notebook_path = (repo_dir / notebook_candidate).resolve()
    else:
        notebook_path = notebook_candidate.resolve()

    print(f" Resolved notebook path: {notebook_path}")
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    notebook_dir = notebook_path.parent

    # Save run parameters to a text file
    with (outputs_path / "inputs.txt").open("w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # Execute notebook with nbconvert
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        notebook_path.name,
        "--output", "OHC_executed.ipynb",
        "--ExecutePreprocessor.timeout=600",
        "--output-dir", str(outputs_path)
    ]

    print(f" Executing notebook (cwd = {notebook_dir})")
    run(cmd, cwd=str(notebook_dir))

    moved = []
    for file in notebook_dir.iterdir():
        if file.suffix.lower() in {".png", ".csv", ".nc"}:
            target = outputs_path / file.name
            shutil.move(str(file), str(target))
            moved.append(file.name)

    print(f" Execution finished. Moved files: {moved}")
    print(f" All outputs are in: {outputs_path}")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(" Subprocess failed:", e)
        raise
