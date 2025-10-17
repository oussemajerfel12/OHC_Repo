import argparse
import os 
import subprocess 
import shutil
import sys
from pathlib import Path

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

    
    notebook_path = Path(args.notebook_path).resolve()
    outputs_path = Path(args.outputs_path).resolve()
    
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    with (outputs_path / "inputs.txt").open("w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute", str(notebook_path),
        "--output", str(outputs_path / "OHC_executed.ipynb"),
        "--ExecutePreprocessor.timeout=600"
    ]

    print("Executing notebook...")
    subprocess.run(cmd, check=True)

    for file in os.listdir("."):
        if file.endswith((".png", ".csv", ".nc")):
            shutil.move(file,outputs_path / file)

if __name__ == "__main__":
    main()           



