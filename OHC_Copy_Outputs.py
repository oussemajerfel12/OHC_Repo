import argparse
from datetime import datetime
from pathlib import Path
import os
import shutil
import json


parser = argparse.ArgumentParser(description="Mock OHC Method Execution")

parser.add_argument("--repository", type=str, required=False, default="", help="Repository path")
parser.add_argument("--data_path", type=str, required=False, help="Path to input data")
parser.add_argument("--outputs_path", type=str, required=False, help="Path to output data")
parser.add_argument("--data_source", type=str, required=True, help="Dataset ID")
parser.add_argument("--id_output_type", type=str, required=True, help="Output type")
parser.add_argument("--working_domain", type=str, required=True, help="Working domain in JSON format")
parser.add_argument("--start_time", type=str, required=True, help="Start date (YYYY-MM-DD)")
parser.add_argument("--end_time", type=str, required=False, help="End date (YYYY-MM-DD)")

args = parser.parse_args()


repository = args.repository
data_path = args.data_path or "/data"
outputs_path = args.outputs_path or "/workspace/MEI/OceanHeatContent"
data_source = args.data_source
id_output_type = args.id_output_type
working_domain = args.working_domain
start_time = datetime.strptime(args.start_time, "%Y-%m-%d")
end_time = datetime.strptime(args.end_time, "%Y-%m-%d") if args.end_time else None


Path(outputs_path).mkdir(parents=True, exist_ok=True)


inputs_file = Path(outputs_path) / "inputs.txt"
with open(inputs_file, "w") as f:
    f.write("=== OHC Copy Outputs - Execution Parameters ===\n\n")
    f.write(f"Repository: {repository}\n")
    f.write(f"Data path: {data_path}\n")
    f.write(f"Outputs path: {outputs_path}\n")
    f.write(f"Data source: {data_source}\n")
    f.write(f"Output type: {id_output_type}\n")
    f.write(f"Working domain: {working_domain}\n")
    f.write(f"Start time: {start_time.strftime('%Y-%m-%d')}\n")
    if end_time:
        f.write(f"End time: {end_time.strftime('%Y-%m-%d')}\n")
print(f" Parameters saved to: {inputs_file}")


ccp_data_path = Path("/ccp_data")
ccp_data_path.mkdir(parents=True, exist_ok=True)


copied_files = []
for file in Path(outputs_path).glob("*"):
    if file.is_file():
        shutil.copy(file, ccp_data_path)
        copied_files.append(file.name)

if copied_files:
    print(f" Copied {len(copied_files)} files to {ccp_data_path}")
else:
    print(f" No files found in {outputs_path} to copy.")


print("Method finished successfully.")
