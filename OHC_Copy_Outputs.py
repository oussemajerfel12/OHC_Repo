import argparse
from datetime import datetime
import os
import shutil

# ----------------------------
# Parse input arguments
# ----------------------------
parser = argparse.ArgumentParser(description="OHC Copy Outputs Mock Method")

parser.add_argument("--repository", type=str, required=False, default="", help="Repository path")
parser.add_argument("--data_path", type=str, required=False, help="Path to input data")
parser.add_argument("--outputs_path", type=str, required=False, help="Path to output data")
parser.add_argument("--data_source", type=str, required=True, help="Dataset ID")
parser.add_argument("--id_output_type", type=str, required=True, help="Output type (e.g., ohc_timeseries)")
parser.add_argument("--working_domain", type=str, required=True, help="Working domain")
parser.add_argument("--start_time", type=str, required=True, help="Start date (YYYY-MM-DD)")
parser.add_argument("--end_time", type=str, required=False, help="End date (YYYY-MM-DD)")

args = parser.parse_args()

# ----------------------------
# Prepare variables and paths
# ----------------------------
repository = args.repository
data_path = args.data_path
outputs_path = args.outputs_path
data_source = args.data_source
id_output_type = args.id_output_type
working_domain = args.working_domain
start_time = datetime.strptime(args.start_time, "%Y-%m-%d")
end_time = datetime.strptime(args.end_time,"%Y-%m-%d")

# Make sure outputs_path exists
os.makedirs(outputs_path, exist_ok=True)

# ----------------------------
# Save all inputs to a text file
# ----------------------------
output_file = os.path.join(outputs_path, "inputs.txt")

with open(output_file, "w") as f:
    f.write("=== OHC Mock-Up Method Inputs ===\n")
    f.write(f"repository: {repository}\n")
    f.write(f"data_path: {data_path}\n")
    f.write(f"outputs_path: {outputs_path}\n")
    f.write(f"data_source: {data_source}\n")
    f.write(f"id_output_type: {id_output_type}\n")
    f.write(f"working_domain: {working_domain}\n")
    f.write(f"start_time: {start_time.strftime('%Y-%m-%d')}\n")
    f.write(f"end_time: {end_time.strftime('%Y-%m-%d')}\n")

print(f" Parameters saved to: {output_file}")

# ----------------------------
# Copy to /ccp_data for Blue-Cloud visibility
# ----------------------------
ccp_target = "/ccp_data"

if os.path.exists(ccp_target):
    try:
        shutil.copy(output_file, os.path.join(ccp_target, "inputs.txt"))
        print(f"Copied to CCP data: {ccp_target}/inputs.txt")
    except Exception as e:
        print(f" Could not copy to {ccp_target}: {e}")
else:
    print(f"CCP data folder not found at {ccp_target}")

print("Method finished successfully.")
