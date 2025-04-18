"""
This script searches for calcium imaging data in the Dandiset by looking for
files containing specific keywords in their paths.
"""

import subprocess
import json

# Execute the command to get assets
try:
    result = subprocess.run(
        ["python", "../tools_cli.py", "dandiset-assets", "001195"],
        capture_output=True,
        text=True,
        check=True
    )
    
    # Parse the JSON output
    data = json.loads(result.stdout)
    
    # Search for files with specific keywords
    imaging_files = []
    for asset in data["results"]["results"]:
        path = asset["path"]
        if "imaging" in path.lower() or "calcium" in path.lower() or "onephoton" in path.lower():
            imaging_files.append({
                "asset_id": asset["asset_id"],
                "path": path,
                "size": asset["size"]
            })
    
    if imaging_files:
        print(f"Found {len(imaging_files)} files related to imaging:")
        for file in imaging_files:
            print(f"- {file['path']} (ID: {file['asset_id']})")
    else:
        print("No imaging files found in the current page.")
        
    # Print the total count of assets
    print(f"\nTotal assets in the dataset: {data['results']['count']}")
    
except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e}")
    print(f"stderr: {e.stderr}")