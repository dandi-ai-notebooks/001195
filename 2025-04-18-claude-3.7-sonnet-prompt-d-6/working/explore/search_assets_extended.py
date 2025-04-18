"""
This script searches for all types of files in the Dandiset with a more extensive
set of keywords that might indicate calcium imaging data.
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
    
    # Get all unique file extensions
    extensions = set()
    for asset in data["results"]["results"]:
        path = asset["path"]
        if "." in path:
            extensions.add(path.split(".")[-1])
    
    print("All file extensions in the dataset:")
    for ext in sorted(extensions):
        print(f"- .{ext}")
    
    # Group files by subject
    subjects = {}
    for asset in data["results"]["results"]:
        path = asset["path"]
        parts = path.split("/")
        if len(parts) > 0:
            subject = parts[0]
            if subject not in subjects:
                subjects[subject] = []
            subjects[subject].append({
                "asset_id": asset["asset_id"],
                "path": path,
                "size": asset["size"]
            })
    
    print("\nFiles grouped by subject:")
    for subject, files in subjects.items():
        print(f"\n{subject} ({len(files)} files):")
        for i, file in enumerate(files[:5]):  # Show only first 5 files per subject
            print(f"- {file['path']}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more files")
    
    # Look for any file that seems different from the icephys pattern
    print("\nFiles not matching typical icephys pattern:")
    for asset in data["results"]["results"]:
        path = asset["path"]
        if not path.endswith("_icephys.nwb"):
            print(f"- {path} (ID: {asset['asset_id']})")
    
except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e}")
    print(f"stderr: {e.stderr}")