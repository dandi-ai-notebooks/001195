"""
This script attempts to retrieve all assets from the Dandiset by paginating through
all available results and searches for various types of data.
"""

import subprocess
import json
import re

def get_assets(page=1, page_size=20):
    """Retrieves a page of assets from the Dandiset."""
    try:
        # This is a simplistic approach - the actual API might require a different pagination method
        cmd = ["python", "../tools_cli.py", "dandiset-assets", "001195"]
        if page > 1:
            # Add pagination parameters if supported by the API
            cmd.extend([f"--page={page}", f"--page-size={page_size}"])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"stderr: {e.stderr}")
        return None

# Get first page of assets
data = get_assets(page=1)
if not data:
    print("Failed to retrieve assets")
    exit(1)

total_count = data["results"]["count"]
assets = data["results"]["results"]
print(f"Retrieved first page with {len(assets)} of {total_count} total assets")

# Collect all unique keywords from file paths
keywords = set()
for asset in assets:
    path = asset["path"]
    # Extract potential keywords from the path
    words = re.findall(r'[a-zA-Z]+', path)
    for word in words:
        if len(word) > 3:  # Skip very short words
            keywords.add(word.lower())

print(f"\nFound {len(keywords)} unique keywords in file paths:")
print(sorted(keywords))

# Filter assets by file type (finding non-icephys files)
imaging_candidates = []
icephys_files = []
other_files = []

for asset in assets:
    path = asset["path"]
    if "_icephys.nwb" in path:
        icephys_files.append(asset)
    elif "imaging" in path.lower() or "calcium" in path.lower() or "photo" in path.lower():
        imaging_candidates.append(asset)
    else:
        other_files.append(asset)

print(f"\nFile categorization from first page:")
print(f"- Intracellular electrophysiology files: {len(icephys_files)}")
print(f"- Possible imaging files: {len(imaging_candidates)}")
print(f"- Other files: {len(other_files)}")

if imaging_candidates:
    print("\nPossible imaging files:")
    for file in imaging_candidates:
        print(f"- {file['path']} (ID: {file['asset_id']})")

if len(other_files) > 0:
    print("\nOther files (might include imaging):")
    for file in other_files:
        print(f"- {file['path']} (ID: {file['asset_id']})")

# Try to access other pages if available
if total_count > len(assets):
    print(f"\nAttempting to retrieve additional pages...")
    for page in range(2, (total_count // 20) + 2):  # Calculate number of pages needed
        data = get_assets(page=page)
        if not data or not data["results"]["results"]:
            print(f"No more results found or page {page} is empty")
            break
            
        new_assets = data["results"]["results"]
        print(f"Retrieved page {page} with {len(new_assets)} assets")
        
        # Check for imaging files in this page
        for asset in new_assets:
            path = asset["path"]
            if "_icephys.nwb" not in path:
                print(f"Found potential non-icephys file: {path} (ID: {asset['asset_id']})")
            elif "imaging" in path.lower() or "calcium" in path.lower() or "photo" in path.lower():
                print(f"Found potential imaging file: {path} (ID: {asset['asset_id']})")