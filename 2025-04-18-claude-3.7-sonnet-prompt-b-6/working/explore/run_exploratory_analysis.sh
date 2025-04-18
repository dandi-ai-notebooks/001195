#!/bin/bash
# This script runs the exploratory analysis scripts and captures their output

echo "Running electrophysiology exploration..."
python explore_icephys.py > icephys_output.txt
echo "Done."

echo "Running calcium imaging exploration..."
python explore_calcium_imaging.py > calcium_imaging_output.txt
echo "Done."

echo "All exploratory analyses completed. Check the output files for results."