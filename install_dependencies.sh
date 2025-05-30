#!/bin/bash
# Activate your virtual environment before running this script!
while IFS= read -r requirement || [ -n "$requirement" ]; do
    # Skip empty lines or comments
    if [[ -z "$requirement" || "$requirement" == \#* ]]; then
        continue
    fi
    echo "Installing $requirement..."
    pip install "$requirement" || echo "Failed to install $requirement, skipping..."
done < requirements.txt
