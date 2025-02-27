#!/usr/bin/env python
"""
Convert Mermaid diagrams to PNG images.

This script converts Mermaid (.mmd) files to PNG images using the Mermaid CLI.
Make sure you have Node.js and the Mermaid CLI installed:
npm install -g @mermaid-js/mermaid-cli
"""

import os
import subprocess
import glob
from pathlib import Path


def convert_mermaid_to_png(mermaid_file, output_file):
    """Convert a Mermaid file to PNG."""
    try:
        # Command to run Mermaid CLI
        cmd = [
            'npx', 
            'mmdc', 
            '-i', mermaid_file, 
            '-o', output_file, 
            '-b', 'transparent'
        ]
        
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Converted {mermaid_file} to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {mermaid_file}: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main function to convert all Mermaid files in the directory."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Find all .mmd files in the directory
    mermaid_files = glob.glob(str(script_dir / "*.mmd"))
    
    # Convert each file
    for mmd_file in mermaid_files:
        output_file = str(Path(mmd_file).with_suffix('.png'))
        convert_mermaid_to_png(mmd_file, output_file)
        
    print(f"Processed {len(mermaid_files)} Mermaid files")


if __name__ == "__main__":
    main() 