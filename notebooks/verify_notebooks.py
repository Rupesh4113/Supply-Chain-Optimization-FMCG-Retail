"""
Verify that code cells in all notebooks execute without syntax errors.
"""

import json
import os
import sys

notebooks = [
    "01_data_generation.ipynb",
    "02_data_preparation.ipynb",
    "03_eda.ipynb",
    "04_clustering.ipynb",
    "05_classification.ipynb",
    "06_business_simulation.ipynb"
]

all_passed = True
for nb_file in notebooks:
    path = os.path.join("notebooks", nb_file)
    with open(path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    print(f"Checking {nb_file}: {len(nb_data['cells'])} cells found.")
    for i, cell in enumerate(nb_data["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            try:
                compile(code, f"{nb_file}_cell_{i}", "exec")
            except SyntaxError as e:
                print(f"  [ERROR] Syntax error in {nb_file} cell {i}: {e}")
                all_passed = False

if all_passed:
    print("\nAll 6 notebooks syntax verified successfully!")
else:
    sys.exit(1)
