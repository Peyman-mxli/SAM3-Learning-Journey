import json
import glob
import os

target_dir = r"c:\Users\bruni\Documents\Experiments\Archive_Zero"
notebooks = [f for f in os.listdir(target_dir) if f.endswith('.ipynb') and '_f_' not in f and 'extra' not in f]
notebooks.sort()

print("--- Notebook Analysis ---")
for nb_name in notebooks:
    path = os.path.join(target_dir, nb_name)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            cells = nb.get("cells", [])
            print(f"\nNotebook: {nb_name} ({len(cells)} cells)")
            for i, cell in enumerate(cells):
                if cell["cell_type"] == "markdown":
                    source = "".join(cell.get("source", []))
                    headers = [line.strip() for line in source.split('\n') if line.startswith('#')]
                    if headers:
                        print(f"  Cell {i} Headers: {', '.join(headers)}")
    except Exception as e:
        print(f"\nNotebook: {nb_name} - Error reading: {e}")
