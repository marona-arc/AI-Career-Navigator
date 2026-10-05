import sys
import re
sys.stdout.reconfigure(encoding='utf-8')

with open('final_project.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

# search for unique industries or value_counts
matches = re.findall(r'r_industry[^\n]+', text)
for m in matches[:10]:
    print(m)

# search for categories in cell 12 output
import nbformat
nb = nbformat.reads(text, as_version=4)
for i in [1, 2, 4, 5, 6, 7, 8, 12]:
    print(f"Cell {i} source:\n", nb.cells[i].source)
