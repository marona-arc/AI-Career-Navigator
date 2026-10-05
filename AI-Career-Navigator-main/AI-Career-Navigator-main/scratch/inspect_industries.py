import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')
import nbformat
import re

with open('final_project.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's search for words like FinTech, Healthcare, Retail, etc.
# Check cell 12 in final_project.ipynb
nb = nbformat.reads(text, as_version=4)
for i in [1, 12, 14, 15]:
    print(f"Cell {i}:\n{nb.cells[i].source}\n")
