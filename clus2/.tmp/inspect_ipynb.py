import json

with open('/Users/dppkb/Documents/GitHub/Machine-Learning/clus2/clustering_project.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source'])
    print(f"Cell {i}: {cell['cell_type']}, len: {len(source)}")
    print(source[:100].replace('\n', '\\n') + '...\n')
