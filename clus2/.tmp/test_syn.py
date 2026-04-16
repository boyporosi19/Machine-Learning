import json
with open('clustering_project_pca.ipynb') as f:
    nb=json.load(f)
c = "\n".join(["".join(cell['source']) for cell in nb['cells'] if cell['cell_type'] == 'code'])
c = c.replace("display(", "print(")
with open('.tmp/run_pca.py', 'w') as f: f.write(c)
