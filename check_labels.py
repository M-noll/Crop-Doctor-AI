import os
DATA_DIR = "plantvillage dataset/color"
folders = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])
for i, name in enumerate(folders):
    print(f"Index {i}: {name}")