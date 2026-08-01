# preprocessing/read_metadata.py

path = r"FULL_PATH_TO_PARAMLOG.DAT"

with open(path,"r",errors="ignore") as f:
    text = f.read()

print(text[:5000])