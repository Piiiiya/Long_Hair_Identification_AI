from huggingface_hub import hf_hub_download
import numpy as np
import os

REPO = "HairCS2027/HairCS"

print("=" * 70)
print("HAIRCS DATA INSPECTION")
print("=" * 70)

# Download one example NPZ
filename = "data/v0/00000.npz"

print(f"\nDownloading/locating: {filename}")

path = hf_hub_download(
    repo_id=REPO,
    filename=filename,
    repo_type="dataset"
)

print("\nFile:")
print(path)

print("\nFile size:")
print(f"{os.path.getsize(path) / (1024 * 1024):.2f} MB")

data = np.load(path, allow_pickle=True)

print("\nNPZ contents:")
print("-" * 70)

for key in data.files:
    value = data[key]

    print(f"\nKEY: {key}")
    print(f"Type : {type(value)}")

    try:
        print(f"Shape: {value.shape}")
        print(f"Dtype: {value.dtype}")
    except Exception:
        pass

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)