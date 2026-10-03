from datasets import load_dataset
from PIL import Image
from pathlib import Path
from collections import Counter
import io

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

DATASET_NAME = "Zhincore/synthetic-human-portrait-attributes"

OUTPUT_DIR = Path("data") / "hair_v3"

# Number of useful images we want locally
TARGET_PER_CLASS = 1500

CLASSES = {
    "long": "long",
    "short": "short"
}

# ---------------------------------------------------------
# CREATE FOLDERS
# ---------------------------------------------------------

for split in ["train", "val", "test"]:
    for label in CLASSES:
        (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("V3 PORTRAIT DATASET PREPARATION")
print("=" * 60)
print(f"Dataset : {DATASET_NAME}")
print(f"Target  : {TARGET_PER_CLASS} images per class")
print()

# ---------------------------------------------------------
# LOAD DATASET IN STREAMING MODE
# ---------------------------------------------------------

print("Connecting to Hugging Face...")

ds = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

print("Dataset connected.")
print("Reading portrait records...")
print()

# ---------------------------------------------------------
# COLLECT IMAGES
# ---------------------------------------------------------

counts = Counter()
saved = Counter()

MAX_RECORDS = 13598

for index, item in enumerate(ds):

    if index >= MAX_RECORDS:
        break

    hair_length = item["hair_length"]

    # Ignore bald and buzzcut
    if hair_length not in CLASSES:
        continue

    # Stop collecting a class once target reached
    if saved[hair_length] >= TARGET_PER_CLASS:
        if all(saved[c] >= TARGET_PER_CLASS for c in CLASSES):
            break
        continue

    counts[hair_length] += 1

    try:
        image = item["image"]

        if image is None:
            continue

        # Convert to RGB
        image = image.convert("RGB")

        # -------------------------------------------------
        # SPLIT
        # -------------------------------------------------

        current_number = saved[hair_length]

        if current_number < 1200:
            split = "train"
        elif current_number < 1350:
            split = "val"
        else:
            split = "test"

        filename = (
            f"{hair_length}_{current_number:05d}.jpg"
        )

        output_path = (
            OUTPUT_DIR
            / split
            / hair_length
            / filename
        )

        image.save(
            output_path,
            format="JPEG",
            quality=95
        )

        saved[hair_length] += 1

        if saved[hair_length] % 100 == 0:
            print(
                f"{hair_length:>5}: "
                f"{saved[hair_length]}/{TARGET_PER_CLASS}"
            )

    except Exception as e:
        print(
            f"Could not save record {index}: {e}"
        )

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print("=" * 60)
print("V3 DATASET PREPARATION COMPLETE")
print("=" * 60)

print()
print("Images saved:")

for label in CLASSES:
    print(f"  {label:>5}: {saved[label]}")

print()
print("Dataset location:")
print(OUTPUT_DIR.resolve())

print()
print("Folder structure:")
print("data/hair_v3/")
print("├── train/")
print("│   ├── long/")
print("│   └── short/")
print("├── val/")
print("│   ├── long/")
print("│   └── short/")
print("└── test/")
print("    ├── long/")
print("    └── short/")