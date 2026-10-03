import os
import shutil
import scipy.io as sio
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "hair"
    / "Market-1501_Attribute"
    / "market_attribute.mat"
)

MARKET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "hair"
    / "Market-1501"
    / "Market-1501-v15.09.15"
)

OUTPUT_LONG = PROJECT_ROOT / "data" / "hair" / "long"
OUTPUT_SHORT = PROJECT_ROOT / "data" / "hair" / "short"


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

OUTPUT_LONG.mkdir(parents=True, exist_ok=True)
OUTPUT_SHORT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD ATTRIBUTE DATA
# ============================================================

print("=" * 70)
print("Loading Market-1501 hair attributes...")
print("=" * 70)

mat = sio.loadmat(MAT_FILE)["market_attribute"][0, 0]

train = mat["train"][0, 0]
test = mat["test"][0, 0]


# ============================================================
# BUILD ID -> HAIR LABEL MAPPING
# ============================================================

identity_to_hair = {}


def clean_identity(value):
    """
    Convert MATLAB nested string such as:
        array(['0002'], dtype='<U4')
    into:
        '0002'
    """
    while hasattr(value, "shape") and value.size == 1:
        value = value.item()

    return str(value).strip()


def add_attributes(dataset):
    image_indices = dataset["image_index"].flatten()
    hair_labels = dataset["hair"].flatten()

    for identity, hair_label in zip(image_indices, hair_labels):
        identity = clean_identity(identity)
        hair_label = int(hair_label)

        identity_to_hair[identity] = hair_label


add_attributes(train)
add_attributes(test)


# ============================================================
# DISPLAY LABEL COUNTS
# ============================================================

short_count = sum(1 for v in identity_to_hair.values() if v == 1)
long_count = sum(1 for v in identity_to_hair.values() if v == 2)

print(f"Total identities: {len(identity_to_hair)}")
print(f"Short-hair identities: {short_count}")
print(f"Long-hair identities:  {long_count}")


# ============================================================
# MARKET-1501 IMAGE FOLDERS
# ============================================================

image_folders = [
    MARKET_ROOT / "bounding_box_train",
    MARKET_ROOT / "bounding_box_test",
]


# ============================================================
# COPY IMAGES
# ============================================================

copied_short = 0
copied_long = 0
skipped = 0
unknown_identity = 0


print()
print("=" * 70)
print("Copying images...")
print("=" * 70)


for folder in image_folders:

    if not folder.exists():
        print(f"WARNING: Folder not found: {folder}")
        continue

    print(f"\nProcessing: {folder.name}")

    for image_path in folder.glob("*.jpg"):

        filename = image_path.name

        # Market-1501 filenames begin with identity:
        # 0002_c1s1_000451_01.jpg
        identity = filename.split("_")[0]

        hair_label = identity_to_hair.get(identity)

        if hair_label is None:
            unknown_identity += 1
            continue

        # 1 = short hair
        if hair_label == 1:

            destination = OUTPUT_SHORT / filename

            if not destination.exists():
                shutil.copy2(image_path, destination)
                copied_short += 1

        # 2 = long hair
        elif hair_label == 2:

            destination = OUTPUT_LONG / filename

            if not destination.exists():
                shutil.copy2(image_path, destination)
                copied_long += 1

        else:
            skipped += 1


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("HAIR DATASET PREPARATION COMPLETE")
print("=" * 70)

print(f"Short-hair images copied : {copied_short}")
print(f"Long-hair images copied  : {copied_long}")
print(f"Unknown identities       : {unknown_identity}")
print(f"Skipped                  : {skipped}")

print()
print("Output folders:")
print(f"SHORT: {OUTPUT_SHORT}")
print(f"LONG : {OUTPUT_LONG}")

print()
print("Existing files now:")

print(
    "Short folder:",
    len(list(OUTPUT_SHORT.glob("*.jpg")))
)

print(
    "Long folder :",
    len(list(OUTPUT_LONG.glob("*.jpg")))
)

print("=" * 70)