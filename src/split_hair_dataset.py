from pathlib import Path
import scipy.io as sio
import shutil
import random
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

MAT_FILE = BASE / "data" / "hair" / "Market-1501_Attribute" / "market_attribute.mat"

MARKET_ROOT = (
    BASE
    / "data"
    / "hair"
    / "Market-1501"
    / "Market-1501-v15.09.15"
)

TRAIN_SOURCE = MARKET_ROOT / "bounding_box_train"
TEST_SOURCE = MARKET_ROOT / "bounding_box_test"

OUTPUT_ROOT = BASE / "data" / "hair_split"


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
TRAIN_RATIO = 0.80


# ============================================================
# LOAD HAIR ATTRIBUTES
# ============================================================

print("=" * 60)
print("Loading Market-1501 hair attributes...")
print("=" * 60)

mat = sio.loadmat(
    MAT_FILE,
    squeeze_me=True,
    struct_as_record=False
)

market_attribute = mat["market_attribute"]

train_ids = list(market_attribute.train.image_index)
train_hair = list(market_attribute.train.hair)

test_ids = list(market_attribute.test.image_index)
test_hair = list(market_attribute.test.hair)


def clean_id(value):
    return str(value).strip()


train_identity_to_hair = {
    clean_id(identity): int(hair)
    for identity, hair in zip(train_ids, train_hair)
}

test_identity_to_hair = {
    clean_id(identity): int(hair)
    for identity, hair in zip(test_ids, test_hair)
}


print(f"Training identities : {len(train_identity_to_hair)}")
print(f"Testing identities  : {len(test_identity_to_hair)}")


# ============================================================
# CREATE IDENTITY-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

random.seed(SEED)

short_train_ids = [
    identity
    for identity, hair in train_identity_to_hair.items()
    if hair == 1
]

long_train_ids = [
    identity
    for identity, hair in train_identity_to_hair.items()
    if hair == 2
]

random.shuffle(short_train_ids)
random.shuffle(long_train_ids)

short_split = int(len(short_train_ids) * TRAIN_RATIO)
long_split = int(len(long_train_ids) * TRAIN_RATIO)

train_ids_final = (
    short_train_ids[:short_split]
    + long_train_ids[:long_split]
)

val_ids_final = (
    short_train_ids[short_split:]
    + long_train_ids[long_split:]
)

random.shuffle(train_ids_final)
random.shuffle(val_ids_final)

train_id_set = set(train_ids_final)
val_id_set = set(val_ids_final)
test_id_set = set(test_identity_to_hair.keys())


print()
print("Identity split:")
print(f"Train identities      : {len(train_id_set)}")
print(f"Validation identities : {len(val_id_set)}")
print(f"Test identities       : {len(test_id_set)}")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    for label in ["short", "long"]:
        folder = OUTPUT_ROOT / split / label
        folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_images(source_folder, identity_to_hair, selected_ids, split_name):

    copied_short = 0
    copied_long = 0
    skipped = 0

    selected_ids = set(selected_ids)

    for image_file in source_folder.glob("*.jpg"):

        filename = image_file.name

        # Example:
        # 0002_c1s1_000451_00.jpg
        identity = filename.split("_")[0]

        if identity not in selected_ids:
            continue

        hair_label = identity_to_hair.get(identity)

        if hair_label == 1:
            destination = OUTPUT_ROOT / split_name / "short"
            copied_short += 1

        elif hair_label == 2:
            destination = OUTPUT_ROOT / split_name / "long"
            copied_long += 1

        else:
            skipped += 1
            continue

        shutil.copy2(
            image_file,
            destination / filename
        )

    return copied_short, copied_long, skipped


# ============================================================
# COPY TRAINING IMAGES
# ============================================================

print()
print("=" * 60)
print("Creating TRAIN split...")
print("=" * 60)

train_short, train_long, train_skipped = copy_images(
    TRAIN_SOURCE,
    train_identity_to_hair,
    train_id_set,
    "train"
)

print(f"Short images : {train_short}")
print(f"Long images  : {train_long}")
print(f"Skipped      : {train_skipped}")


# ============================================================
# COPY VALIDATION IMAGES
# ============================================================

print()
print("=" * 60)
print("Creating VALIDATION split...")
print("=" * 60)

val_short, val_long, val_skipped = copy_images(
    TRAIN_SOURCE,
    train_identity_to_hair,
    val_id_set,
    "val"
)

print(f"Short images : {val_short}")
print(f"Long images  : {val_long}")
print(f"Skipped      : {val_skipped}")


# ============================================================
# COPY OFFICIAL TEST IMAGES
# ============================================================

print()
print("=" * 60)
print("Creating TEST split...")
print("=" * 60)

test_short, test_long, test_skipped = copy_images(
    TEST_SOURCE,
    test_identity_to_hair,
    test_id_set,
    "test"
)

print(f"Short images : {test_short}")
print(f"Long images  : {test_long}")
print(f"Skipped      : {test_skipped}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATASET SPLIT COMPLETE")
print("=" * 60)

print()
print("TRAIN")
print(f"  Short : {train_short}")
print(f"  Long  : {train_long}")
print(f"  Total : {train_short + train_long}")

print()
print("VALIDATION")
print(f"  Short : {val_short}")
print(f"  Long  : {val_long}")
print(f"  Total : {val_short + val_long}")

print()
print("TEST")
print(f"  Short : {test_short}")
print(f"  Long  : {test_long}")
print(f"  Total : {test_short + test_long}")

print()
print(f"Output folder:")
print(OUTPUT_ROOT)

print()
print("Done.")