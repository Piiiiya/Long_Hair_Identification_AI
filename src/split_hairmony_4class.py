from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABEL_FILE = PROJECT_ROOT / "hairmony" / "hairmony_4class_labels.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "hairmony_4class"

OUTPUT_DIR = PROJECT_ROOT / "data" / "hairmony_4class_split"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = ["bald", "short", "medium", "long"]


# ============================================================
# CHECK PATHS
# ============================================================

print("=" * 70)
print("HAIRMONY 4-CLASS DATASET SPLITTER")
print("=" * 70)

if not LABEL_FILE.exists():
    raise FileNotFoundError(
        f"\nLabel file not found:\n{LABEL_FILE}"
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"\nImage directory not found:\n{IMAGE_DIR}"
    )

print(f"\nLabel file : {LABEL_FILE}")
print(f"Image dir  : {IMAGE_DIR}")
print(f"Output dir : {OUTPUT_DIR}")


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading labels...")

df = pd.read_csv(LABEL_FILE)

required_columns = {"image_name", "class"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"CSV must contain columns: {required_columns}"
    )

df = df[["image_name", "class"]].copy()

df["image_name"] = df["image_name"].astype(str).str.strip()
df["class"] = df["class"].astype(str).str.strip().str.lower()


# ============================================================
# VALIDATE CLASSES
# ============================================================

print("\nClasses found:")

print(df["class"].value_counts())

unknown_classes = sorted(
    set(df["class"].unique()) - set(CLASSES)
)

if unknown_classes:
    raise ValueError(
        f"\nUnknown classes found: {unknown_classes}"
    )


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(subset=["image_name"]).reset_index(drop=True)

after = len(df)

print(f"\nDuplicate labels removed: {before - after}")


# ============================================================
# CHECK IMAGE FILES
# ============================================================

print("\nChecking image files...")

existing_mask = df["image_name"].apply(
    lambda x: (IMAGE_DIR / x).is_file()
)

missing_count = int((~existing_mask).sum())

if missing_count > 0:

    print(f"WARNING: Missing images: {missing_count}")

    missing_files = df.loc[
        ~existing_mask, "image_name"
    ].tolist()

    print("\nFirst missing files:")

    for filename in missing_files[:20]:
        print("  ", filename)

    df = df[existing_mask].reset_index(drop=True)

else:

    print("All labeled images found.")


print(f"\nImages available for splitting: {len(df)}")


# ============================================================
# FIRST SPLIT
# TRAIN = 70%
# TEMP = 30%
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=(1.0 - TRAIN_RATIO),
    stratify=df["class"],
    random_state=RANDOM_STATE
)


# ============================================================
# SECOND SPLIT
# VAL = 15%
# TEST = 15%
#
# temp = 30%
# Half of temp = 15%
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["class"],
    random_state=RANDOM_STATE
)


# ============================================================
# PRINT SPLIT COUNTS
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COUNTS")
print("=" * 70)

print(f"\nTRAIN      : {len(train_df)}")
print(f"VALIDATION : {len(val_df)}")
print(f"TEST       : {len(test_df)}")
print(f"TOTAL      : {len(train_df) + len(val_df) + len(test_df)}")


print("\nTRAIN distribution:")
print(train_df["class"].value_counts().sort_index())

print("\nVALIDATION distribution:")
print(val_df["class"].value_counts().sort_index())

print("\nTEST distribution:")
print(test_df["class"].value_counts().sort_index())


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

print("\nCreating output directories...")

for split_name in ["train", "val", "test"]:

    for class_name in CLASSES:

        directory = OUTPUT_DIR / split_name / class_name

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_split(split_df, split_name):

    print(f"\nCopying {split_name} images...")

    copied = 0
    failed = 0

    for _, row in split_df.iterrows():

        filename = row["image_name"]
        class_name = row["class"]

        source = IMAGE_DIR / filename

        destination = (
            OUTPUT_DIR
            / split_name
            / class_name
            / filename
        )

        try:

            shutil.copy2(source, destination)

            copied += 1

        except Exception as e:

            failed += 1

            print(
                f"ERROR copying {filename}: {e}"
            )

    print(
        f"{split_name}: copied={copied}, failed={failed}"
    )

    return copied, failed


# ============================================================
# COPY DATASETS
# ============================================================

train_copied, train_failed = copy_split(
    train_df,
    "train"
)

val_copied, val_failed = copy_split(
    val_df,
    "val"
)

test_copied, test_failed = copy_split(
    test_df,
    "test"
)


# ============================================================
# SAVE SPLIT CSV FILES
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train_labels.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "val_labels.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test_labels.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

total_copied = (
    train_copied
    + val_copied
    + test_copied
)

total_failed = (
    train_failed
    + val_failed
    + test_failed
)

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(f"\nExpected images : {len(df)}")
print(f"Copied images   : {total_copied}")
print(f"Failed images   : {total_failed}")

print(f"\nDataset location:")
print(OUTPUT_DIR)

if total_failed == 0 and total_copied == len(df):

    print("\nSUCCESS!")
    print("All images were split successfully.")

else:

    print("\nWARNING!")
    print("Some images were not copied.")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)