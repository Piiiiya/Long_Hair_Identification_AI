from pathlib import Path
import shutil

ROOT = Path("data/hair_v3")

TRAIN_PER_CLASS = 960
VAL_PER_CLASS = 120
TEST_PER_CLASS = 120

for label in ["long", "short"]:

    train_dir = ROOT / "train" / label
    val_dir = ROOT / "val" / label
    test_dir = ROOT / "test" / label

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Collect ALL images currently belonging to this class
    images = sorted(
        train_dir.glob("*.jpg")
    )

    # Remove duplicate names from any future situation
    images = list(dict.fromkeys(images))

    print()
    print("=" * 50)
    print(label.upper())
    print("Images found:", len(images))

    # We need at least 1200
    if len(images) < 1200:
        print(
            f"WARNING: only {len(images)} images available."
        )
        continue

    # ------------------------------------------------------
    # First 960 -> TRAIN
    # Next 120 -> VAL
    # Next 120 -> TEST
    # ------------------------------------------------------

    train_images = images[:TRAIN_PER_CLASS]
    val_images = images[
        TRAIN_PER_CLASS:
        TRAIN_PER_CLASS + VAL_PER_CLASS
    ]
    test_images = images[
        TRAIN_PER_CLASS + VAL_PER_CLASS:
        TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
    ]

    # Move validation images
    for image in val_images:
        destination = val_dir / image.name
        shutil.move(str(image), str(destination))

    # Move test images
    for image in test_images:
        destination = test_dir / image.name
        shutil.move(str(image), str(destination))

    print("Train:", len(train_images))
    print("Val  :", len(val_images))
    print("Test :", len(test_images))

print()
print("=" * 50)
print("V3 DATASET ORGANIZATION COMPLETE")
print("=" * 50)

for split in ["train", "val", "test"]:

    print()
    print(split.upper())

    for label in ["long", "short"]:

        count = len(
            list(
                (ROOT / split / label).glob("*.jpg")
            )
        )

        print(f"{label:>5}: {count}")