from pathlib import Path
import shutil

ROOT = Path("data/hair_v3")

TRAIN = 960
VAL = 120
TEST = 120
TOTAL = TRAIN + VAL + TEST

print("=" * 60)
print("FINALIZING V3 DATASET SPLIT")
print("=" * 60)

for label in ["long", "short"]:

    print()
    print(f"Processing: {label}")

    # ------------------------------------------------------
    # Collect every image from all three existing splits
    # ------------------------------------------------------

    all_images = []

    for split in ["train", "val", "test"]:

        folder = ROOT / split / label

        if folder.exists():
            all_images.extend(folder.glob("*.jpg"))

    # Remove duplicate paths
    all_images = list(dict.fromkeys(all_images))

    print("Images found:", len(all_images))

    if len(all_images) < TOTAL:
        raise RuntimeError(
            f"{label} has only {len(all_images)} images. "
            f"Need {TOTAL}."
        )

    # ------------------------------------------------------
    # Create temporary folder
    # ------------------------------------------------------

    temp = ROOT / "_temp" / label
    temp.mkdir(parents=True, exist_ok=True)

    # Move ALL images to temporary location first.
    # This prevents collisions between existing splits.
    temp_images = []

    for i, image in enumerate(all_images):

        destination = temp / f"{label}_{i:05d}.jpg"

        shutil.move(
            str(image),
            str(destination)
        )

        temp_images.append(destination)

    # ------------------------------------------------------
    # Assign final splits
    # ------------------------------------------------------

    assignments = [
        ("train", temp_images[:TRAIN]),
        ("val", temp_images[TRAIN:TRAIN + VAL]),
        ("test", temp_images[TRAIN + VAL:TOTAL]),
    ]

    for split, images in assignments:

        destination_folder = ROOT / split / label
        destination_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Remove any remaining jpg files
        for old in destination_folder.glob("*.jpg"):
            old.unlink()

        for i, image in enumerate(images):

            destination = (
                destination_folder /
                f"{label}_{i:05d}.jpg"
            )

            shutil.move(
                str(image),
                str(destination)
            )

        print(
            f"{split.upper():5} : {len(images)}"
        )

    # ------------------------------------------------------
    # Clean temporary directory
    # ------------------------------------------------------

    if temp.exists() and not any(temp.iterdir()):
        temp.rmdir()


# ----------------------------------------------------------
# FINAL VERIFICATION
# ----------------------------------------------------------

print()
print("=" * 60)
print("FINAL V3 DATASET")
print("=" * 60)

for split in ["train", "val", "test"]:

    print()
    print(split.upper())

    for label in ["long", "short"]:

        folder = ROOT / split / label

        count = len(
            list(folder.glob("*.jpg"))
        )

        print(
            f"{label:>5}: {count}"
        )

print()
print("Expected:")
print("TRAIN -> long 960 / short 960")
print("VAL   -> long 120 / short 120")
print("TEST  -> long 120 / short 120")