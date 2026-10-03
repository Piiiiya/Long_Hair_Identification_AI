from datasets import load_dataset
from pathlib import Path
from PIL import Image
import time

DATASET_NAME = "Zhincore/synthetic-human-portrait-attributes"
ROOT = Path("data/hair_v3")

TARGET_SHORT = 1200

# Current short images are all that matter now
existing = []

for split in ["train", "val", "test"]:
    folder = ROOT / split / "short"
    folder.mkdir(parents=True, exist_ok=True)
    existing.extend(folder.glob("*.jpg"))

current = len(existing)

print("=" * 60)
print("FINISHING V3 SHORT-HAIR DATASET")
print("=" * 60)
print(f"Existing short images : {current}")
print(f"Required short images : {TARGET_SHORT}")
print(f"Missing               : {max(0, TARGET_SHORT - current)}")
print()

if current >= TARGET_SHORT:
    print("Short dataset is already complete.")
    raise SystemExit

print("Connecting to Hugging Face...")

ds = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

print("Connected.")
print("Downloading missing short images...")
print()

saved = current
failed = 0

for item in ds:

    if saved >= TARGET_SHORT:
        break

    try:
        if item["hair_length"] != "short":
            continue

        image = item["image"]

        if image is None:
            continue

        image = image.convert("RGB")

        # Give temporary sequential names.
        filename = f"short_extra_{saved:05d}.jpg"

        # Put all new images in train temporarily.
        output = ROOT / "train" / "short" / filename

        if output.exists():
            continue

        image.save(
            output,
            "JPEG",
            quality=95
        )

        saved += 1

        if saved % 25 == 0:
            print(f"short -> {saved}/{TARGET_SHORT}")

    except Exception as e:

        failed += 1
        print(
            f"[SKIP] download failure #{failed}: {e}"
        )

        time.sleep(1)
        continue

print()
print("=" * 60)
print("DOWNLOAD FINISHED")
print("=" * 60)
print(f"Short images available: {saved}")
print(f"Failed downloads      : {failed}")