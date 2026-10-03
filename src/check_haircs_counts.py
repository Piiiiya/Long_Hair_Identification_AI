from huggingface_hub import hf_hub_download
import pandas as pd

REPO = "HairCS2027/HairCS"

versions = [
    "v0", "v1", "v2", "v3", "v4",
    "v5", "v6", "v7", "v8", "v9",
    "v10", "v11", "v12"
]

classes = [
    "short",
    "bob",
    "shoulder",
    "long"
]

print()
print("=" * 65)
print("HairCS LABEL COUNTS")
print("=" * 65)
print(
    f"{'VERSION':<10}"
    f"{'SHORT':>10}"
    f"{'BOB':>10}"
    f"{'SHOULDER':>12}"
    f"{'LONG':>10}"
    f"{'TOTAL':>10}"
)
print("-" * 65)

for version in versions:

    filename = f"label/{version}.csv"

    try:

        path = hf_hub_download(
            repo_id=REPO,
            filename=filename,
            repo_type="dataset"
        )

        df = pd.read_csv(path)

        df["class"] = (
            df["class"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        counts = df["class"].value_counts()

        short_count = int(counts.get("short", 0))
        bob_count = int(counts.get("bob", 0))
        shoulder_count = int(counts.get("shoulder", 0))
        long_count = int(counts.get("long", 0))

        total = (
            short_count
            + bob_count
            + shoulder_count
            + long_count
        )

        print(
            f"{version:<10}"
            f"{short_count:>10}"
            f"{bob_count:>10}"
            f"{shoulder_count:>12}"
            f"{long_count:>10}"
            f"{total:>10}"
        )

    except Exception as e:

        print()
        print(f"ERROR reading {version}: {e}")

print("-" * 65)
print()
print("Mapping we will use:")
print("short              -> SHORT")
print("bob + shoulder     -> MEDIUM")
print("long               -> LONG")
print()