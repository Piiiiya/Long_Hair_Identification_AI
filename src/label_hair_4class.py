import os
import csv
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "hair_4class",
    "source_hisa"
)

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "hair_4class",
    "raw"
)

LOG_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "hair_4class",
    "label_progress.csv"
)

CLASSES = {
    "b": "bald",
    "s": "short",
    "m": "medium",
    "l": "long",
}

# ============================================================
# CREATE FOLDERS
# ============================================================

for class_name in CLASSES.values():
    os.makedirs(
        os.path.join(RAW_DIR, class_name),
        exist_ok=True
    )

os.makedirs(
    os.path.dirname(LOG_FILE),
    exist_ok=True
)

# ============================================================
# FIND IMAGES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

images = []

for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        ext = os.path.splitext(file)[1].lower()

        if ext in IMAGE_EXTENSIONS:
            images.append(
                os.path.join(root, file)
            )

images.sort()

if not images:
    raise RuntimeError(
        f"No images found in:\n{SOURCE_DIR}"
    )

# ============================================================
# LOAD PREVIOUS PROGRESS
# ============================================================

labeled = {}

if os.path.exists(LOG_FILE):

    with open(
        LOG_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            labeled[row["source"]] = row["class"]

print()
print("=" * 60)
print("HAIR 4-CLASS LABELING TOOL")
print("=" * 60)
print(f"Images found       : {len(images)}")
print(f"Already labeled    : {len(labeled)}")
print(f"Remaining          : {len(images) - len(labeled)}")
print("=" * 60)
print()

# ============================================================
# SAVE LABEL
# ============================================================

def save_label(source, class_name):

    labeled[source] = class_name

    destination = os.path.join(
        RAW_DIR,
        class_name,
        os.path.basename(source)
    )

    # Avoid overwriting an existing file
    if not os.path.exists(destination):
        shutil.copy2(source, destination)

    # Rewrite progress file
    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "source",
            "class"
        ])

        for src, cls in labeled.items():
            writer.writerow([
                src,
                cls
            ])

# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Hair Length Dataset Labeler"
)

root.geometry(
    "1100x850"
)

root.configure(
    bg="#202124"
)

current_index = 0

# Find first unlabeled image
while (
    current_index < len(images)
    and images[current_index] in labeled
):
    current_index += 1

# ============================================================
# UI ELEMENTS
# ============================================================

title_label = tk.Label(
    root,
    text="HAIR 4-CLASS DATASET LABELER",
    font=("Arial", 22, "bold"),
    bg="#202124",
    fg="white"
)

title_label.pack(
    pady=10
)

info_label = tk.Label(
    root,
    text="",
    font=("Arial", 14),
    bg="#202124",
    fg="white"
)

info_label.pack(
    pady=5
)

image_label = tk.Label(
    root,
    bg="#202124"
)

image_label.pack(
    pady=10
)

instruction_label = tk.Label(
    root,
    text=(
        "B = Bald     S = Short     M = Medium     "
        "L = Long     N = Skip     ← = Previous"
    ),
    font=("Arial", 16, "bold"),
    bg="#202124",
    fg="white"
)

instruction_label.pack(
    pady=10
)

status_label = tk.Label(
    root,
    text="",
    font=("Arial", 13),
    bg="#202124",
    fg="white"
)

status_label.pack(
    pady=5
)

# ============================================================
# DISPLAY IMAGE
# ============================================================

current_photo = None


def display_image():

    global current_photo

    if current_index >= len(images):

        image_label.config(
            image="",
            text="ALL IMAGES COMPLETED",
            font=("Arial", 30, "bold"),
            fg="white"
        )

        info_label.config(
            text="Dataset labeling finished."
        )

        status_label.config(
            text=""
        )

        return

    source = images[current_index]

    try:

        image = Image.open(source)
        image = image.convert("RGB")

        # Keep aspect ratio
        image.thumbnail(
            (750, 600)
        )

        current_photo = ImageTk.PhotoImage(
            image
        )

        image_label.config(
            image=current_photo,
            text=""
        )

    except Exception as e:

        image_label.config(
            image="",
            text=f"Could not open image:\n{e}",
            font=("Arial", 18),
            fg="red"
        )

    labeled_count = len(labeled)

    info_label.config(
        text=(
            f"Image {current_index + 1} / {len(images)}"
            f"     |     Labeled: {labeled_count}"
        )
    )

    filename = os.path.basename(source)

    status_label.config(
        text=filename
    )


# ============================================================
# KEYBOARD ACTION
# ============================================================

def handle_key(event):

    global current_index

    key = event.keysym.lower()

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    if key in CLASSES:

        if current_index >= len(images):
            return

        source = images[current_index]

        class_name = CLASSES[key]

        save_label(
            source,
            class_name
        )

        current_index += 1

        while (
            current_index < len(images)
            and images[current_index] in labeled
        ):
            current_index += 1

        display_image()

    # --------------------------------------------------------
    # SKIP
    # --------------------------------------------------------

    elif key == "n":

        current_index += 1

        display_image()

    # --------------------------------------------------------
    # PREVIOUS
    # --------------------------------------------------------

    elif key == "left":

        if current_index > 0:

            current_index -= 1

            # Go backwards until we find something
            # that can be viewed
            display_image()


# ============================================================
# BUTTONS
# ============================================================

button_frame = tk.Frame(
    root,
    bg="#202124"
)

button_frame.pack(
    pady=10
)


def create_button(
    text,
    key,
    color
):

    button = tk.Button(
        button_frame,
        text=f"{text}\n[{key.upper()}]",
        font=("Arial", 13, "bold"),
        width=12,
        height=2,
        command=lambda: handle_key(
            type(
                "Event",
                (),
                {"keysym": key}
            )
        ),
        bg=color,
        fg="white"
    )

    button.pack(
        side="left",
        padx=5
    )


create_button(
    "BALD",
    "b",
    "#333333"
)

create_button(
    "SHORT",
    "s",
    "#555555"
)

create_button(
    "MEDIUM",
    "m",
    "#777777"
)

create_button(
    "LONG",
    "l",
    "#999999"
)

create_button(
    "SKIP",
    "n",
    "#444444"
)

# ============================================================
# KEYBOARD BINDING
# ============================================================

root.bind(
    "<Key>",
    handle_key
)

# ============================================================
# START
# ============================================================

display_image()

root.mainloop()