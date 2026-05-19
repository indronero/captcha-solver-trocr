# src\data\clean_split.py

import os
import random
import shutil
from PIL import Image
from src.config import IMAGE_EXTENSIONS, MAX_LENGTH

SEED = 42


def clean_label(filename):
    return os.path.splitext(filename)[0].strip()


def is_valid_label(label):
    return label.isalnum() and 0 < len(label) <= MAX_LENGTH


def is_valid_image(path):
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        img.load()
        return True
    except:
        return False


def split_data(files, seed=42, train_ratio=0.8, val_ratio=0.1):
    rnd = random.Random(seed)
    rnd.shuffle(files)

    n = len(files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        files[:train_end],
        files[train_end:val_end],
        files[val_end:]
    )


def clean_and_split(raw_base, output_base, overwrite=False):
    """
    raw_base: datasets/raw
    output_base: datasets/
    """

    for dataset_name in os.listdir(raw_base):

        raw_path = os.path.join(raw_base, dataset_name)
        
        train_dir = os.path.join(output_base, "train", dataset_name)
        val_dir = os.path.join(output_base, "val", dataset_name)
        test_dir = os.path.join(output_base, "test", dataset_name)

        # SKIP if already processed
        if not overwrite and os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
            print(f"Skipping {dataset_name} (already processed)")
            continue

        print(f"\nProcessing dataset: {dataset_name}")

        valid_files = []
        seen_labels = set()   

        for file in os.listdir(raw_path):

            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            path = os.path.join(raw_path, file)
            label = clean_label(file)

            if not is_valid_label(label):
                continue

            # only remove duplicates WITHIN dataset
            if label in seen_labels:
                continue

            if not is_valid_image(path):
                continue

            valid_files.append((file, label))
            seen_labels.add(label)

        if len(valid_files) == 0:
            print("No valid files, skipping.")
            continue

        print(f"Valid samples: {len(valid_files)}")

        train_files, val_files, test_files = split_data(valid_files, seed=SEED)

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        for split_name, files in splits.items():

            split_dir = os.path.join(output_base, split_name, dataset_name)

            # SAFE OVERWRITE
            if overwrite and os.path.exists(split_dir):
                shutil.rmtree(split_dir)

            os.makedirs(split_dir, exist_ok=True)

            for file, label in files:
                src = os.path.join(raw_path, file)
                dst = os.path.join(split_dir, f"{label}.png")

                try:
                    img = Image.open(src).convert("RGB")
                    img.save(dst)
                except:
                    continue

        print({
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files)
        })