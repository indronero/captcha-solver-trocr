import os
import random
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

COMMON_SUBDIRS = ['', 'images', 'train', 'data', 'sample', 'samples', 'captchas']

random.seed(42)

def load_datasets(dataset_paths, num_examples):

    all_data = []

    for i, dataset_path in enumerate(dataset_paths, 1):

        if not os.path.exists(dataset_path):
            continue

        root_len = len(os.listdir(dataset_path))

        images_dir = None

        if root_len >= 1:
            images_dir = dataset_path
        else:
            for sub in COMMON_SUBDIRS:
                candidate = os.path.join(dataset_path, sub)
                if os.path.exists(candidate) and len(os.listdir(candidate)) > 0:
                    images_dir = candidate
                    break

        if images_dir is None:
            continue

        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png",".jpg",".jpeg"))
        ]

        dataset_data = []

        for filename in image_files:

            path = os.path.join(images_dir, filename)

            text = os.path.splitext(filename)[0]

            dataset_data.append({
                "image_path": path,
                "text": text,
                "source": f"dataset_{i}"
            })

        all_data.extend(dataset_data)

    df = pd.DataFrame(all_data)

    num_sources = df["source"].nunique()

    samples_per_source = num_examples // num_sources

    balanced_df = df.groupby("source").apply(
        lambda g: g.sample(n=min(samples_per_source, len(g)), random_state=42)
    ).reset_index(drop=True)

    df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return train_dataset, val_dataset