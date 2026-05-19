# src\training\dataset_loader.py

import os
import random
import pandas as pd
from datasets import Dataset
from src.config import IMAGE_EXTENSIONS

random.seed(42)


def load_datasets(dataset_paths, num_examples=None):

    all_data = []

    for dataset_path in dataset_paths:

        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        dataset_name = os.path.basename(dataset_path)

        image_files = sorted([
            f for f in os.listdir(dataset_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])

        if len(image_files) == 0:
            raise ValueError(f"No images found in dataset: {dataset_path}")

        for filename in image_files:

            path = os.path.join(dataset_path, filename)
            text = os.path.splitext(filename)[0]

            all_data.append({
                "image_path": path,
                "text": text,
                "source": dataset_name
            })

    if len(all_data) == 0:
        raise ValueError("No valid data found across all datasets.")

    df = pd.DataFrame(all_data)

    # OPTIONAL BALANCING
    if num_examples is not None:

        num_sources = df["source"].nunique()
        samples_per_source = num_examples // num_sources

        df = df.groupby("source", group_keys=False).apply(
            lambda g: g.sample(n=min(samples_per_source, len(g)), random_state=42)
        ).reset_index(drop=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return Dataset.from_pandas(df)