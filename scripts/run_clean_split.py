# scripts\run_clean_split.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.clean_split import clean_and_split
from src.config import RAW_PATH

if __name__ == "__main__":
    clean_and_split(RAW_PATH, "datasets")