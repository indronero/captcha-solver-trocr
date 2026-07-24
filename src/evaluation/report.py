# src/evaluation/report.py

import os
import json
from datetime import datetime
from src.config import EVAL_OUT


def save_report(report):

    os.makedirs(EVAL_OUT, exist_ok=True)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    file_path = os.path.join(EVAL_OUT, f"eval_{timestamp}.json")

    with open(file_path, "w") as f:
        json.dump(report, f, indent=4)

    print("Saved evaluation report:", file_path)