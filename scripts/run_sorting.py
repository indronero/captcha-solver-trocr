import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.sort_predictions import sort_folder

input_folder = sys.argv[1]
output_folder = sys.argv[2]

results = sort_folder(input_folder, output_folder)

print(results)