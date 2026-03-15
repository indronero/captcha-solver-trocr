import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.predictor import predict

image = sys.argv[1]

print(predict(image))