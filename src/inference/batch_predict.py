import os
from src.inference.predictor import predict
from src.config import IMAGE_EXTENSIONS


def batch_predict(folder):

    results = []

    for file in os.listdir(folder):

        if not file.lower().endswith(IMAGE_EXTENSIONS):
            continue

        path = os.path.join(folder, file)

        prediction = predict(path)

        label = os.path.splitext(file)[0]

        results.append({
            "file": file,
            "label": label,
            "prediction": prediction
        })

    return results