import os
import shutil
from src.inference.predictor import predict
from src.config import IMAGE_EXTENSIONS


def sort_folder(input_folder, output_folder):

    correct_dir = os.path.join(output_folder, "correct")
    incorrect_dir = os.path.join(output_folder, "incorrect")

    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)

    correct = 0
    incorrect = 0

    for file in os.listdir(input_folder):

        if not file.lower().endswith(IMAGE_EXTENSIONS):
            continue

        path = os.path.join(input_folder, file)

        prediction = predict(path)

        label = os.path.splitext(file)[0]

        if prediction == label:

            shutil.copy(path, os.path.join(correct_dir, file))
            correct += 1

        else:

            shutil.copy(path, os.path.join(incorrect_dir, file))
            incorrect += 1

    accuracy = correct / (correct + incorrect)

    return {
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy
    }