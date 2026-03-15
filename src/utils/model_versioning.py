import os


def get_next_version(model_dir):

    os.makedirs(model_dir, exist_ok=True)

    versions = []

    for folder in os.listdir(model_dir):

        if folder.startswith("v"):

            try:
                num = int(folder.replace("v", ""))
                versions.append(num)
            except:
                pass

    if not versions:
        next_version = 1
    else:
        next_version = max(versions) + 1

    version_path = os.path.join(model_dir, f"v{next_version}")

    os.makedirs(version_path, exist_ok=True)

    return version_path


def get_latest_version(model_dir):

    if not os.path.exists(model_dir):
        raise Exception("Model directory does not exist")

    versions = []

    for folder in os.listdir(model_dir):

        if folder.startswith("v"):

            try:
                num = int(folder.replace("v", ""))
                versions.append(num)
            except:
                pass

    if not versions:
        raise Exception("No model versions found")

    latest_version = max(versions)

    return os.path.join(model_dir, f"v{latest_version}")