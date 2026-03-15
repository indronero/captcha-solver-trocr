import albumentations as A
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor
from src.config import MODEL_NAME, MAX_LENGTH

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

transform_pipeline = A.Compose([
    A.Affine(scale=(0.85,1.15),translate_percent=(-0.1,0.1),rotate=(-10,10),shear=(-10,10),p=0.7),
    A.OpticalDistortion(distort_limit=0.1,p=0.4),
    A.GridDistortion(num_steps=5,distort_limit=0.2,p=0.4),
    A.ElasticTransform(alpha=1,sigma=50,p=0.3),
    A.GaussNoise(std_range=(10.0/255,50.0/255),p=0.5),
    A.Blur(blur_limit=(3,7),p=0.5),
    A.CoarseDropout(num_holes_range=(1,8),hole_height_range=(8,16),hole_width_range=(8,16),fill=0,p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.6),
    A.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=30,val_shift_limit=20,p=0.5),
])

def preprocess_function(examples,is_train=True):

    images=[Image.open(p).convert("RGB") for p in examples["image_path"]]
    texts=examples["text"]

    if is_train:
        augmented=[transform_pipeline(image=np.array(img))["image"] for img in images]
        images=[Image.fromarray(x) for x in augmented]

    pixel_values=processor(images=images,return_tensors="pt").pixel_values

    labels=processor.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    ).input_ids

    labels=[[l if l!=processor.tokenizer.pad_token_id else -100 for l in label] for label in labels]

    return {"pixel_values":pixel_values,"labels":torch.tensor(labels)}