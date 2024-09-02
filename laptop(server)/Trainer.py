import os
import re
import numpy as np
import cv2
import torch
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast, Transpose, Normalize, Compose, CLAHE, RandomRotate90, GaussianBlur, RandomCrop
)

dataset_dir = './Data'
model_path = './pill_model.h5'

# Load the pre-trained YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to detect pills using YOLOv5
def detect_pills(image):
    results = yolo_model(image)
    bboxes = results.xyxy[0].cpu().numpy()  # xyxy bounding boxes
    return bboxes

# Function to extract unique pill names from the dataset directory
def get_pill_labels(dataset_dir):
    pill_names = set()
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
            pill_name = extract_pill_name(filename)
            if pill_name:
                pill_names.add(pill_name)
    return sorted(list(pill_names))

# Function to extract the pill name from the filename
def extract_pill_name(filename):
    # Assuming the filename format is pillname(anything).extension
    match = re.match(r'([a-zA-Z]+)', filename)
    if match:
        return match.group(1)
    return None

# Data augmentation
def get_augmentations():
    return Compose([
        RandomCrop(width=224, height=224),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5),
        RandomBrightnessContrast(p=0.5),
        Transpose(p=0.5),
        CLAHE(p=0.5),
        RandomRotate90(p=0.5),
        GaussianBlur(p=0.5),
        Normalize()  # Normalize should be the last transformation
    ])

# Function to load data for classification
def load_data(dataset_dir, threshold=0.2):  # Lowered threshold to 0.2
    images = []
    labels = []
    aug = get_augmentations()
    print("Loading data...")
    for filename in tqdm(os.listdir(dataset_dir)):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
            pill_name = extract_pill_name(filename)
            if not pill_name:
                continue
            img_path = os.path.join(dataset_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes = detect_pills(img)
            for bbox in bboxes:
                x1, y1, x2, y2, conf, cls = bbox[:6]
                if conf < threshold:
                    continue
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Convert to int for slicing
                pill_img = img[y1:y2, x1:x2]
                pill_img = cv2.resize(pill_img, (224, 224))
                pill_img = np.array(pill_img, dtype='uint8')  # Ensure uint8 type
                augmented = aug(image=pill_img)
                pill_img = augmented['image']
                images.append(pill_img)
                labels.append(pill_name)
    images = np.array(images, dtype='float32') / 255.0
    return images, pd.get_dummies(labels).values

# Function to create the classification model
def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the classification model
def train_model(images, labels):
    if not os.path.exists(model_path):
        model = create_model(num_classes=labels.shape[1])
    else:
        model = tf.keras.models.load_model(model_path)
        if model.output_shape[-1] != labels.shape[1]:
            model = create_model(num_classes=labels.shape[1])
        else:
            # Recompile the model with a new optimizer instance
            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Training model...")
    history = model.fit(images, labels, epochs=25, batch_size=32, verbose=1)  # Increased epochs for better training
    model.save(model_path)  # Save the model manually
    print(f"Model trained and saved to {model_path}")

# Main script execution
if __name__ == "__main__":
    images, labels = load_data(dataset_dir)
    if len(images) == 0 or len(labels) == 0:
        print("No data found. Please ensure the dataset directory contains valid images with pills.")
    else:
        train_model(images, labels)
