import os
import re
import numpy as np
import cv2
import torch
from yolov5 import YOLOv5
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

model_path = './pill_model.h5'
dataset_dir = './Data'
image_path = './Test/Test.jpg'

# Load the pre-trained classification model
classification_model = load_model(model_path)

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

# Function to classify detected pills
def classify_pills(image, bboxes, pill_labels, threshold=0.2):  # Lowered threshold to 0.2
    for bbox in tqdm(bboxes, desc="Classifying pills"):
        x1, y1, x2, y2, conf, cls = bbox[:6]
        if conf < threshold:
            continue
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Convert to int for slicing
        pill_img = image[y1:y2, x1:x2]
        pill_img = cv2.resize(pill_img, (224, 224))
        pill_img = np.array(pill_img, dtype='float32') / 255.0
        pill_img = np.expand_dims(pill_img, axis=0)
        predictions = classification_model.predict(pill_img)
        pill_name = pill_labels[np.argmax(predictions)]
        print(f"Pill detected at ({x1}, {y1}, {x2}, {y2}) with name: {pill_name}")

# Main script execution
if __name__ == "__main__":
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
    else:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = detect_pills(image)
            high_conf_boxes = [bbox for bbox in bboxes if bbox[4] >= 0.2]  # Filter out low-confidence detections
            print(f"Detected {len(high_conf_boxes)} pills with high confidence.")
            pill_labels = get_pill_labels(dataset_dir)
            print(f"Pill labels: {pill_labels}")
            classify_pills(image, high_conf_boxes, pill_labels)
