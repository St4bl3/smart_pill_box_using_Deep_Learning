# server.py
import socket
import os
import re
import numpy as np
import cv2
import torch
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Model and data paths
model_path = './pill_model.h5'
dataset_dir = './Data'
image_save_path = 'received_image.jpg'

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
    detected_pills = []
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
        detected_pills.append((x1, y1, x2, y2, pill_name))
    return detected_pills

# Socket configuration
SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 5000

# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(1)
print(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")

def receive_image(client_socket):
    # Receive the header containing the image size
    header = client_socket.recv(8)
    image_size = int(header.decode().strip())
    print(f"[*] Image size: {image_size} bytes")

    # Receive the image
    image_data = b''
    while len(image_data) < image_size:
        packet = client_socket.recv(4096)
        if not packet:
            break
        image_data += packet

    # Save the received image
    with open(image_save_path, 'wb') as f:
        f.write(image_data)
    print("[*] Image received and saved.")

    # Load and process the image
    image = cv2.imread(image_save_path)
    if image is None:
        print("Failed to read image")
        client_socket.send("Failed to read image".encode())
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = detect_pills(image)
    high_conf_boxes = [bbox for bbox in bboxes if bbox[4] >= 0.2]  # Filter out low-confidence detections
    print(f"Detected {len(high_conf_boxes)} pills with high confidence.")
    pill_labels = get_pill_labels(dataset_dir)
    print(f"Pill labels: {pill_labels}")
    detected_pills = classify_pills(image, high_conf_boxes, pill_labels)

    # Send a text response with detected pills
    if detected_pills:
        response = "Detected pills:\n" + "\n".join([f"({x1}, {y1}, {x2}, {y2}): {name}" for x1, y1, x2, y2, name in detected_pills])
    else:
        response = "No pill detected"

    client_socket.send(response.encode())

def main():
    while True:
        client_socket, client_address = server_socket.accept()
        print(f"[*] Accepted connection from {client_address}")
        receive_image(client_socket)
        client_socket.close()

if __name__ == "__main__":
    main()
