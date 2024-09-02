# client.py
import socket
import os
import time
import cv2
import RPi.GPIO as GPIO
from hx711 import HX711

# Define server address and port
SERVER_HOST = '192.168.170.59'  # Replace with the IP address of your laptop
SERVER_PORT = 5000

# Define the path to save the image temporarily
IMAGE_PATH = '/home/raspie/captured_image.jpg'  # Adjust the path as needed

# HX711 and Servo configuration
DOUT_PIN = 27
SCK_PIN = 17
SERVO_PIN = 22

# Initialize GPIO for the servo
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz frequency
servo.start(0)

# Initialize HX711
hx = HX711(DOUT_PIN, SCK_PIN)
hx.set_reading_format("MSB", "MSB")
hx.set_reference_unit(1)  # You will need to calibrate this value
hx.reset()
hx.tare()

def capture_image():
    cap = cv2.VideoCapture(0)  # Open the first webcam (index 0)
    if not cap.isOpened():
        print("Could not open webcam")
        return False

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        return False

    cv2.imwrite(IMAGE_PATH, frame)
    print(f"[*] Image captured and saved to {IMAGE_PATH}")

    cap.release()
    cv2.destroyAllWindows()
    return True

def get_weight():
    weight = hx.get_weight(5)
    print(f"[*] Current weight: {weight}")
    return weight

def send_image():
    # Move the servo to 0 degrees before sending
    move_servo(0)
    time.sleep(2)

    # Create a socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    print(f"[*] Connected to {SERVER_HOST}:{SERVER_PORT}")

    # Get the image size
    image_size = os.path.getsize(IMAGE_PATH)
    print(f"[*] Image size: {image_size} bytes")

    # Send the image size
    header = f"{image_size:08}".encode()
    client_socket.send(header)
    print(f"[*] Header sent: {header}")

    # Send the image
    with open(IMAGE_PATH, 'rb') as f:
        while True:
            bytes_read = f.read(4096)
            if not bytes_read:
                break
            client_socket.sendall(bytes_read)
    print("[*] Image sent.")

    # Move the servo to 90 degrees after sending
    move_servo(7.5)
    time.sleep(2)
    move_servo(0)
    time.sleep(0.5)
    move_servo(0)

    # Receive the response
    response = client_socket.recv(4096).decode()  # Increase buffer size for large responses
    print(f"[*] Response from server: {response}")

    # Handle the response
    if "Detected pills" in response:
        print_detected_pills(response)

    client_socket.close()

def move_servo(duty_cycle):
    print(f"[*] Moving servo to duty cycle: {duty_cycle}")
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)

def print_detected_pills(response):
    print("[*] Detected pills from the server:")
    lines = response.split("\n")
    for line in lines:
        if line.startswith("("):
            print(line)

def main():
    try:
        while True:
            if capture_image():
                weight = get_weight()
                send_image()
            time.sleep(15)  # Send an image every 10 seconds for testing
    except KeyboardInterrupt:
        print("[*] Cleaning up GPIO")
        GPIO.cleanup()

if _name_ == "_main_":
    main()