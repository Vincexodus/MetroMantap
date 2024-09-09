import os
import time
import math
import socket
import struct
import pickle
import numpy as np
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from object_classes import classNames

# Load environment variables
load_dotenv('.env.local')

# Configuration variables
influxdb_url = os.getenv("INFLUXDB_URL")
influxdb_token = os.getenv("INFLUXDB_TOKEN")
influxdb_org = os.getenv("INFLUXDB_ORG")
influxdb_bucket = os.getenv("INFLUXDB_BUCKET")
host_ip = os.getenv("HOST_IP")
port = int(os.getenv("PORT"))
yolo_model_name = os.getenv("YOLO_MODEL_NAME")

# Initialize InfluxDB client
client = InfluxDBClient(url=influxdb_url, token=influxdb_token, org=influxdb_org)
write_api = client.write_api(write_options=SYNCHRONOUS)

model = YOLO("assets/YOLOv8m-pose.pt")

# Setup socket to receive video stream
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host_ip, port))
server_socket.listen(5)
print(f"Server: Listening on {host_ip}:{port}")

client_socket, addr = server_socket.accept()
print(f"Connection from: {addr}")

# Constants
payload_size = struct.calcsize("!L")
data = b""
last_write_time = time.time()
roi_x1, roi_y1 = 5, 20  # Top-left corner of ROI (region of interest)
roi_x2, roi_y2 = 500, 500  # Bottom-right corner of ROI

# Function definitions
def generate_color(index):
    """Generate a unique color for each index."""
    np.random.seed(index)  # Seed for reproducibility
    return tuple(np.random.randint(0, 256, 3).tolist())  # Generate a random color

def calculate_overlap_area(x1, y1, x2, y2, roi_x1, roi_y1, roi_x2, roi_y2):
    """Calculate the overlapping area between a bounding box and a region of interest (ROI)."""
    inter_x1 = max(x1, roi_x1)
    inter_y1 = max(y1, roi_y1)
    inter_x2 = min(x2, roi_x2)
    inter_y2 = min(y2, roi_y2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    return inter_width * inter_height

# Define a function to calculate the angle between three keypoints
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # Last point
    
    # Calculate the angle in radians and convert it to degrees.
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # If the angle is greater than 180 degrees, correct it by subtracting it from 360.
    if angle > 180.0:
        angle = 360 - angle

    return angle

# Main loop
try:
    while True:
        # Retrieve message size
        while len(data) < payload_size:
            packet = client_socket.recv(4096)  # 4KB buffer size
            if not packet:
                print("Server: No data received. Closing connection.")
                break
            data += packet

        if len(data) < payload_size:
            print("Server: Incomplete data received for message size. Exiting loop.")
            break

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("!L", packed_msg_size)[0]

        # Retrieve frame data
        while len(data) < msg_size:
            packet = client_socket.recv(4096)
            if not packet:
                print("Server: No more frame data received. Exiting loop.")
                break
            data += packet

        if len(data) < msg_size:
            print(f"Server: Incomplete frame data received. Needed: {msg_size}, Received: {len(data)}. Exiting loop.")
            break

        serialized_data = data[:msg_size]
        data = data[msg_size:]

        try:
            # Deserialize data
            received_data = pickle.loads(serialized_data)
            c1_status = received_data['c1_status']
            c2_status = received_data['c2_status']
            c1_frame = received_data['frame1']
            c2_frame = received_data['frame2']
            c1_fsr_values = received_data['c1_fsr']
            c2_fsr_values = received_data['c2_fsr']

            c1_frame = cv2.imdecode(c1_frame, cv2.IMREAD_COLOR)

            # Display frame with ROI and status
            cv2.rectangle(c1_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), generate_color(5), 2)
            cv2.putText(c1_frame, f"C1-{c1_status}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, generate_color(5), 2)
            
            # Predict using YOLOv8 model
            results = model(c1_frame, show=False, stream=True, verbose=False)

            # Initialize counters
            c1_people_count = 0
            leaving_person_count = 0

            for r in results:
                boxes = r.boxes
                keypoints_data = r.keypoints.data
                
                for i, box in enumerate(boxes):
                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer

                    # Confidence and class name
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    
                    # Only process the "person" class
                    if classNames[cls] == "person":
                        c1_people_count += 1

                        # Extract keypoints for pose detection
                        keypoints = keypoints_data[i]  # Get keypoints for this person
                        if keypoints.shape[0] > 0:
                            # Calculate angle between head (11), hips (13), and knees (15)
                            angle = calculate_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])

                            # Classify as sitting or standing
                            status = 'Sitting' if angle is not None and angle < 110 else 'Standing'

                            # Generate the bounding box color only once
                            color = (0, 255, 0) if status == 'Sitting' else (0, 0, 255)

                            # Draw bounding box and status text
                            cv2.rectangle(c1_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(c1_frame, f"{status}", (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Calculate the bounding box and overlap areas
                        bbox_area = (x2 - x1) * (y2 - y1)
                        overlap_area = calculate_overlap_area(x1, y1, x2, y2, roi_x1, roi_y1, roi_x2, roi_y2)

                        # Check if more than half of the bounding box overlaps with the ROI
                        if overlap_area > 0.5 * bbox_area:
                            leaving_person_count += 1

                        # Draw object info text (class + confidence) once
                        object_info = f"{classNames[cls]} {confidence}"
                        cv2.putText(c1_frame, object_info, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Write sensor and people count data every second
            current_time = time.time()
            if current_time - last_write_time >= 1:
                sensor_point = (Point("sensor_data")
                                .field("fsr1", c1_fsr_values[0])
                                .field("fsr2", c1_fsr_values[1])
                                .field("fsr3", c2_fsr_values[0])
                                .field("fsr4", c2_fsr_values[1])
                                .field("c1_person", c1_people_count)
                                .field("c2_person", c1_people_count))
                write_api.write(bucket=influxdb_bucket, org=influxdb_org, record=sensor_point)
                last_write_time = current_time
                print(f"Server: C1-{c1_status} [{c1_fsr_values[0]}, {c1_fsr_values[1]}] person: {c1_people_count}, C2-{c2_status} [{c2_fsr_values[0]}, {c2_fsr_values[1]}], person: {c1_people_count}")
                print(f"Leaving person: {leaving_person_count}")

        except Exception as e:
            print(f"Server: Deserialization error: {e}")
            break

        cv2.imshow('Video Stream with Prediction', c1_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Server: Exiting on user command.")
            break

except KeyboardInterrupt:
    print("Server: Program interrupted by user.")

finally:
    # Cleanup resources
    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("Server: Sockets closed, resources released.")