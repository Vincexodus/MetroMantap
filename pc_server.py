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

# Calculate the angle between three keypoints
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # If the angle is greater than 180 degrees, correct it by subtracting it from 360.
    if angle > 180.0:
        angle = 360 - angle

    return angle

# Load environment variables
def load_env_variables():
    load_dotenv('.env.local')
    return {
        "influxdb_url": os.getenv("INFLUX_HOST"),
        "influxdb_token": os.getenv("INFLUX_TOKEN"),
        "influxdb_org": os.getenv("INFLUX_ORG"),
        "influxdb_bucket": os.getenv("TRAIN_BUCKET"),
        "host_ip": os.getenv("SOCKET_HOST_IP"),
        "port": int(os.getenv("PORT"))
    }

# Initialize InfluxDB client
def initialize_influxdb(influxdb_url, influxdb_token, influxdb_org):
    client = InfluxDBClient(url=influxdb_url, token=influxdb_token, org=influxdb_org)
    return client.write_api(write_options=SYNCHRONOUS)

# Setup socket to receive input data
def setup_socket(host_ip, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host_ip, port))
    server_socket.listen(5)
    client_socket, addr = server_socket.accept()
    return server_socket, client_socket, addr

def handle_incoming_data(client_socket, payload_size, data):
    while len(data) < payload_size:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet

    if len(data) < payload_size:
        return None, None

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("!L", packed_msg_size)[0]

    while len(data) < msg_size:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet

    if len(data) < msg_size:
        return None, None

    serialized_data = data[:msg_size]
    data = data[msg_size:]

    return pickle.loads(serialized_data), data

def process_frame(c1_frame, results):
    c1_people_count = 0
    leaving_person_count = 0

    for r in results:
        boxes = r.boxes
        keypoints_data = r.keypoints.data

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if classNames[cls] == "person":
                c1_people_count += 1
                keypoints = keypoints_data[i]
                if keypoints.shape[0] > 0:
                    angle = calculate_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
                    status = 'Sitting' if angle is not None and angle < 110 else 'Standing'
                    color = (0, 255, 0) if status == 'Sitting' else (0, 0, 255)
                    cv2.rectangle(c1_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(c1_frame, f"{status}", (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                object_info = f"{classNames[cls]} {confidence}"
                cv2.putText(c1_frame, object_info, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return c1_frame, c1_people_count, leaving_person_count

def write_to_influx(write_api, bucket, org, c1_fsr_values, c2_fsr_values, c1_status, c2_status, c1_people_count, c2_people_count, last_write_time, current_time):
    if current_time - last_write_time >= 1:
        sensor_point = (Point("sensor_data")
                        .field("fsr1", c1_fsr_values[0])
                        .field("fsr2", c1_fsr_values[1])
                        .field("fsr3", c2_fsr_values[0])
                        .field("fsr4", c2_fsr_values[1])
                        .field("c1_status", c1_status)
                        .field("c2_status", c2_status)
                        .field("c1_person", c1_people_count)
                        .field("c2_person", c1_people_count))
        write_api.write(bucket=bucket, org=org, record=sensor_point)
        last_write_time = current_time
    return last_write_time

def process_single_frame(frame, model):
    start_inference_time = time.time()
    results = model(frame, show=False, stream=True, verbose=False)
    yolo_latency = time.time() - start_inference_time

    frame, people_count, leaving_person_count = process_frame(frame, results)

    return frame, people_count, yolo_latency

def annotate_frame(frame, status_text, yolo_latency, fps):
    cv2.putText(frame, status_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"Inference Time: {yolo_latency:.2f} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def main():
    config = load_env_variables()
    write_api = initialize_influxdb(config['influxdb_url'], config['influxdb_token'], config['influxdb_org'])
    model = YOLO("assets/YOLOv8m-pose.pt")
    server_socket, client_socket, addr = setup_socket(config['host_ip'], config['port'])
    print(f"Connection from: {addr}")

    payload_size = struct.calcsize("!L")
    data = b""
    last_write_time = time.time()

    try:
        while True:
            received_data, data = handle_incoming_data(client_socket, payload_size, data)
            if not received_data:
                break

            start_time = time.time()

            c1_status = received_data['c1_status']
            c2_status = received_data['c2_status']
            c1_frame = cv2.imdecode(received_data['frame1'], cv2.IMREAD_COLOR)
            c2_frame = cv2.imdecode(received_data['frame2'], cv2.IMREAD_COLOR)

            c1_fsr_values = received_data['c1_fsr']
            c2_fsr_values = received_data['c2_fsr']

            # Process both frames
            c1_frame, c1_people_count, c1_yolo_latency = process_single_frame(c1_frame, model)
            c2_frame, c2_people_count, c2_yolo_latency = process_single_frame(c2_frame, model)

            # Calculate frame processing time and FPS
            frame_time = time.time() - start_time
            fps = 1 / frame_time if frame_time > 0 else 0

            # Add status and inference time to both frames
            annotate_frame(c1_frame, f"C1-{c1_status}", c1_yolo_latency, fps)
            annotate_frame(c2_frame, f"C2-{c2_status}", c2_yolo_latency, fps)

            current_time = time.time()
            last_write_time = write_to_influx(
                write_api,
                config['influxdb_bucket'],
                config['influxdb_org'],
                c1_fsr_values,
                c2_fsr_values,
                c1_status,
                c2_status,
                c1_people_count,
                c2_people_count,
                last_write_time,
                current_time
            )

            # Display both frames
            cv2.imshow('C1 Video Stream with Prediction', c1_frame)
            cv2.imshow('C2 Video Stream with Prediction', c2_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        client_socket.close()
        server_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
