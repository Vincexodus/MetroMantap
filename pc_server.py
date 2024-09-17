import os
import time
import socket
import struct
import pickle
import numpy as np
import cv2
import speedtest
import torch
from dotenv import load_dotenv
from ultralytics import YOLO
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from object_classes import classNames

def calculate_posture_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # If the angle is greater than 180 degrees, correct it by subtracting it from 360.
    if angle > 180.0:
        angle = 360 - angle

    return angle

def detect_face_direction(keypoints):
    if len(keypoints) < 5:
        return "Unknown"

    nose_x, nose_y = keypoints[0][:2]
    left_eye_x, left_eye_y = keypoints[1][:2]
    right_eye_x, right_eye_y = keypoints[2][:2]
    left_ear_x, left_ear_y = keypoints[3][:2]
    right_ear_x, right_ear_y = keypoints[4][:2]

    # Calculate horizontal face direction
    if nose_x < left_eye_x and nose_x < left_ear_x:
        return "Left"
    elif nose_x > right_eye_x and nose_x > right_ear_x:
        return "Right"
    
    # Calculate vertical face direction
    if nose_y < min(left_eye_y, right_eye_y):
        return "Up"
    elif nose_y > max(left_eye_y, right_eye_y):
        return "Down"

def run_speedtest():
    st = speedtest.Speedtest()
    st.get_best_server()
    download_speed = st.download() / 1_000_000  # Convert from bits/s to Mbits/s
    upload_speed = st.upload() / 1_000_000  # Convert from bits/s to Mbits/s
    ping = st.results.ping
    return download_speed, upload_speed, ping

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

def write_to_influx(write_api, bucket, org, one_way_latency, c1_fsr_values, c2_fsr_values, c1_status, c2_status, c1_people_count, c2_people_count, download_speed, upload_speed, ping, last_write_time, current_time):
    if current_time - last_write_time >= 1:
        sensor_point = (Point("sensor_data")
                        .field("one_way_latency", one_way_latency)
                        .field("fsr1", c1_fsr_values[0])
                        .field("fsr2", c1_fsr_values[1])
                        .field("fsr3", c2_fsr_values[0])
                        .field("fsr4", c2_fsr_values[1])
                        .field("c1_status", c1_status)
                        .field("c2_status", c2_status)
                        .field("c1_person", c1_people_count)
                        .field("c2_person", c2_people_count)
                        .field("download_speed", download_speed)
                        .field("upload_speed", upload_speed)
                        .field("ping", ping))
        write_api.write(bucket=bucket, org=org, record=sensor_point)
        last_write_time = current_time
    return last_write_time

def process_single_frame(frame, model):
    start_inference_time = time.perf_counter()
    
    results = model(frame, show=False, stream=True, verbose=False)

    people_count = 0
    frame_height = frame.shape[0]  # Get the frame height to compare head position

    for r in results:
        boxes = r.boxes
        keypoints_data = r.keypoints.data

        for i, box in enumerate(boxes):
            confidence = box.conf[0]
            if confidence < 0.4:  # ignore low confidence
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls = int(box.cls[0])

            if classNames[cls] == "person":
                people_count += 1
                keypoints = keypoints_data[i].cpu().numpy()  # Move keypoints tensor to CPU

                if keypoints.shape[0] > 0:
                    # Calculate angle between keypoints
                    angle = calculate_posture_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])

                    head_y = keypoints[0][1]  # y-coordinate of the head keypoint
                    if angle > 130 and head_y < frame_height / 2:
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(frame, "Standing", (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        object_info = f"{classNames[cls]} {confidence:.2f}"
                        cv2.putText(frame, object_info, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame, people_count, time.perf_counter() - start_inference_time

def annotate_frame(index, frame, status, people_count, one_way_latency, yolo_latency, fps):
    cv2.putText(frame, f"C{index+1}:{status}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"Passengers:{people_count}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"Input Latency: {one_way_latency:.4f} s", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"YOLO Inference Time: {yolo_latency:.4f} s", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"FPS: {fps:.2f}", (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def main():
    # Use GPU for prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("assets/YOLOv8n-pose.pt")
    model = model.to(device)

    config = load_env_variables()
    write_api = initialize_influxdb(config['influxdb_url'], config['influxdb_token'], config['influxdb_org'])
    
    server_socket, client_socket, addr = setup_socket(config['host_ip'], config['port'])
    print(f"Connection from: {addr}")

    payload_size = struct.calcsize("!L")
    data = b""
    last_write_time = time.time()
    last_speedtest_time = time.time()
    speedtest_interval = 60  # Interval to run speed test

    try:
        while True:
            received_data, data = handle_incoming_data(client_socket, payload_size, data)
            if not received_data:
                break

            start_time = time.time()
            receive_time = time.time()

            send_time = received_data['send_timestamp']
            c1_status = received_data['c1_status']
            c2_status = received_data['c2_status']
            c1_frame = cv2.imdecode(received_data['frame1'], cv2.IMREAD_COLOR)
            c2_frame = cv2.imdecode(received_data['frame2'], cv2.IMREAD_COLOR)

            # Resize to half of the original size
            original_height_c1, original_width_c1 = c1_frame.shape[:2]
            original_height_c2, original_width_c2 = c2_frame.shape[:2]

            c1_frame = cv2.resize(c1_frame, (original_width_c1 // 2, original_height_c1 // 2))
            c2_frame = cv2.resize(c2_frame, (original_width_c2 // 2, original_height_c2 // 2))

            c1_fsr_values = received_data['c1_fsr']
            c2_fsr_values = received_data['c2_fsr']

            # Capture reception time
            one_way_latency = (receive_time - send_time) / 2

            # Process both frames
            c1_frame, c1_people_count, c1_yolo_latency = process_single_frame(c1_frame, model)
            c2_frame, c2_people_count, c2_yolo_latency = process_single_frame(c2_frame, model)

            # Calculate frame processing time and FPS
            frame_time = time.time() - start_time
            fps = 1 / frame_time if frame_time > 0 else 0

            # Add status and inference time to both frames
            annotate_frame(0, c1_frame, c1_status, c1_people_count, one_way_latency, c1_yolo_latency, fps)
            annotate_frame(1, c2_frame, c2_status, c2_people_count, one_way_latency, c2_yolo_latency, fps)

            # Display both frames
            cv2.imshow('Cabin 1 Video Stream with Prediction', c1_frame)
            cv2.imshow('Cabin 2 Video Stream with Prediction', c2_frame)

            # Run speed test periodically
            current_time = time.time()
            if current_time - last_speedtest_time >= speedtest_interval:
                download_speed, upload_speed, ping = run_speedtest()
                last_speedtest_time = current_time
            else:
                download_speed = upload_speed = ping = None

            last_write_time = write_to_influx(
                write_api,
                config['influxdb_bucket'],
                config['influxdb_org'],
                one_way_latency,
                c1_fsr_values,
                c2_fsr_values,
                c1_status,
                c2_status,
                c1_people_count,
                c2_people_count,
                download_speed,
                upload_speed,
                ping,
                last_write_time,
                current_time
            )

            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

    finally:
        client_socket.close()
        server_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()