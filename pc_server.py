import os
import time
import math
import socket
import struct
import pickle
import numpy as np
import cv2
import speedtest
from dotenv import load_dotenv
from ultralytics import YOLO
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from object_classes import classNames
from concurrent.futures import ThreadPoolExecutor

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

def process_frame(c1_frame, train_status, results):
    people_count = 0
    standing_passengers_count = 0

    for r in results:
        boxes = r.boxes
        keypoints_data = r.keypoints.data

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if classNames[cls] == "person":
                people_count += 1
                keypoints = keypoints_data[i]
                if keypoints.shape[0] > 0:
                    angle = calculate_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
                    status = 'Sitting' if angle is not None and angle < 110 else 'Standing'
                    color = (0, 255, 0) if status == 'Sitting' else (0, 0, 255)
                    
                    if status == 'Standing':
                        standing_passengers_count += 1
                        cv2.rectangle(c1_frame, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(c1_frame, f"{status}", (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        object_info = f"{classNames[cls]} {confidence}"
                        cv2.putText(c1_frame, object_info, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # assume passengers leaving if stand up when train slowing down
    if train_status == 'slowing_down':
        people_count -= standing_passengers_count

    return c1_frame, people_count

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

def process_single_frame(frame, train_status, model):
    start_inference_time = time.perf_counter()
    results = model(frame, show=False, stream=True, verbose=False)
    yolo_latency = time.perf_counter() - start_inference_time
    frame, people_count = process_frame(frame, train_status, results)

    return frame, people_count, yolo_latency

def annotate_frame(frame, status_text, one_way_latency, yolo_latency, fps):
    # Convert yolo_latency from seconds to milliseconds
    yolo_latency_ms = yolo_latency * 1000
    
    cv2.putText(frame, status_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"Input Latency: {one_way_latency:.4f} s", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"YOLO Inference Time: {yolo_latency_ms:.4f} ms", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"FPS: {fps:.2f}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

# Add new function to handle video capture
def process_video_file(video_path, train_status, model, speedtest_interval):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break

        # Process the frame with YOLO
        frame, people_count, yolo_latency = process_single_frame(frame, train_status, model)

        # Get the FPS from the video file
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Add annotations to the frame
        annotate_frame(frame, f"Status: {train_status}", one_way_latency=0, yolo_latency=yolo_latency, fps=fps)

        # Yield the frame and people count to the caller
        yield frame, people_count
    
    cap.release()
    cv2.destroyAllWindows()

def stream_video(index, video_path, train_status, model, write_api, config, last_write_time, last_speedtest_time, speedtest_interval):
    c1_people_count = 0
    c2_people_count = 0

    for frame, people_count in process_video_file(video_path, train_status, model, speedtest_interval):
        current_time = time.time()

        cv2.imshow(f'Video Stream {index + 1} - {video_path}', frame)

        if index == 0:
            c1_people_count = people_count
        else:
            c2_people_count = people_count

        if current_time - last_speedtest_time >= speedtest_interval:
            download_speed, upload_speed, ping = run_speedtest()
            last_speedtest_time = current_time
        else:
            download_speed = upload_speed = ping = None

        last_write_time = write_to_influx(
            write_api,
            config['influxdb_bucket'],
            config['influxdb_org'],
            one_way_latency=0.0, 
            c1_fsr_values=[0, 0],
            c2_fsr_values=[0, 0],
            c1_status=train_status,
            c2_status=train_status,
            c1_people_count=c1_people_count,
            c2_people_count=c2_people_count,
            download_speed=download_speed,
            upload_speed=upload_speed,
            ping=ping,
            last_write_time=last_write_time,
            current_time=current_time
        )
        # Check for quit command
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    config = load_env_variables()
    write_api = initialize_influxdb(config['influxdb_url'], config['influxdb_token'], config['influxdb_org'])
    model = YOLO("assets/YOLOv8m-pose.pt")

    stream_from_video = True
    speedtest_interval = 60

    if stream_from_video:
        last_write_time = time.time()
        last_speedtest_time = time.time()
        # Define video paths and statuses for two streams
        video_1 = (0, "assets/train_cabin (1).mp4", "moving")
        video_2 = (1, "assets/train_cabin (2).mp4", "stopped")

        # Use ThreadPoolExecutor to stream videos concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(stream_video, video_1[0], video_1[1], video_1[2], model, write_api, config, last_write_time, last_speedtest_time, speedtest_interval),
                executor.submit(stream_video, video_2[0], video_2[1], video_2[2], model, write_api, config, last_write_time, last_speedtest_time, speedtest_interval)
            ]

            # Wait for both video streams to finish
            for future in futures:
                future.result()
    else:
        server_socket, client_socket, addr = setup_socket(config['host_ip'], config['port'])
        print(f"Connection from: {addr}")

        payload_size = struct.calcsize("!L")
        data = b""
        last_write_time = time.time()
        last_speedtest_time = time.time()

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

                c1_fsr_values = received_data['c1_fsr']
                c2_fsr_values = received_data['c2_fsr']

                # Capture reception time
                one_way_latency = (receive_time - send_time) / 2

                # Process both frames
                c1_frame, c1_people_count, c1_yolo_latency = process_single_frame(c1_frame, c1_status, model)
                c2_frame, c2_people_count, c2_yolo_latency = process_single_frame(c2_frame, c2_status, model)

                # Calculate frame processing time and FPS
                frame_time = time.time() - start_time
                fps = 1 / frame_time if frame_time > 0 else 0

                # Add status and inference time to both frames
                annotate_frame(c1_frame, f"C1-{c1_status}", one_way_latency, c1_yolo_latency, fps)
                annotate_frame(c2_frame, f"C2-{c2_status}", one_way_latency, c2_yolo_latency, fps)

                # Run speed test periodically
                current_time = time.time()
                if current_time - last_speedtest_time >= speedtest_interval:
                    download_speed, upload_speed, ping = run_speedtest()
                    last_speedtest_time = current_time
                else:
                    download_speed = upload_speed = ping = None

                # Write data to InfluxDB
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
      
                # Display both frames
                cv2.imshow('C1 Video Stream with Prediction', c1_frame)
                cv2.imshow('C2 Video Stream with Prediction', c2_frame)

                # Check for quit command
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        finally:
            client_socket.close()
            server_socket.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
