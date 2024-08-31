import os
import math
import numpy as np
import cv2
import socket
import struct
import pickle
from dotenv import load_dotenv
from ultralytics import YOLOv10
from object_classes import classNames
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

load_dotenv()

influxdb_url = os.getenv("INFLUXDB_URL")
influxdb_token = os.getenv("INFLUXDB_TOKEN")
influxdb_org = os.getenv("INFLUXDB_ORG")
influxdb_bucket = os.getenv("INFLUXDB_BUCKET")

# Initialize InfluxDB client
client = InfluxDBClient(url=influxdb_url, token=influxdb_token, org=influxdb_org)
write_api = client.write_api(write_options=SYNCHRONOUS)

host_ip = os.getenv("HOST_IP")
port = int(os.getenv("PORT"))

# Setup socket to receive video stream
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind((host_ip, port))
server_socket.listen(5)
print(f"Server: Listening on {host_ip}:{port}")

client_socket, addr = server_socket.accept()
print(f"Connection from: {addr}")

data = b""
payload_size = struct.calcsize("!L")

# Load the pretrained YOLOv10 model
yolo_model_name = os.getenv("YOLO_MODEL_NAME")
model = YOLOv10.from_pretrained(yolo_model_name)

def generate_color(index):
    """
    Generate a unique color for each index.
    :param index: Index or ID of the bounding box
    :return: A tuple representing the color in BGR format
    """
    np.random.seed(index)  # Seed the random number generator for reproducibility
    return tuple(np.random.randint(0, 256, 3).tolist())  # Generate a random color

try:
    while True:
        # Retrieve message size (4 bytes)
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

        # Retrieve the frame data based on the size
        while len(data) < msg_size:
            packet = client_socket.recv(4096)
            if not packet:
                print("Server: No more frame data received. Exiting loop.")
                break
            data += packet

        if len(data) < msg_size:
            print("Server: Incomplete frame data received. Needed: {}, Received: {}. Exiting loop.".format(msg_size, len(data)))
            break

        serialized_data = data[:msg_size]
        data = data[msg_size:]

        try:
            # Deserialize data using pickle
            received_data = pickle.loads(serialized_data)
            fsr_values = received_data['fsr']
            frame = received_data['frame']
            print(f"Server: FSR Values: {fsr_values}")
            
            # Decode frame back from JPEG
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            # Predict using YOLOv10 model
            results = model.predict(frame, stream=True)

            for r in results:
                boxes = r.boxes

                for i, box in enumerate(boxes):
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    color = generate_color(i)
                    # put box in cam
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100

                    # class name
                    cls = int(box.cls[0])
                    objectInfo = classNames[cls] + " " + str(confidence)

                    # append prediction details to frame
                    cv2.putText(frame, objectInfo, org=[x1, y1-5], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=2)

                    # Push bounding box data to InfluxDB
                    point = Point("passenger_detection") \
                        .tag("class", classNames[cls]) \
                        .field("confidence", confidence) \
                        .field("x1", x1) \
                        .field("y1", y1) \
                        .field("x2", x2) \
                        .field("y2", y2)
                    write_api.write(bucket=influxdb_bucket, org=influxdb_org, record=point)

            for sensor, value in fsr_values.items():
                point = Point("fsr_readings").tag("sensor", sensor).field("value", value)
                write_api.write(bucket=influxdb_bucket, org=influxdb_org, record=point)

        except Exception as e:
            print(f"Server: Deserialization error: {e}")
            break

        # Display the frame
        cv2.imshow('Video Stream with YOLOv10 Prediction', frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Server: Exiting on user command.")
            break

except KeyboardInterrupt:
    print("Server: Program interrupted by user.")

finally:
    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("Server: Sockets closed, resources released.")
