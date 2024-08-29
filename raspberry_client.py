import cv2
import socket
import struct
import pickle

# Setup socket to send video stream
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = '192.168.0.126'  # IP address of the server
server_port = 9999  # Port to connect to

client_socket.connect((server_ip, server_port))
print(f"Client: Connected to server at {server_ip}:{server_port}")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
desired_fps = 15  # Set a reasonable frame rate
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Compression parameters
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Quality scale (0-100)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Client: Error: Could not read frame.")
            break

        # Compress the frame using JPEG
        result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
        if not result:
            print("Client: Error: Could not encode frame.")
            break

        # Serialize the compressed frame
        data = pickle.dumps(encoded_img)
        size = len(data)
        message = struct.pack("!L", size) + data

        # Send the frame size and data to the server
        client_socket.sendall(message)

except Exception as e:
    print(f"Client: An error occurred: {e}")

finally:
    cap.release()
    client_socket.close()
    print("Client: Camera released, socket closed.")

