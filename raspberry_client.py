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

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Client: Error: Could not open webcam.")
    exit()

# Set the desired resolution (e.g., 640x480)
width = 640
height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
print(f"Client: Set webcam resolution to {width}x{height}")

# Set the desired FPS (frames per second)
desired_fps = 15
cap.set(cv2.CAP_PROP_FPS, desired_fps)
print(f"Client: Set webcam FPS to {desired_fps}")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Client: Error: Could not read frame.")
            break

        # Serialize the frame using pickle
        data = pickle.dumps(frame)
        # Pack the size of the frame (4 bytes) + frame data
        size = len(data)
        print(f"Client: Frame size: {size} bytes")
        message = struct.pack("!L", size) + data  # Use network byte order

        # Send the frame size and data to the server
        client_socket.sendall(message)

except Exception as e:
    print(f"Client: An error occurred: {e}")

finally:
    # Clean up
    cap.release()
    client_socket.close()
    print("Client: Camera released, socket closed.")
