import cv2
import socket
import struct
import pickle

# Setup socket to receive video stream
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '0.0.0.0'  # Listen on all network interfaces
port = 9999  # Port to listen on

server_socket.bind((host_ip, port))
server_socket.listen(5)
print(f"Listening on {host_ip}:{port}")

client_socket, addr = server_socket.accept()
print(f"Connection from: {addr}")

# Create a file-like object to read the video stream
video_stream = client_socket.makefile('rb')

try:
    while True:
        # Read frame size
        size = struct.unpack(">L", client_socket.recv(4))[0]

        # Read frame data
        data = b''
        while len(data) < size:
            packet = client_socket.recv(4096)
            if not packet:
                break
            data += packet

        # Deserialize frame
        frame = pickle.loads(data)

        # Display frame
        cv2.imshow('Video', frame)

        # Read sensor data
        sensor_data = client_socket.recv(1024).decode()
        print(sensor_data)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()
