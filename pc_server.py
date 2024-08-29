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
print(f"Server: Listening on {host_ip}:{port}")

client_socket, addr = server_socket.accept()
print(f"Server: Connection from {addr}")

data = b""
payload_size = struct.calcsize("!L")

try:
    while True:
        # Retrieve message size (4 bytes)
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet:
                print("Server: No data received. Closing connection.")
                break
            data += packet

        if len(data) < payload_size:
            print("Server: Incomplete data for message size.")
            break

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("!L", packed_msg_size)[0]

        # Retrieve the frame data based on the size
        while len(data) < msg_size:
            packet = client_socket.recv(4096)
            if not packet:
                print("Server: No more frame data received. Closing connection.")
                break
            data += packet

        if len(data) < msg_size:
            print(f"Server: Incomplete frame data. Needed: {msg_size}, Received: {len(data)}")
            break

        frame_data = data[:msg_size]
        data = data[msg_size:]

        try:
            # Deserialize frame using pickle
            encoded_img = pickle.loads(frame_data)
            # Decode the JPEG image
            frame = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            if frame is None:
                print("Server: Error: Decoded frame is None.")
                break
        except Exception as e:
            print(f"Server: Deserialization error: {e}")
            break

        # Display the frame
        cv2.imshow('Video Stream', frame)

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
