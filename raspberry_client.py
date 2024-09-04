import spidev
import time
import cv2
import socket
import struct
import pickle
import keyboard  # To detect keyboard inputs

# Initialize SPI for ADC
spi = spidev.SpiDev()
spi.open(0, 0)  # Open SPI bus 0, device (CS) 0
spi.max_speed_hz = 1350000

def read_adc(channel):
    if channel < 0 or channel > 7:
        raise ValueError("ADC channel must be between 0 and 7.")
    command = [1, (8 + channel) << 4, 0]
    adc_response = spi.xfer2(command)
    adc_value = ((adc_response[1] & 3) << 8) + adc_response[2]
    return adc_value

# Setup socket to send video stream
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = '192.168.0.126'  # IP address of the server
server_port = 9999  # Port to connect to

client_socket.connect((server_ip, server_port))
print(f"Client: Connected to server at {server_ip}:{server_port}")

# Initialize the webcam
cap = cv2.VideoCapture(0)
desired_fps = 15  # Set a reasonable frame rate
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Compression parameters
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Quality scale (0-100)

# Dictionary to map keys to train statuses for both cabins
train_status_c1 = {
    '1': 'starting',
    '2': 'running',
    '3': 'stopping'
}

train_status_c2 = {
    '4': 'starting',
    '5': 'running',
    '6': 'stopping'
}

c1_status = 'running'  # Default status for cabin 1
c2_status = 'running'  # Default status for cabin 2

try:
    while True:
        # Check for keyboard input to update cabin 1 status
        for key in train_status_c1.keys():
            if keyboard.is_pressed(key):
                c1_status = train_status_c1[key]
                print(f"Client: Cabin 1 status changed to {c1_status}")
                break

        # Check for keyboard input to update cabin 2 status
        for key in train_status_c2.keys():
            if keyboard.is_pressed(key):
                c2_status = train_status_c2[key]
                print(f"Client: Cabin 2 status changed to {c2_status}")
                break

        # Read FSR values from ADC
        fsr1_value = read_adc(0)
        fsr2_value = read_adc(1)
        fsr3_value = read_adc(2)
        c1_fsr_values = [fsr1_value, fsr2_value]
        c2_fsr_values = [fsr3_value, 0]

        # Capture first video frame
        ret1, frame1 = cap.read()
        if not ret1:
            print("Client: Error: Could not read first frame.")
            break

        # Compress the frame using JPEG
        result1, encoded_img1 = cv2.imencode('.jpg', frame1, encode_param)

        # Combine FSR data, train statuses, and encoded frames
        data = {
            'c1_fsr': c1_fsr_values,
            'c2_fsr': c2_fsr_values,
            'c1_status': c1_status,
            'c2_status': c2_status,
            'frame1': encoded_img1,
            'frame2': encoded_img1,  # Use the same frame for now
        }

        # Serialize the combined data
        serialized_data = pickle.dumps(data)
        size = len(serialized_data)
        message = struct.pack("!L", size) + serialized_data

        # Send the frame size and data to the server
        client_socket.sendall(message)

        print(f"Client: C1 Status: {c1_status}, FSR Values: {c1_fsr_values}, C2 Status: {c2_status}, FSR Values: {c2_fsr_values},")
        # Optional: Delay to control the rate of sending data
        time.sleep(0.5)

except Exception as e:
    print(f"Client: An error occurred: {e}")

finally:
    cap.release()
    client_socket.close()
    spi.close()
    print("Client: Camera released, socket closed, SPI connection closed.")
