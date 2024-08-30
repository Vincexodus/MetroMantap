import spidev
import time
import cv2
import socket
import struct
import pickle

# Initialize SPI for ADC
spi = spidev.SpiDev()
spi.open(0, 0)  # Open SPI bus 0, device (CS) 0
spi.max_speed_hz = 1350000

def read_adc(channel):
    """
    Function to read SPI data from MCP3008 ADC.
    :param channel: ADC channel (0-7)
    :return: Digital value read from the channel (0-1023 for 10-bit ADC)
    """
    if channel < 0 or channel > 7:
        raise ValueError("ADC channel must be between 0 and 7.")

    # MCP3008 read command: Start bit, Single/Diff bit, Channel bits
    command = [1, (8 + channel) << 4, 0]
    adc_response = spi.xfer2(command)
    # Combine response bytes into a single value
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

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
desired_fps = 15  # Set a reasonable frame rate
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Compression parameters
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Quality scale (0-100)

try:
    while True:
        # Read FSR values from ADC
        fsr1_value = read_adc(0)  # Channel 0 for FSR 1
        fsr2_value = read_adc(1)  # Channel 1 for FSR 2
        fsr3_value = read_adc(2)  # Channel 2 for FSR 3
        fsr_values = [fsr1_value, fsr2_value, fsr3_value]

        # Capture video frame
        ret, frame = cap.read()
        if not ret:
            print("Client: Error: Could not read frame.")
            break

        # Compress the frame using JPEG
        result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
        if not result:
            print("Client: Error: Could not encode frame.")
            break

        # Combine FSR data and encoded frame
        data = {
            'fsr': fsr_values,
            'frame': encoded_img
        }

        # Serialize the combined data
        serialized_data = pickle.dumps(data)
        size = len(serialized_data)
        message = struct.pack("!L", size) + serialized_data

        # Send the frame size and data to the server
        client_socket.sendall(message)

        # Optional: Delay to control the rate of sending data
        time.sleep(0.1)

except Exception as e:
    print(f"Client: An error occurred: {e}")

finally:
    cap.release()
    client_socket.close()
    spi.close()
    print("Client: Camera released, socket closed, SPI connection closed.")
