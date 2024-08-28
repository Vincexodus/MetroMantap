import cv2
import socket
import struct
import pickle
import spidev
import time

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

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width to 1280 for 720p
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height to 720 for 720p

# Setup socket for sending data
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = '192.168.1.100'  # Replace with PC's IP address
port = 9999  # Same port as server

client_socket.connect((server_ip, port))
print(f"Connected to {server_ip}:{port}")

try:
    while True:
        # Read the video frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Serialize frame
        data = pickle.dumps(frame)
        size = len(data)

        # Send frame size and frame data
        client_socket.sendall(struct.pack(">L", size) + data)

        # Read FSR values from ADC
        fsr1_value = read_adc(0)  # FSR connected to channel 0
        fsr2_value = read_adc(1)  # FSR connected to channel 1
        fsr3_value = read_adc(2)  # FSR connected to channel 2

        # Send sensor data over the socket
        sensor_data = f"FSR1:{fsr1_value},FSR2:{fsr2_value},FSR3:{fsr3_value}\n"
        client_socket.sendall(sensor_data.encode())

        # Optional: Add delay if needed to manage processing load
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    # Clean up
    cap.release()
    client_socket.close()
    spi.close()
