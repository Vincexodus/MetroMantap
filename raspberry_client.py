import os
import spidev
import time
import cv2
import socket
import struct
import pickle
import ssl
from datetime import datetime
from dotenv import load_dotenv

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000
load_dotenv('.env.local')

def read_adc(channel):
    if channel < 0 or channel > 7:
        raise ValueError("ADC channel must be between 0 and 7.")
    command = [1, (8 + channel) << 4, 0]
    adc_response = spi.xfer2(command)
    adc_value = ((adc_response[1] & 3) << 8) + adc_response[2]
    return adc_value

def setup_ssl_client(server_ip, server_port, ca_cert_path):
    context = ssl.create_default_context(cafile=ca_cert_path)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        secure_socket = context.wrap_socket(client_socket, server_hostname=server_ip)
        secure_socket.connect((server_ip, server_port))
        print(f"Client: Connected to server at {server_ip}:{server_port} over SSL")
        cert = secure_socket.getpeercert()
        print("Server certificate details:")
        print(f"Subject: {dict(x[0] for x in cert['subject'])}")
        print(f"Issuer: {dict(x[0] for x in cert['issuer'])}")
        return secure_socket
    except Exception as e:
        print(f"An error occurred while setting up the SSL connection: {e}")
        return None

def main():
    server_ip = os.getenv("SOCKET_HOST_IP")
    server_port = os.getenv("PORT")
    ca_cert_path = "assets/ssl/ca.crt"

    # Setup SSL connection
    secure_socket = setup_ssl_client(server_ip, int(server_port), ca_cert_path)
    if not secure_socket:
        print("Failed to establish SSL connection. Exiting.")
        return

    stream_from_webcam = False
    video1_path = "assets/train_cabin (2).mp4"
    video2_path = "assets/train_cabin (5).mp4"

    if stream_from_webcam:
        cap = cv2.VideoCapture(0)
        cap2 = cap
    else:
        cap = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)

    if not cap.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both video sources.")
        secure_socket.close()
        return

    desired_fps = 30
    cap.set(cv2.CAP_PROP_FPS, desired_fps)
    cap2.set(cv2.CAP_PROP_FPS, desired_fps)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

    train_status_c1 = {ord('1'): 'starting', ord('2'): 'running', ord('3'): 'slowing_down'}
    train_status_c2 = {ord('4'): 'starting', ord('5'): 'running', ord('6'): 'slowing_down'}
    c1_status = 'running'
    c2_status = 'running'

    cv2.namedWindow("ControlWindow", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret1, frame1 = cap.read()
            ret2, frame2 = cap2.read()

            if not stream_from_webcam:
                if not ret1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret1, frame1 = cap.read()
                if not ret2:
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret2, frame2 = cap2.read()
            else:
                if not ret1 or not ret2:
                    print("Client: Error: Could not read one or both frames.")
                    break

            key = cv2.waitKey(1) & 0xFF
            if key in train_status_c1:
                c1_status = train_status_c1[key]
            elif key in train_status_c2:
                c2_status = train_status_c2[key]
            elif key == ord('q'):
                break

            fsr1_value, fsr2_value = read_adc(0), read_adc(1)
            fsr3_value, fsr4_value = read_adc(2), read_adc(3)
            c1_fsr_values = [fsr1_value, fsr2_value]
            c2_fsr_values = [fsr3_value, fsr4_value]

            # Compress and resize frames
            frame1 = cv2.resize(frame1, (640, 360))
            frame2 = cv2.resize(frame2, (640, 360))
            _, encoded_img1 = cv2.imencode('.jpg', frame1, encode_param)
            _, encoded_img2 = cv2.imencode('.jpg', frame2, encode_param)

            send_timestamp = time.time()
            read_time = datetime.fromtimestamp(send_timestamp)
            data = {
                'send_timestamp': send_timestamp,
                'c1_fsr': c1_fsr_values,
                'c2_fsr': c2_fsr_values,
                'c1_status': c1_status,
                'c2_status': c2_status,
                'frame1': encoded_img1,
                'frame2': encoded_img2,
            }

            serialized_data = pickle.dumps(data)

            size = len(serialized_data)
            message = struct.pack("!L", size) + serialized_data

            secure_socket.sendall(message)

            print(f"{read_time}: C1 Status: {c1_status}, FSR Values: {c1_fsr_values}, C2 Status: {c2_status}, FSR Values: {c2_fsr_values}")
            
            time.sleep(0.03)

    except Exception as e:
        print(f"Client: An error occurred: {e}")

    finally:
        cap.release()
        cap2.release()
        secure_socket.close()
        spi.close()
        cv2.destroyAllWindows()
        print("Client: Camera released, secure socket closed, SPI connection closed.")

if __name__ == "__main__":
    main()