# Train Density Monitor - Real Time System (RTS)

Real-Time Crowd Management System for train stations using sensor technology, AI-driven analytics, and real-time data processing to improve passenger flow, reduce congestion, and optimize train operations and scheduling

### To-Do List

| Task                                      | Status                  |
| ----------------------------------------- | ----------------------- |
| Socket setup & Raspberry PI wiring        | ✅ Done                 |
| PC-Raspberry Pi socket communication      | ✅ Done                 |
| Combine camera feed + sensor reading      | ✅ Done                 |
| YOLOv10 Prediction & feed display         | ✅ Done                 |
| Integration with InfluxDB & Grafana       | ✅ Done                 |
| Dashboard Display on Grafana              | ✅ Done, Pending Review |
| FPS & Latency Display on PC               | 🔨 Work in Progress     |
| Fix socket communication high latency     | 🔨 Work in Progress     |
| Security measure for socket communication | ❌ Not started          |
| Demo Scene Setup                          | ❌ Not started          |
| Revamp Architecture Diagram               | ❌ Not started          |
| Revamp Poster Design                      | ❌ Not started          |

### Server Setup - Central Processing (PC)

1. Navigate to [influxData cloud](https://us-east-1-1.aws.cloud2.influxdata.com) to create a bucket called `Main`

2. Under API Tokens, create `All Access API Token` & copy the token

3. Download [Grafana installer](https://grafana.com/grafana/download?platform=windows)

4. Start grafana by running `grafana-server.exe` in `bin` directory

5. Open browser with address `http://localhost:3000/`, login with `admin` `admin`

6. Under `Home > Connections > Data sources`, add influxDB as data source

7. influxDB data source config:

   | Param          | Value                                           |
   | -------------- | ----------------------------------------------- |
   | Query Language | `Flux`                                          |
   | URL            | `https://us-east-1-1.aws.cloud2.influxdata.com` |
   | Organization   | `NextGen Hackathon`                             |
   | Token          | `<ALL_ACCESS_API_TOKEN>`                        |

8. Under `Home > Dashboards`, add visualization for new dashboard

9. Paste code example (written in Flux) to display FSR sensor readings for past 3 hours

   ```bash
    from(bucket: "Main")
      |> range(start: -3h)
      |> filter(fn: (r) => r._measurement == "sensor_data")
      |> filter(fn: (r) => r._field == "fsr1" or r._field == "fsr2" or r._field == "fsr3")
   ```

8. Clone the repo

   ```bash
   git clone https://github.com/Vincexodus/Train-Density-Monitor-RTS.git && cd .\Train-Density-Monitor-RTS\
   ```

9. Duplicate `.env.example`, rename duplicated file to `.env.local` and replace the values accordingly

10. Create virtual python environment and activate it

    ```bash
    virtualenv venv
    venv\Scripts\activate
    ```

10. Download necessary python modules

    ```bash
    pip install -r requirements.txt
    ```

11. Run server script

    ```bash
    py pc_server.py
    ```

### Client Setup - Raspberry Pi 4

1. Clone the repo and install python modules

   ```bash
   git clone https://github.com/Vincexodus/Train-Density-Monitor-RTS.git && cd .\Train-Density-Monitor-RTS\
   sudo pip install opencv-python spidev
   ```

2. Enable SPI on the Raspberry Pi

   ```bash
    sudo raspi-config
   ```

3. Navigate to Interfacing Options > SPI > Enable.

4. Run client script`raspberry_client.py` in Thonny IDE
