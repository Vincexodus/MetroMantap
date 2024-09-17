# Train Density Monitor - Real Time System (RTS)

<h1 align="center" style="display: flex; justify-content: center; align-items: center;">
  <img src="/assets/footage_small.gif" style="width:100%;"/>
  <img src="/assets/dashboard_small.gif" style="width:57.5%;"/>
</h1>

<p align="center">
  <i align="center">Real-Time Crowd Management System for train stations using sensor technology, AI-driven analytics, and real-time data processing to improve passenger flow, reduce congestion, and optimize train operations and scheduling</i>
</p>

<!-- ### To-Do List

| Task                                      | Status                 |
| ----------------------------------------- | ---------------------- |
| Batch Size Prediction, Async I/O, Vetorize function, Threading, Stream Buffer | ✨ Enhancements                  |
| Security measure for socket communication | ❌ Not needed yet      | -->

### Server Setup - Central Processing (PC)

1. Install [InfluxDB](https://docs.influxdata.com/influxdb/v2/install/?t=Windows#download-and-install-influxdb-v2) and extract the zip to `C:\Program Files\InfluxData\`, rename and move all files to parent folder

2. Start InfluxDB

   ```bash
   cd 'C:\Program Files\InfluxData' && ./influxd
   ```

3. Grant network access, `InfluxDB UI` can be viewed at `http://localhost:8086`

4. In `InfluxDB UI`, setup user account with org `NextGen Hackathon` to obtain `All Access API Token`, save the token

5. Under `Dashboard`, `Create Dashboard > Add a Template`, paste this [URL](https://raw.githubusercontent.com/Vincexodus/Train-Density-Monitor-RTS/main/assets/train_density_monitor_dashboard.json)

6. Open new terminal, clone the repo and navigate to root directory

   ```bash
   git clone https://github.com/Vincexodus/Train-Density-Monitor-RTS.git && cd .\Train-Density-Monitor-RTS\
   ```

7. Duplicate `.env.example` and rename the duplicated file to `.env.local`, replace the values accordingly

8. Create virtual python environment and activate it

   ```bash
   virtualenv venv
   venv\Scripts\activate
   ```

9. Download required python modules

   ```bash
   pip install -r requirements.txt
   ```

10. Run server script

    ```bash
    py pc_server.py
    ```

### Client Setup - Raspberry Pi 4

1. Install telegraf

   ```bash
    wget -q https://repos.influxdata.com/influxdata-archive_compat.key
    sudo apt-get update && sudo apt-get install telegraf
   ```

2. Add environment variables for Telegraf

   ```bash
    export INFLUX_HOST=http://<PC-IP-ADDRESS>:8086 # PC IPv4 Address, NOT localhost
    export INFLUX_TOKEN=<ALL_ACCESS_API_TOKEN>
    export INFLUX_ORG="NextGen Hackathon"
   ```

3. Start Telegraf, command can be found `InfluxDB UI > Load Data > Telegraf > Setup Instructions`

   ```bash
    telegraf --config http://<PC-IP-ADDRESS>:8086/api/v2/telegrafs/...
   ```

4. Open new terminal, clone the repo and install python modules

   ```bash
   git clone https://github.com/Vincexodus/Train-Density-Monitor-RTS.git && cd .\Train-Density-Monitor-RTS\
   sudo pip install opencv-python spidev
   ```

5. Enable SPI on the Raspberry Pi

   ```bash
    sudo raspi-config
   ```

6. Navigate to Interfacing Options > SPI > Enable.

7. Run client script `raspberry_client.py` in Thonny IDE

### References

- [How to Setup a Raspberry Pi Pressure Pad (FSR)](https://pimylifeup.com/raspberry-pi-pressure-pad/)
- [Monitoring Your Raspberry Pi System using InfluxDB Telegraf](https://randomnerdtutorials.com/monitor-raspberry-pi-influxdb-telegraf/)
