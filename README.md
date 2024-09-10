# Train Density Monitor - Real Time System (RTS)

Real-Time Crowd Management System for train stations using sensor technology, AI-driven analytics, and real-time data processing to improve passenger flow, reduce congestion, and optimize train operations and scheduling

### To-Do List

| Task                                      | Status                 |
| ----------------------------------------- | ---------------------- |
| Revamp dashboard                          | Done✅, Pending review |
| Option to trasmit sample cabin footage    | 🔨WIP                  |
| Demo Scene Setup                          | ❌ Not started         |
| Security measure for socket communication | ❌ Not needed yet      |

### Server Setup - Central Processing (PC)

1. Install [InfluxDB](https://docs.influxdata.com/influxdb/v2/install/?t=Windows#download-and-install-influxdb-v2) and extract the zip to `C:\Program Files\InfluxData\`, rename and move all files to parent folder

2. Start InfluxDB

   ```bash
   cd 'C:\Program Files\InfluxData' && ./influxd
   ```

3. Grant network access, `InfluxDB UI` can be viewed at `http://localhost:8086`

4. In `InfluxDB UI`, setup user account with org `NextGen Hackathon` to obtain `All Access API Token`, save the token

5. Under `Dashboard`, `Create Dashboard > Add a Template`, paste URL: `https://raw.githubusercontent.com/Vincexodus/Train-Density-Monitor-RTS/main/assets/raspberry-pi-system.json`

6. Clone the repo and navigate to root directory

   ```bash
   git clone https://github.com/Vincexodus/Train-Density-Monitor-RTS.git && cd .\Train-Density-Monitor-RTS\
   ```

7. Duplicate `.env.example`, rename duplicated file to `.env.local` and replace the values accordingly

8. Create virtual python environment and activate it

   ```bash
   virtualenv venv
   venv\Scripts\activate
   ```

9. Download necessary python modules

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

4. Clone the repo and install python modules

   ```bash
   git clone https://github.com/Vincexodus/Train-Density-Monitor-RTS.git && cd .\Train-Density-Monitor-RTS\
   sudo pip install opencv-python spidev
   ```

5. Enable SPI on the Raspberry Pi

   ```bash
    sudo raspi-config
   ```

6. Navigate to Interfacing Options > SPI > Enable.

7. Run client script`raspberry_client.py` in Thonny IDE

### References

- [Monitoring Your Raspberry Pi System using InfluxDB Telegraf](https://randomnerdtutorials.com/monitor-raspberry-pi-influxdb-telegraf/)
