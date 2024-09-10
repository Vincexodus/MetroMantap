# Train Density Monitor - Real Time System (RTS)

Real-Time Crowd Management System for train stations using sensor technology, AI-driven analytics, and real-time data processing to improve passenger flow, reduce congestion, and optimize train operations and scheduling

### To-Do List

| Task                                      | Status            |
| ----------------------------------------- | ----------------- |
| Revamp dashboard                          | 🔨WIP             |
| Option to trasmit sample cabin footage    | 🔨WIP             |
| Demo Scene Setup                          | ❌ Not started    |
| Security measure for socket communication | ❌ Not needed yet |

### Server Setup - Central Processing (PC)

1. Install [InfluxDB](https://docs.influxdata.com/influxdb/v2/install/?t=Windows#download-and-install-influxdb-v2) and extract the zip to `C:\Program Files\InfluxData\`, rename and move all files to parent folder

2. Start InfluxDB

   ```bash
   cd 'C:\Program Files\InfluxData' && ./influxd
   ```

3. Grant network access, InfluxDB UI can be viewed at `http://localhost:8086`

4. In the UI, setup user account to obtain `All Access API Token`, copy the token

5. Install [Influx CLI](https://docs.influxdata.com/influxdb/cloud/tools/influx-cli/?t=Windows) and extract the zip to `C:\Program Files\Influx-CLI\`, rename and move all files to parent folder

6. Open new terminal, create an `influx` cli config and set it active, then apply the Raspberry Pi template

   ```bash
   cd 'C:\Program Files\Influx-CLI'
   .\influx config create -a -n main_config -u http://localhost:8086/ -t <ALL_ACCESS_API_TOKEN> -o "NextGen Hackathon"
   .\influx apply -u https://raw.githubusercontent.com/influxdata/community-templates/master/raspberry-pi/raspberry-pi-system.yml
   ```
7. Dashboard can be viewed under `Dashboard > Raspberry Pi System` once telegraf from Raspberry Pi run successfully

8. Download [Grafana installer](https://grafana.com/grafana/download?platform=windows)

9. Start grafana by running `grafana-server.exe` in `bin` directory

10. Open browser with address `http://localhost:3000/`, login with `admin` `admin`

11. Under `Home > Connections > Data sources`, add influxDB as data source

12. influxDB data source config:

    | Param          | Value                    |
    | -------------- | ------------------------ |
    | Query Language | `Flux`                   |
    | URL            | `http://localhost:8086`  |
    | Organization   | `NextGen Hackathon`      |
    | Token          | `<ALL_ACCESS_API_TOKEN>` |

13. Under `Home > Dashboards`, add visualization for new dashboard

14. Paste code example (written in Flux) to display FSR sensor readings for past 3 hours

    ```bash
     from(bucket: "Main")
       |> range(start: -3h)
       |> filter(fn: (r) => r._measurement == "sensor_data")
       |> filter(fn: (r) => r._field == "fsr1" or r._field == "fsr2" or r._field == "fsr3")
    ```

15. Clone the repo

    ```bash
    git clone https://github.com/Vincexodus/Train-Density-Monitor-RTS.git && cd .\Train-Density-Monitor-RTS\
    ```

16. Duplicate `.env.example`, rename duplicated file to `.env.local` and replace the values accordingly

17. Create virtual python environment and activate it

    ```bash
    virtualenv venv
    venv\Scripts\activate
    ```

18. Download necessary python modules

    ```bash
    pip install -r requirements.txt
    ```

19. Run server script

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
    telegraf --config http://192.168.0.190:8086/api/v2/telegrafs/0da1f37571242000
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
