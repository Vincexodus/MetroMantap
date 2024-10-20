# MetroMantap

<p align="center">
  <i align="center">Real-Time Crowd Management System for train stations using sensor technology, AI-driven analytics, and real-time data processing to improve passenger flow, reduce congestion, and optimize train operations and scheduling</i>
</p>

## Future Enhancements

- [ ] Color grouping for passengers (dashboard)
- [ ] GET request from LSTM API (dashboard)
- [ ] Asymmetric Encryption
- [ ] Cabin exit gesture detection
- [ ] Use GET request to scrape ridership data (LSTM)

## Server Setup - Central Processing (PC)

### Pre-requisites:

- [PyTorch](https://pytorch.org/get-started/pytorch-2.0/#requirements)
- [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Setup

1. Install [InfluxDB](https://docs.influxdata.com/influxdb/v2/install/?t=Windows#download-and-install-influxdb-v2) and extract the zip to `C:\Program Files\InfluxData\`, rename and move all files to parent folder

2. Start InfluxDB

   ```bash
   cd 'C:\Program Files\InfluxData' && ./influxd
   ```

3. Grant network access, `InfluxDB UI` can be viewed at `http://localhost:8086`

4. In `InfluxDB UI`, setup user account with org `NextGen Hackathon` to obtain `All Access API Token`, save the token

5. In `InfluxDB UI > Load Data > Buckets`, create 2 buckets `main` and `rasp-pi`

6. Under `Dashboard`, `Create Dashboard > Add a Template`, paste this [URL](https://raw.githubusercontent.com/Vincexodus/MetroMantap/main/assets/train_density_monitor_dashboard.json)

7. Open new terminal, clone the repo and navigate to root directory

   ```bash
   git clone https://github.com/Vincexodus/MetroMantap.git && cd .\MetroMantap\
   ```

8. Duplicate `.env.example` and rename the duplicated file to `.env.local`, replace the values accordingly

9. Install [OpenSSL](https://slproweb.com/products/Win32OpenSSL.html) and generate self-signed ssl cert

   ```bash
   # root directory
   cd assets && ./ssl_cert.sh
   ```

10. Copy generated folder called `ssl` to Raspberry PI.

11. Create virtual python environment and activate it

    ```bash
    virtualenv venv
    venv\Scripts\activate
    ```

12. Download required python modules

    ```bash
    pip install -r requirements.txt
    ```

13. Run server script (recommend with [PyTorch](https://pytorch.org/get-started/pytorch-2.0/#requirements) + [Cuda](https://developer.nvidia.com/cuda-downloads) installed)

    ```bash
    py pc_server.py
    ```

## Client Setup - Raspberry Pi 4

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
   git clone https://github.com/Vincexodus/MetroMantap.git && cd .\MetroMantap\
   sudo pip install opencv-python spidev ssl
   ```

5. Enable SPI on the Raspberry Pi

   ```bash
    sudo raspi-config
   ```

6. Navigate to Interfacing Options > SPI > Enable.

7. Run client script `raspberry_client.py` in Thonny IDE

## LSTM Model Training and Inference

1. Data Collection:

   - Download the HTML file from [data.gov.my dashboard](https://data.gov.my/dashboard/rapid-explorer)
   - Run the following commands to extract and scrape data:

   ```bash
      cd ridership-LSTM
      python data-scraping/train-combination.py  # Extract possible OD pair combinations
      python data-scraping/od-ridership.py       # Scrape the latest daily OD ridership data
      python data-scraping/monthly-ridership.py  # Query the OpenAPI for monthly ridership data
   ```

2. Data Preparation:

   - Use `data-preparation/prepare-data.py` to process and prepare the collected data

3. Model Training:

   - `model/lstm.py` is used to train the LSTM models for each OD pair

4. API:
   - `api.py` provides an API interface for the trained models
   - Open a new terminal and navigate to `ridership-LSTM` directory
   - Start the app using `uvicorn api:app --reload`
   - This endpoint reads the `future_predictions.csv` file from each OD pair folder in the `model/lstm/` directory and returns the predictions as a JSON response
   - The API will be available at `http://0.0.0.0:8000/predictions/`

## References

- [How to Setup a Raspberry Pi Pressure Pad (FSR)](https://pimylifeup.com/raspberry-pi-pressure-pad/)
- [Monitoring Your Raspberry Pi System using InfluxDB Telegraf](https://randomnerdtutorials.com/monitor-raspberry-pi-influxdb-telegraf/)
- [Easy Step-by-Step Guide to Installing CUDA for PyTorch](https://medium.com/@fernandopalominocobo/installing-cuda-for-pytorch-easily-explained-windows-users-4d3b7db5f2e0)

## Acknowledgments

- Data source: [data.gov.my](https://data.gov.my)
- This project was developed as part of the MMU ZTE 5G Hackathon
