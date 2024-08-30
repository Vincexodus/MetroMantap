# Train-Density-Monitor-RTS

Real-Time Crowd Management System for train stations using sensor technology, AI-driven analytics, and real-time data processing to improve passenger flow, reduce congestion, and optimize train operations and scheduling.

### Server Setup - Central Processing (PC)

1. Setup InfluxDB [here](https://docs.influxdata.com/influxdb/v2/install/?t=Windows)

2. Extract the zip into `C:\Program Files\InfluxData\influxdb`

3. Start InfluxDB

   ```bash
    cd -Path 'C:\Program Files\InfluxData\influxdb'
    ./influxd
   ```

4. Download telegraf using powershell

   ```bash
    wget `
    https://dl.influxdata.com/telegraf/releases/telegraf-1.31.3_windows_amd64.zip `
    -UseBasicParsing `
    -OutFile telegraf-1.31.3_windows_amd64.zip
    Expand-Archive .\telegraf-1.31.3_windows_amd64.zip `
    -DestinationPath 'C:\Program Files\Telegraf\'
   ```

5. Extract the zip fie & move `telegraf.exe` & `telegraf.conf` from `C:\Program Files\Telegraf\telegraf-1.31.3` to `C:\Program Files\Telegraf`

6. Install telegraf as Windows service

   ```bash
    # > C:\Program Files\Telegraf
    .\telegraf.exe --service install `
    --config "C:\Program Files\Telegraf\telegraf.conf"
   ```

7. Create infuxData bucket called `Main` [here](https://us-east-1-1.aws.cloud2.influxdata.com)

8. Replace `OUTPUTS PLUGINS` section of `telegraf.conf`

   ```bash
    [[outputs.influxdb_v2]]
      ## The URLs of the InfluxDB cluster nodes.
      ##
      ## Multiple URLs can be specified for a single cluster, only ONE of the
      ## urls will be written to each interval.
      ##   ex: urls = ["https://us-west-2-1.aws.cloud2.influxdata.com"]
      urls = ["https://us-east-1-1.aws.cloud2.influxdata.com"]

      ## API token for authentication.
      token = "$INFLUX_TOKEN"

      ## Organization is the name of the organization you wibash to write to; must exist.
      organization = "NextGen Hackathon"

      ## Destination bucket to write into.
      bucket = "Main"

      ## The value of this tag will be used to determine the bucket.  If this
      ## tag is not set the 'bucket' option is used as the default.
      # bucket_tag = ""

      ## If true, the bucket tag will not be added to the metric.
      # exclude_bucket_tag = false

      ## Timeout for HTTP messages.
      # timeout = "5s"

      ## Additional HTTP headers
      # http_headers = {"X-Special-Header" = "Special-Value"}

      ## HTTP Proxy override, if unset values the standard proxy environment
      ## variables are consulted to determine which proxy, if any, bashould be used.
      # http_proxy = "http://corporate.proxy:3128"

      ## HTTP User-Agent
      # user_agent = "telegraf"

      ## Content-Encoding for write request body, can be set to "gzip" to
      ## compress body or "identity" to apply no encoding.
      # content_encoding = "gzip"

      ## Enable or disable uint support for writing uints influxdb 2.0.
      # influx_uint_support = false

      ## Optional TLS Config for use on HTTP connections.
      # tls_ca = "/etc/telegraf/ca.pem"
      # tls_cert = "/etc/telegraf/cert.pem"
      # tls_key = "/etc/telegraf/key.pem"
      ## Use TLS but skip chain & host verification
      # insecure_skip_verify = false
   ```

9. Start telegraf service

   ```bash
    .\telegraf  --config "C:\Program Files\Telegraf\telegraf.conf"
   ```

10. Create all access api token for grafana [here](https://us-east-1-1.aws.cloud2.influxdata.com)

11. Download Grafana (installer) [here](https://grafana.com/grafana/download?platform=windows)

12. Start grafana by running `grafana-server.exe` in `bin` directory

13. Open browser with address `http://localhost:3000/`, login with `admin` `admin`

14. Under `Home > Connections > Data sources` add influxDB as data source

15. influxDB data source config:

    | Param          | Value                                           |
    | -------------- | ----------------------------------------------- |
    | Query Language | `Flux`                                          |
    | URL            | `https://us-east-1-1.aws.cloud2.influxdata.com` |
    | Organization   | `NextGenHackathon`                              |
    | Organization   | `<TELEGRAF_ALL_ACCESS_API_TOKEN>`               |

16. Start telegraf service

    ```bash
    py pc_server.py
    ```

### Client Setup - Raspberry Pi 4

1. Install OpenCV for Video Capture

   ```bash
    sudo apt-get update
    sudo apt-get install python3-opencv
   ```

2. Install SPI-Py and Spidev for SPI Communication with ADC

   ```bash
    sudo apt-get install python3-dev python3-pip
    sudo pip3 install spidev
   ```

3. Enable SPI on the Raspberry Pi

   ```bash
    sudo raspi-config
   ```

4. Navigate to Interfacing Options > SPI > Enable.

5. Run `raspberry_client.py` in Thonny IDE.
