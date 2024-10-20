import requests
from bs4 import BeautifulSoup
import json
import csv
from urllib.parse import quote
from datetime import datetime
import time

# Path to your input CSV file containing Origin-Destination pairs
input_csv_filename = 'od_combinations.csv'

# Output CSV filenames for daily and monthly data
daily_csv_filename = 'od_daily_data_2.csv'
monthly_csv_filename = 'od_monthly_data_2.csv'

# Base URL for constructing the O-D links
base_url = "https://data.gov.my/dashboard/rapid-explorer/rail/"

# Function to encode station names for URLs
def encode_station(station):
    return quote(station, safe='')

# Function to convert timestamps to readable dates
def convert_timestamps_to_dates(timestamps):
    return [datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d') for ts in timestamps]

# Create and initialize daily CSV file
with open(daily_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Origin', 'Destination', 'Date', 'A_to_B Daily Passengers', 'B_to_A Daily Passengers'])

# Create and initialize monthly CSV file
with open(monthly_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Origin', 'Destination', 'Month', 'A_to_B Monthly Passengers', 'B_to_A Monthly Passengers'])

# Read the Origin-Destination pairs from the input CSV file
with open(input_csv_filename, newline='', encoding='utf-8') as csvfile:
    od_pairs = csv.reader(csvfile)
    next(od_pairs)  # Skip the header row

    # Loop through O-D pairs and scrape data
    for row in od_pairs:
        origin, destination = row
        # Construct the URL
        url = f"{base_url}{encode_station(origin)}/{encode_station(destination)}"
        print(f"Scraping URL: {url}")

        try:
            # Make an HTTP GET request
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code != 200:
                print(f"Failed to retrieve data from {url} (Status code: {response.status_code})")
                continue

            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the JSON data in the script tag
            script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
            if not script_tag:
                print(f"Error: JSON data not found for {url}")
                continue

            # Parse the JSON data
            json_data = json.loads(script_tag.string)

            # Extract A_to_B and B_to_A daily and monthly passengers
            a_to_b_daily = json_data['props']['pageProps']['A_to_B']['data'].get('daily', {})
            a_to_b_monthly = json_data['props']['pageProps']['A_to_B']['data'].get('monthly', {})

            # Check if B_to_A exists before accessing it
            b_to_a = json_data['props']['pageProps'].get('B_to_A')
            if b_to_a:
                b_to_a_daily = b_to_a.get('daily', {})
                b_to_a_monthly = b_to_a.get('monthly', {})
            else:
                print(f"No B_to_A data for {origin} -> {destination}")
                b_to_a_daily = {}
                b_to_a_monthly = {}

            # Extract passengers data
            a_to_b_daily_passengers = a_to_b_daily.get('passengers', [])
            a_to_b_monthly_passengers = a_to_b_monthly.get('passengers', [])
            b_to_a_daily_passengers = b_to_a_daily.get('passengers', [])
            b_to_a_monthly_passengers = b_to_a_monthly.get('passengers', [])

            # Extract corresponding timestamps (dates)
            a_to_b_daily_dates = convert_timestamps_to_dates(a_to_b_daily.get('x', []))
            a_to_b_monthly_dates = convert_timestamps_to_dates(a_to_b_monthly.get('x', []))
            b_to_a_daily_dates = convert_timestamps_to_dates(b_to_a_daily.get('x', []))
            b_to_a_monthly_dates = convert_timestamps_to_dates(b_to_a_monthly.get('x', []))

            # Save daily data to daily CSV file
            with open(daily_csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write daily rows (A_to_B and B_to_A)
                for i, date in enumerate(a_to_b_daily_dates):
                    a_to_b_daily_value = a_to_b_daily_passengers[i] if i < len(a_to_b_daily_passengers) else None
                    b_to_a_daily_value = b_to_a_daily_passengers[i] if i < len(b_to_a_daily_passengers) else None
                    writer.writerow([origin, destination, date, a_to_b_daily_value, b_to_a_daily_value])

            # Save monthly data to monthly CSV file
            with open(monthly_csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write monthly rows (A_to_B and B_to_A)
                for i, date in enumerate(a_to_b_monthly_dates):
                    a_to_b_monthly_value = a_to_b_monthly_passengers[i] if i < len(a_to_b_monthly_passengers) else None
                    b_to_a_monthly_value = b_to_a_monthly_passengers[i] if i < len(b_to_a_monthly_passengers) else None
                    writer.writerow([origin, destination, date, a_to_b_monthly_value, b_to_a_monthly_value])

            print(f"Data for {origin} -> {destination} saved successfully.")

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"Error scraping {url}: {e}")

        # Optional: Add a delay between requests to avoid overwhelming the server
        time.sleep(0.5)

print("Scraping completed.")
