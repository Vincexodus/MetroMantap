import csv
from bs4 import BeautifulSoup

# Load your HTML file (assuming it's saved as 'data.html')
with open('Rapid Rail Explorer _ data.gov.my.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Dictionary to store O-D pairs
od_combinations = {}

# Locate the section that contains origin-destination data
# Assuming the data is stored in the 'dropdown' section within 'rail'
od_section = soup.find_all('script')

for script in od_section:
    if '"dropdown":' in script.text:  # Adjust this to fit the specific section
        # Here we assume the format similar to the one you posted, parse the JavaScript-like data
        od_data = script.string.strip()
        # Extract the origin-destination part (adjust according to your exact structure)
        lines = od_data.splitlines()

        current_origin = None

        for line in lines:
            line = line.strip()

            if '": [' in line:  # Detect a new origin
                # Extract the full station name including the ID (e.g., "AG02: Sentul")
                origin = line.split(":")[0].strip('"')
                station_name = line.split(": ")[1].strip('["')
                full_origin = f'{origin}: {station_name}'
                current_origin = full_origin
                od_combinations[current_origin] = []

            elif '",' in line and current_origin:  # Detect a destination under the origin
                destination = line.strip('",')
                od_combinations[current_origin].append(destination)

# Write the O-D combinations to a CSV file
with open('od_combinations.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Origin', 'Destination'])  # Header row
    for origin, destinations in od_combinations.items():
        for destination in destinations:
            writer.writerow([origin, destination])

print("O-D combinations have been saved to 'od_combinations.csv'")
