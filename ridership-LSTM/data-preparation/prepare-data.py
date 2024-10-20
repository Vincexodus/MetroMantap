import pandas as pd

# Load the input CSV file
input_file = 'data\od\od-daily-data.csv'
output_file = 'data\od\od-daily-data-processed.csv'

# Read the CSV into a DataFrame
df = pd.read_csv(input_file)

# Filter out rows where 'A0: All Stations' is in either the Origin or Destination columns
filtered_df = df[(df['Origin'] != 'A0: All Stations') & (df['Destination'] != 'A0: All Stations')]

# Remove the 'B_to_A Daily Passengers' column
filtered_df = filtered_df.drop(columns=['B_to_A Daily Passengers'])

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv(output_file, index=False)

print(f'Filtered data has been saved to {output_file}')

# Sort the filtered DataFrame by 'A_to_B Daily Passengers' in descending order
# and drop duplicates to keep the highest value for each unique Origin-Destination pair
top_10_unique_pairs = (
    filtered_df
    .sort_values('A_to_B Daily Passengers', ascending=False)
    .drop_duplicates(subset=['Origin', 'Destination'])
    .head(10)[['Origin', 'Destination']]
)

# Filter the original DataFrame to include only rows with these top 10 pairs
top_10_entries = filtered_df.merge(top_10_unique_pairs, on=['Origin', 'Destination'])

# Display the entries for the top 10 unique Origin-Destination pairs
print("Entries for the top 10 Origin-Destination pairs based on A_to_B Daily Passengers:")
print(top_10_entries)

# Save the top 10 unique Origin-Destination rows to a new CSV file
output_file = 'data/testing/high-traffic-pairs.csv'
top_10_entries.to_csv(output_file, index=False)

print(f'Filtered data with all entries for the top 10 pairs has been saved to {output_file}')


