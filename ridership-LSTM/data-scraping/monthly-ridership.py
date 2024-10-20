import pandas as pd

URL_DATA = 'https://storage.data.gov.my/transportation/ridership_headline.parquet'

df = pd.read_parquet(URL_DATA)
if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])

print(df)