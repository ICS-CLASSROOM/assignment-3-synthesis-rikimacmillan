import pandas as pd
import pyarrow.parquet as pq

# Load the original Parquet file
df = pd.read_parquet("data/merged_encounter_data.parquet")

# Convert datetime columns to microsecond precision
for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
    df[col] = df[col].dt.floor("us")  # Convert to microsecond precision

# Save the updated Parquet file
df.to_parquet("compatible_merged_encounter_data.parquet", engine="pyarrow")


import pandas as pd

try:
    df = pd.read_parquet("compatible_merged_encounter_data.parquet")
    print(df.info())
except Exception as e:
    print(f"Error reading Parquet file locally: {e}")
