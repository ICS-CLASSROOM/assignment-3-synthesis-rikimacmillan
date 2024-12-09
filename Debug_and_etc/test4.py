import pandas as pd
from pandas import json_normalize

# Load the DataFrame
df = pd.read_parquet("compatible_merged_encounter_data.parquet")

# Flatten 'demographics' and 'soap' columns
demographics_df = json_normalize(df["demographics"])
soap_df = json_normalize(df["soap"])

# Merge flattened columns with the main DataFrame
df_flattened = pd.concat([df.drop(columns=["demographics", "soap"]), demographics_df, soap_df], axis=1)

# Convert datetime64[ns] to a Spark-compatible format
df_flattened["date_of_service"] = df_flattened["date_of_service"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Save the updated DataFrame to a new Parquet file
df_flattened.to_parquet("flattened_compatible_merged_encounter_data.parquet", engine="pyarrow")
