import pandas as pd
import json
import pyarrow as pa
import pyarrow.parquet as pq

# Load the JSONL file
jsonl_path = r"Parsed_notes.jsonl"

# Read JSONL into a list of dictionaries
parsed_notes = []
with open(jsonl_path, 'r') as file:
    for line in file:
        parsed_notes.append(json.loads(line))

# Convert to a Pandas DataFrame
parsed_df = pd.json_normalize(parsed_notes)
print(parsed_df.head())


# Save the parsed notes to a Parquet file
parquet_path = r"new_parsed_notes.parquet"

# Convert DataFrame to Arrow Table and save as Parquet
table = pa.Table.from_pandas(parsed_df)
pq.write_table(table, parquet_path)

print(f"Data successfully saved to Parquet at {parquet_path}")

