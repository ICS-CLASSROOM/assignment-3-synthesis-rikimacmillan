import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from pyspark.sql.functions import col, count, countDistinct, min, max, when
from pyspark.sql.functions import sum, round
from pyspark.sql.functions import date_trunc, month, rank, countDistinct, sum
from pyspark.sql.window import Window
import time
import faiss
import numpy as np
import builtins
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pydantic import BaseModel, EmailStr, ValidationError
from typing import List, Optional

# Load encounters_assignment_1.csv
encounters_path = r"data/encounters_assignment_1.csv"
encounters_df = pd.read_csv(encounters_path)

# Convert timestamp to numeric
encounters_df["START"] = pd.to_datetime(encounters_df["START"])
encounters_df["START_NUMERIC"] = encounters_df["START"].apply(lambda x: x.timestamp())

# Encode patient IDs and codes as numeric features
le_patient = LabelEncoder()
le_code = LabelEncoder()

encounters_df["PATIENT_ENCODED"] = le_patient.fit_transform(encounters_df["PATIENT"])
encounters_df["CODE_ENCODED"] = le_code.fit_transform(encounters_df["CODE"])

# Create embeddings: Combine START_NUMERIC, PATIENT_ENCODED, and CODE_ENCODED
encounters_embeddings = encounters_df[["START_NUMERIC", "PATIENT_ENCODED", "CODE_ENCODED"]].values.astype(np.float32)

# Create FAISS index
dimension = encounters_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(encounters_embeddings)

# Save FAISS index
faiss.write_index(faiss_index, "data/faiss_encounters_index.bin")

# Example: Search for similar encounters
query_embedding = np.array([encounters_embeddings[0]])  # Use first encounter as an example
D, I = faiss_index.search(query_embedding, k=5)  # Find top-5 similar encounters

print("Query Result Distances:", D)
print("Query Result Indices:", I)
print("Matching Encounters:", encounters_df.iloc[I[0]])


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("COVID-19 Analysis").getOrCreate()

# Load Parquet File
parquet_path = r"data/merged_encounter_data.parquet"
encounter_df = spark.read.parquet(parquet_path)

# Load encounters_assignment_1.csv for integration
encounters_spark_df = spark.read.csv(encounters_path, header=True, inferSchema=True)


age_bins = [0, 5, 10, 17, 30, 50, 70, 150]
age_labels = ["0-5", "6-10", "11-17", "18-30", "31-50", "51-70", "71+"]

encounter_df = encounter_df.withColumn("age_range",
                                       when((col("age") >= 0) & (col("age") <= 5), "0-5")
                                       .when((col("age") > 5) & (col("age") <= 10), "6-10")
                                       .when((col("age") > 10) & (col("age") <= 17), "11-17")
                                       .when((col("age") > 17) & (col("age") <= 30), "18-30")
                                       .when((col("age") > 30) & (col("age") <= 50), "31-50")
                                       .when((col("age") > 50) & (col("age") <= 70), "51-70")
                                       .otherwise("71+"))

age_distribution = encounter_df.groupBy("age_range").count()
age_distribution.show()


