import os
import openai
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from pydantic import ValidationError
import openai
import json

# Load CSV files
encounters_df = pd.read_csv("data/encounters_assignment_1.csv")
encounter_types_df = pd.read_csv("data/encounters_types_assignment_1.csv")
immunizations_df = pd.read_csv("data/immunizations_assignment_1.csv")
medications_df = pd.read_csv("data/medications_assignment_1.csv")
observations_df = pd.read_csv("data/observations_assignment_1.csv")

# Inspect the data
print("Encounters Data:")
print(encounters_df.head())
print("\nEncounter Types Data:")
print(encounter_types_df.head())
print("\nImmunizations Data:")
print(immunizations_df.head())
print("\nMedications Data:")
print(medications_df.head())
print("\nObservations Data:")
print(observations_df.head())


import os

# Define the path to the directory containing encounter notes
notes_dir = r"data/encounter_notes"

# List all text files in the directory
text_files = [os.path.join(notes_dir, f) for f in os.listdir(notes_dir) if f.endswith('.txt')]
print("Text Files Found:", text_files)


from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional
from datetime import datetime

class Address(BaseModel):
    city: str = Field(description="City of residence.")
    state: str = Field(description="State of residence.")
    postal_code: Optional[str] = Field(None, description="Postal code.")

class Demographics(BaseModel):
    name: str = Field(description="Full name of the patient.")
    dob: str = Field(description="Date of birth in MM/DD/YYYY format.")
    age: int = Field(description="Calculated age of the patient in years.")
    gender: str = Field(description="Gender of the patient.")
    address: Address = Field(description="Address details.")
    insurance: str = Field(description="Insurance information.")
    mrn: str = Field(description="Unique medical record number.")

    @field_validator("dob")
    def validate_dob_format(cls, value):
        try:
            datetime.strptime(value, "%m/%d/%Y")
        except ValueError:
            raise ValueError("Date of Birth must be in MM/DD/YYYY format.")
        return value

class Vitals(BaseModel):
    temperature: Optional[float] = Field(None, description="Body temperature in Celsius.")
    heart_rate: Optional[float] = Field(None, description="Heart rate in beats per minute.")
    blood_pressure: Optional[str] = Field(None, description="Blood pressure (e.g., 120/80).")
    respiratory_rate: Optional[float] = Field(None, description="Respiratory rate.")
    o2_saturation: Optional[float] = Field(None, description="Oxygen saturation.")

class SOAP(BaseModel):
    subjective: List[str] = Field(description="Subjective symptoms described by the patient.")
    objective: Vitals = Field(description="Objective findings.")
    assessment: List[str] = Field(description="Doctor's evaluation.")
    plan: List[str] = Field(description="Proposed treatment plan.")

class Encounter(BaseModel):
    encounter_note: str = Field(description="Full text of the encounter note.")
    date_of_service: datetime = Field(description="Date and time of the encounter.")
    demographics: Demographics = Field(description="Demographic information.")
    soap: SOAP = Field(description="SOAP note details.")
    provider_id: str = Field(description="Provider's unique identifier.")
    facility_id: Optional[str] = Field(None, description="Facility identifier.")
    encounter_duration: Optional[int] = Field(None, description="Duration in minutes.")
    encounter_type: str = Field(description="Type of encounter (e.g., Urgent Care).")



openai.api_key = "Secret!"

# Define the OpenAI prompt generator
def generate_openai_prompt(note: str) -> str:
    schema_description = """
    Your task is to extract structured information from unstructured medical encounter notes using the schema below. 
    - For the demographics section, calculate the patient's age in years based on the Date of Birth (DOB) and Date of Service (DOS). 
    - Ensure the Date of Birth (dob) is strictly in MM/DD/YYYY format.
    - Ensure `date_of_service` is formatted as `YYYY-MM-DD HH:MM:SS`.
    - Include the calculated age under the demographics section.
    - Ensure the output is a valid JSON object strictly following the schema.

    JSON Schema:
    {
        "encounter_note": "string - Full text of the encounter note.",
        "date_of_service": "datetime - Date and time of the encounter.",
        "demographics": {
            "name": "string - Full name of the patient.",
            "dob": "string - Patient's date of birth in MM/DD/YYYY format.",
            "age": "int - Patient's age in years, calculated from DOB and DOS.",
            "gender": "string - Gender of the patient (e.g., Male, Female).",
            "address": {
                "city": "string - City of residence.",
                "state": "string - State of residence.",
                "postal_code": "string - Postal code (if available)."
            },
            "insurance": "string - Name of the insurance provider.",
            "mrn": "string - Unique medical record number."
        },
        "soap": {
            "subjective": ["string - List of symptoms described by the patient."],
            "objective": {
                "vitals": {
                    "temperature": "float - Body temperature in Celsius.",
                    "heart_rate": "float - Heart rate in beats per minute.",
                    "blood_pressure": "string - Blood pressure in mmHg (e.g., 120/80).",
                    "respiratory_rate": "float - Respiratory rate in breaths per minute.",
                    "o2_saturation": "float - Oxygen saturation percentage."
                }
            },
            "assessment": ["string - List of diagnoses or evaluations."],
            "plan": ["string - List of proposed treatments or next steps."]
        },
        "provider_id": "string - Unique identifier of the healthcare provider.",
        "facility_id": "string - Unique identifier of the facility.",
        "encounter_duration": "int - Duration of the encounter in minutes.",
        "encounter_type": "string - Type of encounter (e.g., Urgent Care, Ambulatory)."
    }
    """
    prompt = f"""
    Extract structured information from the following medical encounter note. Ensure the data strictly adheres to the JSON schema described above. 
    - Calculate the patient's age in years by subtracting the year of birth (DOB) from the year of the Date of Service (DOS). 
    - Ensure all fields are present, even if optional fields (like postal code) are null. 

    Encounter Note:
    {note}
    """
    return schema_description + prompt

# Parse encounter notes using OpenAI
def parse_encounter_notes(note: str) -> dict:
    prompt = generate_openai_prompt(note)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical assistant skilled at extracting structured data from medical notes."},
                {"role": "user", "content": prompt}
            ]
        )
        # Convert the JSON string output to a Python dictionary
        return json.loads(response['choices'][0]['message']['content'])
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None

# Preprocess the date_of_service field (for preventing validation errors)
def preprocess_date_of_service(json_data: dict) -> dict:
    try:
        if "date_of_service" in json_data:
            raw_date = json_data["date_of_service"]
            # If already in ISO format, skip further processing
            if "T" in raw_date or "-" in raw_date:
                return json_data
            # Parse and reformat non-standard datetime
            parsed_date = datetime.strptime(raw_date.split("-")[0], "%B %d, %Y %H:%M")
            json_data["date_of_service"] = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error during date_of_service preprocessing: {e}")
    return json_data

# Directory containing encounter notes
notes_dir = r"data/encounter_notes"
text_files = [os.path.join(notes_dir, f) for f in os.listdir(notes_dir) if f.endswith('.txt')]

# Process and validate encounter notes
validated_encounters = []

for file_path in text_files:
    with open(file_path, 'r') as f:
        raw_note = f.read()
        print(f"Processing: {file_path}")
        parsed_note = parse_encounter_notes(raw_note)
        if parsed_note:
            try:
                # Preprocess the parsed JSON data
                parsed_json = preprocess_date_of_service(parsed_note)
                # Validate the preprocessed data using Pydantic
                validated_data = Encounter.model_validate(parsed_json)
                validated_encounters.append(validated_data.model_dump())
            except ValidationError as e:
                print(f"Validation Error for {file_path}:", e.json(indent=2))

# Save validated data to Parquet
validated_df = pd.DataFrame(validated_encounters)
parquet_path = r"data/merged_encounter_data.parquet"
table = pa.Table.from_pandas(validated_df)
pq.write_table(table, parquet_path)

print("Data successfully saved to Parquet.")