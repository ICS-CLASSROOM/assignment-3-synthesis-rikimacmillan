from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional
from datetime import datetime

class Address(BaseModel):
    city: str = Field(description="The city where the patient lives. Should be under DEMOGRAPHICS header.")
    state: str = Field(description="The state where the patient lives. Should be under DEMOGRAPHICS header.")
    postal_code: str = Field(description="The postal code of the patient's address.")

class Demographics(BaseModel):
    name: str = Field(description="The full name of the patient as provided under DEMOGRAPHICS.")
    dob: str = Field(description="The date of birth in MM/DD/YYYY format.")
    gender: str = Field(description="Gender of the patient, e.g., Male, Female, or Non-binary.")
    address: Address = Field(description="Nested address details of the patient.")
    insurance: str = Field(description="Name of the insurance provider.")
    mrn: str = Field(description="The unique medical record number.")

    @field_validator("dob")
    def validate_dob(cls, value):
        try:
            datetime.strptime(value, "%m/%d/%Y")
        except ValueError:
            raise ValueError("Date of Birth must be in MM/DD/YYYY format.")
        return value

class Vitals(BaseModel):
    height: Optional[float] = Field(None, description="Patient's height in cm.")
    weight: Optional[float] = Field(None, description="Patient's weight in kg.")
    bmi: Optional[float] = Field(None, description="Calculated Body Mass Index (BMI).")
    bp: Optional[str] = Field(None, description="Blood pressure in mmHg format (e.g., 120/80).")
    hr: Optional[float] = Field(None, description="Heart rate in beats per minute.")
    rr: Optional[float] = Field(None, description="Respiratory rate in breaths per minute.")
    temperature: Optional[float] = Field(None, description="Body temperature in °C.")
    o2_saturation: Optional[float] = Field(None, description="Oxygen saturation percentage on room air.")

class LaboratoryTesting(BaseModel):
    test_name: str = Field(description="Name of the test performed.")
    result: str = Field(description="The result of the test.")

class SOAP(BaseModel):
    subjective: List[str] = Field(description="Patient's subjective description of their symptoms.")
    objective: Vitals = Field(description="Doctor's objective findings including vitals.")
    laboratory_testing: List[LaboratoryTesting] = Field(description="Laboratory testing details.")
    assessment: List[str] = Field(description="Doctor's evaluation based on subjective and objective data.")
    plan: List[str] = Field(description="Proposed treatment plan.")

class Encounter(BaseModel):
    encounter_note: str = Field(description="The full encounter note as text.")
    date_of_service: datetime = Field(description="The date and time when the encounter took place.")
    demographics: Demographics = Field(description="Patient demographic details.")
    recent_visit_reason: Optional[str] = Field(None, description="Reason for the most recent visit.")
    recent_visit_date: Optional[str] = Field(None, description="Date of the most recent visit in MM/DD/YYYY format.")
    soap: SOAP = Field(description="SOAP note structure.")
    provider_id: str = Field(description="Unique identifier of the provider.")
    facility_id: str = Field(description="Unique identifier of the facility.")
    encounter_duration: Optional[int] = Field(None, description="Duration of the encounter in minutes.")
    encounter_type: str = Field(description="Type of encounter, e.g., Ambulatory, Emergency.")

    @field_validator("recent_visit_date", mode="before")
    def validate_recent_visit_date(cls, value):
        if value:
            try:
                datetime.strptime(value, "%m/%d/%Y")
            except ValueError:
                raise ValueError("Recent Visit Date must be in MM/DD/YYYY format.")
        return value
