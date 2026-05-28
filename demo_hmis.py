"""
Demo: Shows what HMIS JSON looks like from a specific doctor transcript.
This simulates what happens when you upload audio with this content.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.analysis.hmis_mapper import HMISMapper

# ─── This is what Gemini would extract from the audio transcript: ───
# Transcript: "patient has reported fever, stomach pain since three days,
# family history of hypertension (mother) and diabetes type 2 (father).
# Dolo 650 three times a day for fever..."

SAMPLE_CASESHEET = {
    "symptoms": [
        {
            "name": "Fever",
            "since": {"value": 0, "unit": ""},
            "severity": "",
            "laterality": "",
            "finding_status": "Present",
            "details": ""
        },
        {
            "name": "Stomach Pain",
            "since": {"value": 3, "unit": "days"},
            "severity": "",
            "laterality": "",
            "finding_status": "Present",
            "details": "since three days"
        }
    ],
    "medications": [
        {
            "name": "Dolo 650",
            "duration": {"value": 0, "unit": ""},
            "frequency": "three times a day",
            "timing": "after meals",
            "dose": {"value": 1, "unit": "tablet"},
            "instruction": "for fever"
        }
    ],
    "diagnosis": [],
    "advices": [],
    "followup": {"date": "", "notes": ""},
    "PrescribedTests": [],
    "DiagnosticResults": [],
    "medicalHistory": {
        "patientHistory": {
            "patientMedicalConditions": [],
            "currentMedications": [],
            "familyHistory": [
                {"name": "Hypertension", "who": "Mother", "since": "", "notes": "", "status": "Active"},
                {"name": "Diabetes Mellitus Type 2", "who": "Father", "since": "", "notes": "", "status": "Active"}
            ],
            "lifestyleHabits": [],
            "foodOtherAllergy": [],
            "pastProcedures": [],
            "recentTravelHistory": [],
            "vaccinationHistory": [],
            "drugAllergy": []
        }
    },
    "examinations": [],
    "bodyVitalSigns": [],
    "prescriptionNotes": "",
    "confidence_score": 90
}


def main():
    print("=" * 60)
    print("TRANSCRIPT:")
    print("=" * 60)
    print('"patient has reported fever, stomach pain since')
    print(' three days, family history of hypertension (mother)')
    print(' and diabetes type 2 (father).')
    print(' Dolo 650 three times a day for fever..."')

    # Convert to HMIS
    mapper = HMISMapper()
    hmis = mapper.map_casesheet_to_hmis(SAMPLE_CASESHEET)

    print("\n" + "=" * 60)
    print("HMIS JSON OUTPUT (ready to pass to HMIS system):")
    print("=" * 60)
    print(json.dumps(hmis, indent=2, ensure_ascii=False))

    # Save
    with open("outputs/demo_transcript_hmis.json", "w", encoding="utf-8") as f:
        json.dump(hmis, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: outputs/demo_transcript_hmis.json")


if __name__ == "__main__":
    main()
