"""
Test script: Validates the HMIS mapper using a sample extracted casesheet.
Run: python test_hmis_mapper.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.analysis.hmis_mapper import HMISMapper


# ─── Sample extracted casesheet (like what Gemini produces from audio) ───
SAMPLE_CASESHEET = {
    "advices": [
        {"text": "Take full medicine for 3 days"},
        {"text": "Rest and drink plenty of water"}
    ],
    "diagnosis": [
        {
            "name": "Acute Febrile Illness",
            "since": {"value": 3, "unit": "days"},
            "status": "Suspected",
            "severity": "Moderate",
            "laterality": "",
            "details": ""
        }
    ],
    "followup": {
        "date": "2026-05-08",
        "notes": "Follow up after 3 days"
    },
    "PrescribedTests": [
        {"name": "Complete Blood Count (CBC)", "remark": "urgent"},
        {"name": "Hemoglobin (Hb)", "remark": "check levels"}
    ],
    "DiagnosticResults": [
        {"name": "X-Ray Chest", "unit": "", "value": "", "interpretation": "Normal", "remark": "PA view"}
    ],
    "medicalHistory": {
        "patientHistory": {
            "patientMedicalConditions": [
                {"name": "Chronic kidney disease (CKD)", "since": "2020", "status": "Active", "notes": ""}
            ],
            "currentMedications": [],
            "familyHistory": [],
            "lifestyleHabits": [],
            "foodOtherAllergy": [],
            "pastProcedures": [],
            "recentTravelHistory": [],
            "vaccinationHistory": [],
            "drugAllergy": []
        }
    },
    "examinations": [
        {"name": "General Examination", "notes": "Patient looks pale"}
    ],
    "bodyVitalSigns": [
        {"name": "Body Temperature", "value": {"qt": "100", "unit": "°F"}},
        {"name": "Systolic BP", "value": {"qt": "120", "unit": "mmHg"}},
        {"name": "Diastolic BP", "value": {"qt": "80", "unit": "mmHg"}},
        {"name": "Heart Rate", "value": {"qt": "90", "unit": "bpm"}},
        {"name": "SpO2", "value": {"qt": "98", "unit": "%"}},
        {"name": "Weight", "value": {"qt": "74", "unit": "kg"}},
        {"name": "Height", "value": {"qt": "162", "unit": "cm"}}
    ],
    "medications": [
        {
            "name": "Dolo 650",
            "duration": {"value": 5, "unit": "days"},
            "frequency": "thrice a day",
            "timing": "after meals",
            "dose": {"value": 1, "unit": "tablet"},
            "instruction": "for fever"
        },
        {
            "name": "Pantop DSR 40",
            "duration": {"value": 6, "unit": "days"},
            "frequency": "once a day",
            "timing": "before breakfast",
            "dose": {"value": 1, "unit": "capsule"},
            "instruction": "for acidity"
        },
        {
            "name": "Calpol Syrup",
            "duration": {"value": 0, "unit": ""},
            "frequency": "",
            "timing": "",
            "dose": {"value": 0, "unit": ""},
            "instruction": "take as needed"
        }
    ],
    "symptoms": [
        {
            "name": "fever",
            "since": {"value": 3, "unit": "days"},
            "severity": "Moderate",
            "laterality": "",
            "finding_status": "Present",
            "details": ""
        },
        {
            "name": "headache",
            "since": {"value": 2, "unit": "days"},
            "severity": "Mild",
            "laterality": "",
            "finding_status": "Present",
            "details": ""
        }
    ],
    "prescriptionNotes": "Rest well and take fluids",
    "confidence_score": 90
}


def main():
    print("=" * 60)
    print("🧪 Testing HMIS Mapper")
    print("=" * 60)

    mapper = HMISMapper()
    hmis_output = mapper.map_casesheet_to_hmis(SAMPLE_CASESHEET)

    result = hmis_output

    # ─── Validate each section ───
    checks = []

    # 1. Vitals
    vitals = result.get("additionalVitals", [])
    checks.append(("additionalVitals", len(vitals) == 7, f"Expected 7, got {len(vitals)}"))
    if vitals:
        vital_ids = [v["vitalId"] for v in vitals]
        checks.append(("vital IDs correct", "body_temperature" in vital_ids and "heart_rate" in vital_ids, 
                       f"IDs: {vital_ids}"))

    # 2. Medications
    meds = result.get("medications", [])
    checks.append(("medications", len(meds) == 3, f"Expected 3, got {len(meds)}"))
    if meds:
        checks.append(("med name", meds[0]["prescribedMedicine"] == "Dolo 650", meds[0].get("prescribedMedicine")))
        checks.append(("med frequency", meds[0]["frequency"] == "1-1-1", meds[0].get("frequency")))
        checks.append(("med timing", meds[0]["timing"] == "After Meal", meds[0].get("timing")))
        checks.append(("med duration", meds[0]["duration"] == 5, meds[0].get("duration")))
        checks.append(("dosePhases present", len(meds[0].get("dosePhases", [])) == 1, ""))

    # 3. Chief Complaints
    complaints = result.get("cheifComplaints", [])
    checks.append(("cheifComplaints", len(complaints) == 2, f"Expected 2, got {len(complaints)}"))
    if complaints:
        checks.append(("complaint value", complaints[0]["value"] == "fever", complaints[0].get("value")))

    # 4. Diagnosis
    diags = result.get("diagnosis", [])
    checks.append(("diagnosis", len(diags) == 1, f"Expected 1, got {len(diags)}"))

    # 5. Lab Investigations
    labs = result.get("labInvestigations", [])
    checks.append(("labInvestigations", len(labs) == 2, f"Expected 2, got {len(labs)}"))

    # 6. Diagnostic Investigations
    diag_inv = result.get("diagnosticInvestigations", [])
    checks.append(("diagnosticInvestigations", len(diag_inv) == 1, f"Expected 1, got {len(diag_inv)}"))

    # 7. Advices
    advices = result.get("advices", [])
    checks.append(("advices", len(advices) == 2, f"Expected 2, got {len(advices)}"))

    # 8. Medical Advices
    bg = result.get("medicalAdvices", [])
    checks.append(("medicalAdvices", len(bg) == 1, f"Expected 1, got {len(bg)}"))

    # 9. Follow-up
    followup = result.get("followUp", {})
    checks.append(("followUp", followup.get("followUpDate") == "2026-05-08", followup.get("followUpDate")))

    # 10. Diagnostics Total
    checks.append(("diagnostics total", (len(labs) + len(diag_inv)) == 3, f"Expected 3, got {len(labs) + len(diag_inv)}"))

    # 11. Treatment
    treatment = result.get("treatment", [])
    checks.append(("treatment", len(treatment) == 1, f"Expected 1, got {len(treatment)}"))

    # 12. IDs are empty (HMIS assigns)
    if meds:
        checks.append(("med ID empty", meds[0]["medicationId"] == "", meds[0].get("medicationId")))
    if complaints:
        checks.append(("complaint ID empty", complaints[0]["id"] == "", complaints[0].get("id")))

    # ─── Print results ───
    print()
    passed = 0
    failed = 0
    for name, success, detail in checks:
        if success:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name} — {detail}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'=' * 60}")

    # Save full output for review
    output_path = os.path.join("outputs", "test_hmis_output.json")
    os.makedirs("outputs", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hmis_output, f, indent=4, ensure_ascii=False)
    print(f"\n📄 Full HMIS JSON saved to: {output_path}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
