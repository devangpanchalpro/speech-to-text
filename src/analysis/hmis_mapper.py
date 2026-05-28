import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional


# ─── Vital Sign Name → HMIS vitalId Lookup ───
VITAL_ID_MAP = {
    # Temperature
    "body temperature": "body_temperature",
    "temperature": "body_temperature",
    "temp": "body_temperature",
    # Blood Pressure
    "systolic bp": "systolic_bp",
    "systolic blood pressure": "systolic_bp",
    "systolic": "systolic_bp",
    "diastolic bp": "diastolic_bp",
    "diastolic blood pressure": "diastolic_bp",
    "diastolic": "diastolic_bp",
    "bp systolic": "systolic_bp",
    "bp diastolic": "diastolic_bp",
    # Heart
    "heart rate": "heart_rate",
    "pulse": "heart_rate",
    "pulse rate": "heart_rate",
    "hr": "heart_rate",
    # Respiratory
    "respiratory rate": "respiratory_rate",
    "respiration rate": "respiratory_rate",
    "rr": "respiratory_rate",
    "breathing rate": "respiratory_rate",
    # Oxygen
    "oxygen saturation": "oxygen_saturation_spo2",
    "spo2": "oxygen_saturation_spo2",
    "o2 saturation": "oxygen_saturation_spo2",
    "o2 sat": "oxygen_saturation_spo2",
    # Height / Weight / BMI
    "height": "body_height",
    "body height": "body_height",
    "weight": "body_weight",
    "body weight": "body_weight",
    "bmi": "body_mass_index",
    "body mass index": "body_mass_index",
    # Blood Glucose
    "blood glucose": "randomBloodGlucose",
    "random blood glucose": "randomBloodGlucose",
    "rbs": "randomBloodGlucose",
    "blood sugar": "randomBloodGlucose",
    "glucose": "randomBloodGlucose",
}

# ─── Duration Unit → HMIS durationPeriod ───
DURATION_PERIOD_MAP = {
    "day": 1, "days": 1, "d": 1,
    "week": 2, "weeks": 2, "w": 2,
    "month": 3, "months": 3, "m": 3,
    "year": 4, "years": 4, "y": 4,
}

# ─── Frequency text → HMIS format (e.g. "1-1-1") ───
FREQUENCY_MAP = {
    # Once daily
    "once a day": "1-0-0",
    "once daily": "1-0-0",
    "od": "1-0-0",
    "once": "1-0-0",
    "1 time a day": "1-0-0",
    # Twice daily
    "twice a day": "1-0-1",
    "twice daily": "1-0-1",
    "bd": "1-0-1",
    "bid": "1-0-1",
    "2 times a day": "1-0-1",
    # Thrice daily
    "thrice a day": "1-1-1",
    "thrice daily": "1-1-1",
    "thrice day": "1-1-1",
    "three times a day": "1-1-1",
    "tid": "1-1-1",
    "tds": "1-1-1",
    "3 times a day": "1-1-1",
    # Four times daily
    "four times a day": "1-1-1-1",
    "qid": "1-1-1-1",
    "4 times a day": "1-1-1-1",
    # SOS / As needed
    "as needed": "SOS",
    "sos": "SOS",
    "when needed": "SOS",
    "prn": "SOS",
    # Morning only
    "morning": "1-0-0",
    "in the morning": "1-0-0",
    # Night only
    "night": "0-0-1",
    "at night": "0-0-1",
    "bedtime": "0-0-1",
    "at bedtime": "0-0-1",
    "hs": "0-0-1",
}

# ─── Timing text → HMIS format ───
TIMING_MAP = {
    "after food": "After Meal",
    "after meals": "After Meal",
    "after meal": "After Meal",
    "after eating": "After Meal",
    "after lunch": "After Meal",
    "after dinner": "After Meal",
    "after breakfast": "After Meal",
    "before food": "Before Meal",
    "before meals": "Before Meal",
    "before meal": "Before Meal",
    "before eating": "Before Meal",
    "before breakfast": "Before Meal",
    "empty stomach": "Before Meal",
    "on empty stomach": "Before Meal",
    "with food": "With Meal",
    "with meals": "With Meal",
    "any time": "Any Time",
    "anytime": "Any Time",
}


class HMISMapper:
    """
    Converts voice-extracted EMR casesheet JSON into HMIS-compatible JSON format.
    
    - IDs (visitId, patientId, etc.) are always blank — HMIS assigns them.
    - Only includes fields that have actual data from the voice extraction.
    - Output is a flat JSON (no "result" wrapper).
    """

    def __init__(self):
        self.now = datetime.now()

    def map_casesheet_to_hmis(self, casesheet: Dict, metadata: Optional[Dict] = None) -> Dict:
        """
        Main entry point: takes extracted casesheet and returns HMIS JSON body.
        
        Args:
            casesheet: The extracted EMR casesheet from voice processing
            metadata: Optional metadata (doctor name, patient name, etc.)
            
        Returns:
            HMIS-compatible JSON dictionary (flat, no "result" wrapper)
        """
        if not casesheet:
            return self._get_empty_hmis()

        hmis = {
            # ─── Context fields (always blank — HMIS assigns) ───
            "visitId": "",
            "patientId": "",
            "facilityId": "",
            "healthProfessionalId": "",
            "appointmentId": "",
            "visitTime": self.now.isoformat(),
            "doctorPrivateNote": casesheet.get("prescriptionNotes", ""),
            "isConsultationEnded": False,
            "linkWithAbha": False,

            # ─── Mapped from voice extraction ───
            "additionalVitals": self._map_vitals(casesheet.get("bodyVitalSigns", [])),
            "medications": self._map_medications(casesheet.get("medications", [])),
            "cheifComplaints": self._map_chief_complaints(casesheet.get("symptoms", [])),
            "diagnosis": self._map_diagnosis(casesheet.get("diagnosis", [])),
            "diagnosticInvestigations": self._map_diagnostic_investigations(
                casesheet.get("DiagnosticResults", [])
            ),
            "labInvestigations": self._map_lab_investigations(
                casesheet.get("PrescribedTests", [])
            ),
            "advices": self._map_advices(casesheet.get("advices", [])),
            "medicalAdvices": self._map_medical_advices(casesheet.get("examinations", [])),
            "treatment": self._map_treatment(casesheet.get("prescriptionNotes", ""),
                                             casesheet.get("medicalHistory", {})),
            "extendedProps": [],
            "eyeExamination": {},
        }

        # Only include followUp if data exists
        followup = self._map_followup(casesheet.get("followup", {}))
        if followup:
            hmis["followUp"] = followup

        return hmis

    # ─── Individual Field Mappers ────────────────────────────────────

    def _map_vitals(self, vitals: List[Dict]) -> List[Dict]:
        """Map bodyVitalSigns → additionalVitals"""
        mapped = []
        for vital in vitals:
            name = vital.get("name", "").lower().strip()
            value_obj = vital.get("value", {})
            value = str(value_obj.get("qt", "")) if isinstance(value_obj, dict) else str(value_obj)

            if not name or not value:
                continue

            vital_id = self._lookup_vital_id(name)
            if vital_id:
                mapped.append({
                    "value": value,
                    "vitalId": vital_id
                })
            else:
                # Use the name itself as a custom vitalId
                mapped.append({
                    "value": value,
                    "vitalId": name.replace(" ", "_").lower()
                })
        return mapped

    def _map_medications(self, medications: List[Dict]) -> List[Dict]:
        """Map medications → HMIS medications format"""
        mapped = []
        for med in medications:
            name = med.get("name", "")
            if not name:
                continue

            dosage = med.get("dose", {}).get("value", 1) if isinstance(med.get("dose"), dict) else 1
            if dosage == 0:
                dosage = 1

            raw_frequency = med.get("frequency", "")
            frequency = self._convert_frequency(raw_frequency)

            raw_timing = med.get("timing", "")
            timing = self._convert_timing(raw_timing)

            duration_obj = med.get("duration", {})
            duration = duration_obj.get("value", 0) if isinstance(duration_obj, dict) else 0
            duration_unit = duration_obj.get("unit", "days") if isinstance(duration_obj, dict) else "days"
            duration_period = DURATION_PERIOD_MAP.get(duration_unit.lower(), 1)

            instruction = med.get("instruction", "")

            hmis_med = {
                "prescribedMedicine": name,
                "dosage": dosage,
                "frequency": frequency,
                "timing": timing if timing else "Any Time",
                "duration": duration if duration > 0 else 1,
                "durationPeriod": duration_period,
                "medicineType": 1,
                "reason": instruction,
                "medicationId": "",
                "isInternal": False,
                "scheduleDays": [],
                "medicationSchedule": 1,
                "unitType": 897,
                "additionalData": {
                    "medicineComposition": ""
                },
                "dosePhases": [
                    {
                        "order": 1,
                        "dosage": dosage,
                        "duration": duration if duration > 0 else 1,
                        "durationPeriod": duration_period,
                        "timing": timing if timing else "Any Time",
                        "frequency": frequency
                    }
                ]
            }
            mapped.append(hmis_med)
        return mapped

    def _map_chief_complaints(self, symptoms: List[Dict]) -> List[Dict]:
        """Map symptoms → cheifComplaints"""
        mapped = []
        now_str = self.now.isoformat()
        for symptom in symptoms:
            name = symptom.get("name", "")
            if not name:
                continue

            mapped.append({
                "id": "",
                "value": name,
                "recordedDate": now_str,
                "additionalData": {
                    "medicalConceptId": ""
                }
            })
        return mapped

    def _map_diagnosis(self, diagnoses: List[Dict]) -> List[Dict]:
        """Map diagnosis → HMIS diagnosis"""
        mapped = []
        for diag in diagnoses:
            name = diag.get("name", "")
            if not name:
                continue

            mapped.append({
                "id": "",
                "value": name,
                "additionalData": {
                    "medicalConceptId": ""
                }
            })
        return mapped

    def _map_lab_investigations(self, tests: List[Dict]) -> List[Dict]:
        """Map PrescribedTests → labInvestigations"""
        mapped = []
        now_str = self.now.isoformat()
        for test in tests:
            name = test.get("name", "")
            if not name:
                continue

            mapped.append({
                "id": "",
                "value": name,
                "instructions": test.get("remark", ""),
                "recordedDate": now_str
            })
        return mapped

    def _map_diagnostic_investigations(self, results: List[Dict]) -> List[Dict]:
        """Map DiagnosticResults → diagnosticInvestigations"""
        mapped = []
        now_str = self.now.isoformat()
        for result in results:
            name = result.get("name", "")
            if not name:
                continue

            mapped.append({
                "id": "",
                "value": name,
                "instructions": result.get("remark", ""),
                "recordedDate": now_str
            })
        return mapped

    def _map_advices(self, advices: List[Dict]) -> List[Dict]:
        """Map advices → HMIS advices"""
        mapped = []
        for advice in advices:
            text = advice.get("text", "")
            if not text:
                continue

            mapped.append({
                "value": text,
                "id": "",
                "additionalData": {}
            })
        return mapped

    def _map_medical_advices(self, examinations: List[Dict]) -> List[Dict]:
        """Map examinations → medicalAdvices (clinical findings/notes)"""
        mapped = []
        for exam in examinations:
            name = exam.get("name", "")
            notes = exam.get("notes", "")
            text = f"{name}: {notes}".strip(": ") if name or notes else ""
            if text:
                mapped.append({
                    "value": text,
                    "id": ""
                })
        return mapped

    def _map_followup(self, followup: Dict) -> Dict:
        """Map followup → HMIS followUp"""
        date = followup.get("date", "")
        notes = followup.get("notes", "")

        if not date and not notes:
            return {}

        # Try to calculate followUpAfter from date
        follow_up_after = 0
        follow_up_date = date
        if date:
            try:
                parsed = datetime.strptime(date, "%Y-%m-%d")
                delta = parsed - self.now
                follow_up_after = max(delta.days, 1)
            except (ValueError, TypeError):
                try:
                    parsed = datetime.strptime(date, "%d/%m/%Y")
                    delta = parsed - self.now
                    follow_up_after = max(delta.days, 1)
                    follow_up_date = parsed.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    follow_up_after = 0

        return {
            "followUpId": "",
            "followUpPeriod": 1,
            "appointmentType": 1,
            "consultationType": 0,
            "followUpAfter": follow_up_after,
            "followUpDate": follow_up_date,
            "followUpTime": "",
            "notes": notes,
            "oldVisitId": ""
        }

    def _map_treatment(self, prescription_notes: str, medical_history: Dict) -> List[str]:
        """Map prescriptionNotes + any extra notes → treatment[]"""
        treatments = []
        if prescription_notes and prescription_notes.strip():
            treatments.append(prescription_notes.strip())
        return treatments

    # ─── Helper / Lookup Functions ───────────────────────────────────

    def _lookup_vital_id(self, name: str) -> Optional[str]:
        """Find the HMIS vitalId for a given vital sign name."""
        name_lower = name.lower().strip()

        # Direct match
        if name_lower in VITAL_ID_MAP:
            return VITAL_ID_MAP[name_lower]

        # Partial match
        for key, vital_id in VITAL_ID_MAP.items():
            if key in name_lower or name_lower in key:
                return vital_id

        return None

    def _convert_frequency(self, raw: str) -> str:
        """Convert frequency text to HMIS format (e.g. '1-1-1')."""
        if not raw:
            return "1-0-0"

        raw_lower = raw.strip().lower()

        # Direct match
        if raw_lower in FREQUENCY_MAP:
            return FREQUENCY_MAP[raw_lower]

        # Partial match
        for key, value in FREQUENCY_MAP.items():
            if key in raw_lower:
                return value

        # Already in HMIS format (e.g. "1-1-1")
        if re.match(r'^\d+(-\d+)+$', raw.strip()):
            return raw.strip()

        return "1-0-0"

    def _convert_timing(self, raw: str) -> str:
        """Convert timing text to HMIS format."""
        if not raw:
            return "Any Time"

        raw_lower = raw.strip().lower()

        # Direct match
        if raw_lower in TIMING_MAP:
            return TIMING_MAP[raw_lower]

        # Partial match
        for key, value in TIMING_MAP.items():
            if key in raw_lower:
                return value

        # Return original capitalized if no match
        return raw.strip().title() if raw.strip() else "Any Time"

    def _get_empty_hmis(self) -> Dict:
        """Return a minimal empty HMIS JSON structure."""
        return {
            "visitId": "",
            "patientId": "",
            "facilityId": "",
            "healthProfessionalId": "",
            "appointmentId": "",
            "visitTime": self.now.isoformat(),
            "doctorPrivateNote": "",
            "isConsultationEnded": False,
            "linkWithAbha": False,
            "additionalVitals": [],
            "medications": [],
            "cheifComplaints": [],
            "diagnosis": [],
            "diagnosticInvestigations": [],
            "labInvestigations": [],
            "advices": [],
            "medicalAdvices": [],
            "treatment": [],
            "extendedProps": [],
            "eyeExamination": {},
        }
