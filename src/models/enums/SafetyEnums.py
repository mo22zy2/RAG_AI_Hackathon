from enum import Enum


class RiskLevel(str, Enum):
    ALLOWED = "allowed"
    NEEDS_CAUTION = "needs_caution"
    REFUSE_REDIRECT = "refuse_redirect"


class SafetyReason(str, Enum):
    EMPTY_QUERY = "empty_query"
    POSSIBLE_EMERGENCY = "possible_emergency"
    PERSONAL_MEDICATION_ADVICE = "personal_medication_advice"
    ANIMAL_OR_PET_QUESTION = "animal_or_pet_question"
    CLEARLY_NON_CLINICAL = "clearly_non_clinical"
    OUTSIDE_ASTHMA_SCOPE = "outside_asthma_scope"
    PATIENT_SPECIFIC_SCENARIO = "patient_specific_scenario"
    WITHIN_ASTHMA_GUIDELINE_SCOPE = "within_asthma_guideline_scope"
    SCOPE_UNDETERMINED_DEFERRING_TO_RETRIEVAL = "scope_undetermined_deferring_to_retrieval"
    BLOCKED_BY_SAFETY_CLASSIFIER = "blocked_by_safety_classifier"
    INSUFFICIENT_OFFICIAL_EVIDENCE = "insufficient_official_evidence"
