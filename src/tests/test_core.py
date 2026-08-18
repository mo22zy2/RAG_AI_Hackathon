"""
Unit tests for core RAG safety, retrieval quality, and citation mechanics.

Run from src/:
    pytest tests/test_core.py -v
"""
import sys
import os
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is on the path so controller imports resolve
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Bypass controllers/__init__.py which pulls in DataController -> fastapi
# by pre-populating the controllers package as a namespace module.
import types
if "controllers" not in sys.modules:
    sys.modules["controllers"] = types.ModuleType("controllers")
    sys.modules["controllers"].__path__ = [str(SRC_DIR / "controllers")]


# ---------------------------------------------------------------------------
# Helpers — create a minimal NLPController without a live DB / LLM
# ---------------------------------------------------------------------------

def _make_controller():
    from controllers.NLPController import NLPController
    ctrl = NLPController(
        vectordb_client=MagicMock(),
        generation_client=MagicMock(),
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
        rerank_client=MagicMock(),
    )
    return ctrl


# ===================================================================
# 1. Safety classifier — classify_input_risk
# ===================================================================

class TestSafetyClassifier:

    @pytest.fixture(autouse=True)
    def _ctrl(self):
        self.ctrl = _make_controller()

    def test_emergency_breathing(self):
        r = self.ctrl.classify_input_risk("I can't breathe, my lips are blue")
        assert r["risk_level"] == "refuse_redirect"
        assert r["reason"] == "possible_emergency"

    def test_emergency_ambulance(self):
        r = self.ctrl.classify_input_risk("Call an ambulance now")
        assert r["risk_level"] == "refuse_redirect"

    def test_emergency_arabic(self):
        r = self.ctrl.classify_input_risk("لا أستطيع التنفس")
        assert r["risk_level"] == "refuse_redirect"

    def test_personal_dosing(self):
        r = self.ctrl.classify_input_risk("Should I take more of my inhaler tonight?")
        assert r["risk_level"] == "refuse_redirect"
        assert r["reason"] == "personal_medication_advice"

    def test_stop_medication(self):
        r = self.ctrl.classify_input_risk("Can I stop my medication if I feel better?")
        assert r["risk_level"] == "refuse_redirect"

    def test_animal_question(self):
        r = self.ctrl.classify_input_risk("My cat is wheezing. Is it asthma?")
        assert r["risk_level"] == "refuse_redirect"
        assert r["reason"] == "animal_or_pet_question"

    def test_pet_allergen_trigger_is_in_scope(self):
        # A pet mentioned as an asthma *trigger* is a legitimate, in-scope
        # clinical question (guidelines cover allergen avoidance) — it
        # should not be refused as an out-of-scope veterinary question.
        r = self.ctrl.classify_input_risk("Can pet dander trigger an asthma exacerbation?")
        assert r["risk_level"] == "allowed"

    def test_dog_asthma_trigger_is_in_scope(self):
        r = self.ctrl.classify_input_risk("Is dog hair a known asthma trigger?")
        assert r["risk_level"] == "allowed"

    def test_programming_question(self):
        r = self.ctrl.classify_input_risk("Write me a Python function to sort a list.")
        assert r["risk_level"] == "refuse_redirect"
        assert r["reason"] == "clearly_non_clinical"

    def test_diabetes_out_of_scope(self):
        r = self.ctrl.classify_input_risk("What are the latest diabetes screening recommendations?")
        assert r["risk_level"] == "refuse_redirect"

    def test_hypertension_out_of_scope(self):
        r = self.ctrl.classify_input_risk("How should hypertension be treated in adults?")
        assert r["risk_level"] == "refuse_redirect"

    def test_personal_symptom_self(self):
        r = self.ctrl.classify_input_risk("I have coughing and wheezing. What does the guideline say?")
        assert r["risk_level"] == "needs_caution"

    def test_personal_symptom_family(self):
        r = self.ctrl.classify_input_risk("My son is wheezing at night. Does he have asthma?")
        assert r["risk_level"] == "needs_caution"

    def test_asthma_management(self):
        r = self.ctrl.classify_input_risk("What is the initial pharmacological treatment for newly diagnosed asthma?")
        assert r["risk_level"] == "allowed"

    def test_asthma_diagnosis(self):
        r = self.ctrl.classify_input_risk("How is asthma diagnosed in adults?")
        assert r["risk_level"] == "allowed"

    def test_mart_question(self):
        r = self.ctrl.classify_input_risk("What is MART therapy?")
        assert r["risk_level"] == "allowed"

    def test_empty_query(self):
        r = self.ctrl.classify_input_risk("")
        assert r["risk_level"] == "refuse_redirect"
        assert r["reason"] == "empty_query"

    def test_none_query(self):
        r = self.ctrl.classify_input_risk(None)
        assert r["risk_level"] == "refuse_redirect"


# ===================================================================
# 2. Confidence gate — _build_confidence
# ===================================================================

class TestConfidenceGate:

    @pytest.fixture(autouse=True)
    def _ctrl(self):
        self.ctrl = _make_controller()

    def _doc(self, score=0.8, doc_name="GINA_2026_Strategy_Report", page=10, org="gina"):
        return {
            "score": score,
            "metadata": {
                "document_name": doc_name,
                "page_number": page,
                "org": org,
                "source_url": "https://ginasthma.org/test.pdf",
            },
            "text": "Test chunk text about asthma",
        }

    def test_high_confidence(self):
        retrieved = [self._doc(0.85) for _ in range(4)]
        c = self.ctrl._build_confidence(retrieved, retrieved[:3])
        assert c["confidence_level"] == "high"
        assert c["generation_allowed"] is True

    def test_medium_confidence(self):
        retrieved = [self._doc(0.40) for _ in range(3)]
        with patch("controllers.NLPController.get_settings") as mock_settings:
            mock_settings.return_value.ANSWER_MIN_TOP_SCORE = 0.0
            mock_settings.return_value.ANSWER_MIN_EVIDENCE_COUNT = 1
            c = self.ctrl._build_confidence(retrieved, retrieved)
        assert c["confidence_level"] == "medium"
        assert c["generation_allowed"] is True

    def test_low_confidence(self):
        retrieved = [self._doc(0.15) for _ in range(3)]
        with patch("controllers.NLPController.get_settings") as mock_settings:
            mock_settings.return_value.ANSWER_MIN_TOP_SCORE = 0.0
            mock_settings.return_value.ANSWER_MIN_EVIDENCE_COUNT = 1
            c = self.ctrl._build_confidence(retrieved, retrieved)
        assert c["confidence_level"] == "low"
        assert c["generation_allowed"] is True

    def test_no_evidence_blocks_generation(self):
        c = self.ctrl._build_confidence([], [])
        assert c["generation_allowed"] is False
        assert c["confidence_level"] == "insufficient"

    def test_score_threshold_blocks(self):
        retrieved = [self._doc(0.30) for _ in range(3)]
        c = self.ctrl._build_confidence(retrieved, retrieved)
        assert c["generation_allowed"] is False
        assert c["confidence_level"] == "insufficient"

    def test_score_threshold_passes(self):
        retrieved = [self._doc(0.60) for _ in range(3)]
        c = self.ctrl._build_confidence(retrieved, retrieved)
        assert c["generation_allowed"] is True

    def test_non_official_docs_not_counted(self):
        retrieved = [self._doc(0.80, doc_name="SomeBlog", org="", page=10)]
        c = self.ctrl._build_confidence(retrieved, retrieved)
        assert c["evidence_count"] == 0
        assert c["generation_allowed"] is False


# ===================================================================
# 3. Citation verification — verify_citations
# ===================================================================

class TestCitationVerification:

    @pytest.fixture(autouse=True)
    def _ctrl(self):
        self.ctrl = _make_controller()

    def _sources(self):
        return [
            {"chunk_id": 1, "document_name": "GINA_2026_Strategy_Report",
             "page_number": 112, "text": "content about step-down therapy"},
            {"chunk_id": 2, "document_name": "NICE_NG80_Asthma",
             "page_number": 18, "text": "initial pharmacological treatment"},
        ]

    def test_single_citation(self):
        answer = "Use low-dose ICS [GINA_2026_Strategy_Report, p. 112]."
        cites = self.ctrl.verify_citations(answer, self._sources())
        assert len(cites) == 1
        assert cites[0]["document"] == "GINA_2026_Strategy_Report"
        assert cites[0]["page_number"] == 112
        assert cites[0]["supported"] is True

    def test_multiple_citations(self):
        answer = "Treatment [GINA_2026_Strategy_Report, p. 112] and diagnosis [NICE_NG80_Asthma, p. 18]."
        cites = self.ctrl.verify_citations(answer, self._sources())
        assert len(cites) == 2
        assert all(c["supported"] for c in cites)

    def test_unsupported_citation(self):
        answer = "Something [Unknown_Doc, p. 999]."
        cites = self.ctrl.verify_citations(answer, self._sources())
        assert len(cites) == 1
        assert cites[0]["supported"] is False

    def test_combined_citations_are_split(self):
        answer = "Evidence [GINA_2026_Strategy_Report, p. 112, NICE_NG80_Asthma, p. 18]."
        cites = self.ctrl.verify_citations(answer, self._sources())
        assert len(cites) >= 2
        docs = {c["document"] for c in cites}
        assert "GINA_2026_Strategy_Report" in docs
        assert "NICE_NG80_Asthma" in docs

    def test_no_citations_returns_empty(self):
        cites = self.ctrl.verify_citations("No citations here.", self._sources())
        assert cites == []


# ===================================================================
# 4. Query expansion — expand_query
# ===================================================================

class TestQueryExpansion:

    @pytest.fixture(autouse=True)
    def _ctrl(self):
        self.ctrl = _make_controller()

    def test_mart_expansion(self):
        result = self.ctrl.expand_query("What is MART therapy?")
        assert "maintenance-and-reliever" in result.lower() or "ics-formoterol" in result.lower()

    def test_saba_expansion(self):
        result = self.ctrl.expand_query("Why does GINA advise against SABA-only treatment?")
        assert "short-acting beta" in result.lower() or "salbutamol" in result.lower()

    def test_gina_expansion(self):
        result = self.ctrl.expand_query("What does GINA recommend?")
        assert "global strategy" in result.lower() or "2026" in result.lower()

    def test_nice_expansion(self):
        result = self.ctrl.expand_query("What does NICE say about asthma?")
        assert "ng80" in result.lower() or "diagnosis" in result.lower()

    def test_nhlbi_expansion(self):
        result = self.ctrl.expand_query("What does NHLBI recommend?")
        assert "naepp" in result.lower() or "epr-4" in result.lower() or "quick reference" in result.lower()

    def test_step_down_expansion(self):
        result = self.ctrl.expand_query("How should therapy be stepped down?")
        assert "de-escalat" in result.lower() or "stepping down" in result.lower()

    def test_non_pharmacological_expansion(self):
        result = self.ctrl.expand_query("What non-pharmacologic strategies are recommended?")
        assert "smoking cessation" in result.lower() or "physical activity" in result.lower()

    def test_risk_exacerbation_expansion(self):
        result = self.ctrl.expand_query("Which features identify patients at increased risk of exacerbations?")
        assert "risk factors" in result.lower() or "severe exacerbation" in result.lower()

    def test_no_match_returns_original(self):
        q = "What is the capital of France?"
        result = self.ctrl.expand_query(q)
        assert result == q

    def test_expansion_appended_to_original(self):
        result = self.ctrl.expand_query("What is MART?")
        assert result.startswith("What is MART?")
        assert "Expanded retrieval terms:" in result


# ===================================================================
# 5. Unsupported claim detection — detect_unsupported_claims
# ===================================================================

class TestUnsupportedClaims:

    @pytest.fixture(autouse=True)
    def _ctrl(self):
        self.ctrl = _make_controller()

    def _selected_docs(self):
        return [{
            "text": "Low-dose ICS is the preferred controller for Step 1.",
            "metadata": {
                "document_name": "GINA_2026_Strategy_Report",
                "page_number": 23,
                "org": "gina",
                "source_url": "https://ginasthma.org/test.pdf",
            },
        }]

    def test_supported_claim_no_unsupported(self):
        answer = "GINA recommends low-dose ICS [GINA_2026_Strategy_Report, p. 23]."
        cites = self.ctrl.verify_citations(answer, self._selected_docs())
        unsupported = self.ctrl.detect_unsupported_claims(answer, self._selected_docs(), cites)
        assert len(unsupported) == 0

    def test_unsupported_numeric_claim(self):
        answer = "Patients should take 500mg of ICS daily [GINA_2026_Strategy_Report, p. 23]."
        cites = self.ctrl.verify_citations(answer, self._selected_docs())
        unsupported = self.ctrl.detect_unsupported_claims(answer, self._selected_docs(), cites)
        assert len(unsupported) > 0


# ===================================================================
# 6. Answer quality builder — build_answer_quality
# ===================================================================

class TestAnswerQuality:

    @pytest.fixture(autouse=True)
    def _ctrl(self):
        self.ctrl = _make_controller()

    def test_empty_answer_quality(self):
        q = self.ctrl.build_answer_quality("", [], [], verify_claims=False)
        assert q["citation_faithfulness"] == 1.0
        assert q["unsupported_claim_rate"] == 0.0

    def test_cited_answer_quality(self):
        sources = [{"chunk_id": 1, "document_name": "GINA_2026_Strategy_Report",
                     "page_number": 23, "text": "Low-dose ICS preferred."}]
        selected = [{"text": "Low-dose ICS preferred.",
                     "metadata": {"document_name": "GINA_2026_Strategy_Report",
                                  "page_number": 23, "org": "gina",
                                  "source_url": "https://ginasthma.org/test.pdf"}}]
        answer = "GINA recommends low-dose ICS [GINA_2026_Strategy_Report, p. 23]."
        q = self.ctrl.build_answer_quality(answer, sources, selected, verify_claims=True)
        assert q["citation_faithfulness"] >= 0.5
        assert q["unsupported_claim_rate"] == 0.0


# ===================================================================
# 7. Cohere embedding provider — query vs. document input_type
# ===================================================================
# Regression test for a confirmed bug: `document_type` was compared against
# the DocumentType *Enum member* instead of its `.value`, so the branch was
# always False and every embed call — including queries — used
# input_type="search_document". Cohere's embed models are asymmetric, so
# this silently degraded retrieval whenever EMBEDDING_BACKEND=COHERE.

class TestCoHereEmbedInputType:

    def _provider_with_fake_client(self):
        from stores.llm.providers.CoHereProvider import CoHereProvider

        provider = CoHereProvider(api_key="test-key")
        provider.set_embedding_model(model_id="embed-test", embedding_size=4)

        captured = {}

        class _Embeddings:
            float = [[0.1, 0.2, 0.3, 0.4]]

        class _Response:
            embeddings = _Embeddings()

        async def fake_embed(**kwargs):
            captured.update(kwargs)
            return _Response()

        provider.client = MagicMock()
        provider.client.embed = fake_embed
        return provider, captured

    def test_query_document_type_uses_search_query(self):
        import asyncio
        from stores.llm.LLMEnums import DocumentType

        provider, captured = self._provider_with_fake_client()
        asyncio.run(provider.embed_text("what is asthma?", document_type=DocumentType.QUERY.value))
        assert captured["input_type"] == "search_query"

    def test_document_document_type_uses_search_document(self):
        import asyncio
        from stores.llm.LLMEnums import DocumentType

        provider, captured = self._provider_with_fake_client()
        asyncio.run(provider.embed_text("chunk text", document_type=DocumentType.DOCUMENT.value))
        assert captured["input_type"] == "search_document"


# ===================================================================
# 8. PGVectorProvider — collection-name identifier validation
# ===================================================================
# Table/collection names can't be bound as SQL parameters, so PGVectorProvider
# validates them against a strict regex before interpolating them into SQL.
# This is the one identifier-interpolation path in the codebase; it's worth
# a standing regression test.

class TestPGVectorIdentifierValidation:

    def test_accepts_valid_collection_name(self):
        from stores.vectordb.providers.PGVectorProvider import PGVectorProvider
        assert PGVectorProvider._validate_collection_name("collection_1024_1") == "collection_1024_1"

    def test_rejects_sql_metacharacters(self):
        from stores.vectordb.providers.PGVectorProvider import PGVectorProvider
        with pytest.raises(ValueError):
            PGVectorProvider._validate_collection_name("collection_1; DROP TABLE chunks;--")

    def test_rejects_leading_digit(self):
        from stores.vectordb.providers.PGVectorProvider import PGVectorProvider
        with pytest.raises(ValueError):
            PGVectorProvider._validate_collection_name("1_collection")
