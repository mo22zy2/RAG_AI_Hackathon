from typing import List
import re

from models.db_schemas import DataChunk
from stores.llm.LLMEnums import DocumentType
from helpers.config import get_settings
from helpers.safety_config import get_safety_config, compile_rule_pattern, config_signature
from models.enums.SafetyEnums import RiskLevel, SafetyReason

from .BaseController import BaseController
from models.db_schemas import Project
import json
DOC_HEADER_OVERHEAD_CHARS = 220
FOOTER_RESERVE_CHARS = 900 
class NLPController(BaseController):
    
    
    
    def __init__(self, vectordb_client,generation_client,embedding_client,template_parser, rerank_client=None):
        super().__init__()
        self.vectordb_client=vectordb_client
        self.generation_client=generation_client
        self.embedding_client=embedding_client
        self.template_parser=template_parser
        self.rerank_client=rerank_client
        
        
    def create_collection_name(self,project_id:int):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()
    
    async def reset_vector_db_collection(self,project:Project):
        collection_name=self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name)
    
    async def get_vector_db_collection_info(self ,project:Project):
        collection_name=self.create_collection_name(project_id=project.project_id)
        collection_info=await self.vectordb_client.get_collection_info(collection_name)
        
        return json.loads(
            json.dumps(collection_info,default=lambda x: x.__dict__)
        )
    
    async def index_into_vector_db(self,project:Project,
                             chunks:List[DataChunk],
                             chunk_ids:List[int],
                             do_reset:bool=False
                             ):
        collection_name=self.create_collection_name(project_id=project.project_id)
        filtered_chunks=[c for c in chunks if c.chunk_text and len(c.chunk_text.strip()) > 1]
        texts=[c.chunk_text.strip() for c in filtered_chunks]
        metadata=[self._parse_chunk_metadata(c.chunk_metadata) for c in filtered_chunks]
        record_ids=[c.chunk_id for c in filtered_chunks]
        settings=get_settings()

        if not texts:
            return True

        vectors=[]
        for i in range(0, len(texts), settings.EMBEDDING_BATCH_SIZE):
            batch_vectors=await self.embedding_client.embed_text(
                text=texts[i:i + settings.EMBEDDING_BATCH_SIZE],
                document_type=DocumentType.DOCUMENT.value
            )
            if not batch_vectors:
                return False
            vectors.extend(batch_vectors)

        if len(vectors) != len(texts):
            return False

        
        _= await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
            
            
        )
        _= await self.vectordb_client.insert_many(collection_name=collection_name,
                                            texts=texts,
                                            metadata= metadata,
                                            vector=vectors,
                                            record_ids=record_ids,
                                            batch_size=settings.VECTOR_INSERT_BATCH_SIZE
                                            )
        
        
        return True
    
    
    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10,
                                          score_threshold: float = None,
                                          metadata_filter: dict = None,
                                          retrieval_mode: str = "hybrid",
                                          rerank: bool = False,
                                          rerank_top_k: int = None,
                                          expand_query: bool = True):
        """
        Retrieve chunks from the project collection.

        retrieval_mode: "vector" | "keyword" | "hybrid"
        rerank: if True, fetch retrieval_mode candidates (top RETRIEVAL_TOP_K)
                then reorder with the rerank_client and trim to `limit`.
        Returns a JSON-safe list of dicts.
        """
        query_vector = None
        collection_name = self.create_collection_name(project_id=project.project_id)
        expanded_query = self.expand_query(text) if expand_query else text
        # BM25/keyword search uses the RAW query to avoid precision dilution
        # from the many OR'd expanded terms; vector embedding uses the full
        # expanded query for richer semantic context.
        raw_query = text

        vectors = await self.embedding_client.embed_text(
            text=expanded_query,
            document_type=DocumentType.QUERY.value
        )

        if vectors and isinstance(vectors, list) and len(vectors) > 0:
            query_vector = vectors[0]

        if retrieval_mode == "keyword":
            results = await self.vectordb_client.search_by_keyword(
                collection_name=collection_name,
                query=raw_query,
                limit=limit,
                metadata_filter=metadata_filter,
            )
        elif retrieval_mode == "vector":
            if query_vector is None:
                return None
            results = await self.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter,
            )
        else:  # hybrid
            if query_vector is None:
                return None
            results = await self.vectordb_client.search_hybrid(
                collection_name=collection_name,
                vector=query_vector,
                query=raw_query,
                limit=limit,
                metadata_filter=metadata_filter,
                rrf_k=get_settings().HYBRID_RRF_K,
            )

        if not results:
            return [] if results == [] else None

        if rerank and self.rerank_client and query_vector is not None:
            settings = get_settings()
            candidate_k = max(limit, settings.RETRIEVAL_TOP_K)
            rerank_limit = rerank_top_k or settings.RERANK_TOP_K

            candidates = await self._build_rerank_candidates(
                collection_name=collection_name,
                query=raw_query,
                vector=query_vector,
                limit=candidate_k,
                metadata_filter=metadata_filter,
                score_threshold=score_threshold,
                fallback_results=results,
            )

            reranked = await self.rerank_client.rerank(
                query=expanded_query,
                documents=candidates or results,
                top_n=rerank_limit,
            )
            results = reranked[:limit]

        return json.loads(
            json.dumps(results, default=lambda x: x.__dict__)
        )
        
        
    async def answer_rag_question(self,project:Project,query:str,limit:int =5,
                                  score_threshold:float=None,
                                  metadata_filter:dict=None,
                                  include_sources:bool=True,
                                  retrieval_mode:str="hybrid",
                                  rerank:bool=True,
                                  rerank_top_k:int=None,
                                  expand_query:bool=True,
                                  verify_claims:bool=True,
                                  conversation_history:list=None):

        answer , full_prompt , chat_history=None,None,None
        settings=get_settings()
        conversation_block=self._build_conversation_block(conversation_history)
        retrieval_query=self._build_retrieval_query(query, conversation_history)
        risk_assessment=self.classify_input_risk(query)
        refusal_answer=self._build_refusal_answer(risk_assessment)
        disclaimer=self._clinical_disclaimer()

        if risk_assessment["risk_level"] == "refuse_redirect":
            confidence=self._build_confidence([], [], question=query)
            confidence["generation_allowed"]=False
            confidence["reason"]="blocked_by_safety_classifier"
            quality=self.build_answer_quality(refusal_answer, [], [], verify_claims=False)
            evidence_panel = {"total_retrieved": 0, "total_selected": 0, "retrieval_coverage": {"documents": [], "unique_documents": 0, "page_range": {}}, "chunks": []}
            return refusal_answer, None, None, [], risk_assessment, confidence, quality, disclaimer, evidence_panel
        
        retrived_document= await self.search_vector_db_collection(
            project=project,
            text=retrieval_query,
            limit=max(limit, settings.RETRIEVAL_TOP_K),
            score_threshold=score_threshold if score_threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD,
            metadata_filter=metadata_filter,
            retrieval_mode=retrieval_mode,
            rerank=rerank,
            rerank_top_k=rerank_top_k or settings.RERANK_TOP_K,
            expand_query=expand_query,
        )
        
        if not retrived_document or len(retrived_document)==0:
            confidence=self._build_confidence([], [], question=query)
            quality=self.build_answer_quality("", [], [], verify_claims=False)
            evidence_panel = {"total_retrieved": 0, "total_selected": 0, "retrieval_coverage": {"documents": [], "unique_documents": 0, "page_range": {}}, "chunks": []}
            return "", None, None, [], risk_assessment, confidence, quality, disclaimer, evidence_panel
        
        
        
        system_prompt=self.template_parser.get("rag","system_prompt")
        
        
        reserved_chars = FOOTER_RESERVE_CHARS + len(query) + len(conversation_block) + 50
        selected_documents=self._select_documents_for_prompt(
            retrived_document=retrived_document,
            max_documents=limit or settings.ANSWER_TOP_K,
            max_context_chars=settings.MAX_CONTEXT_CHARS,
            reserved_chars=reserved_chars,
        )
        confidence=self._build_confidence(retrived_document, selected_documents, question=query)

        if not confidence["generation_allowed"]:
            refusals = (get_safety_config().get("refusals") or {})
            refusal = refusals.get(
                "insufficient_official_evidence",
                (
                    "I do not have enough official guideline evidence to answer this safely. "
                    "Please consult a qualified healthcare professional or ask a question "
                    "within the indexed asthma guideline scope."
                ),
            )
            sources=self._build_sources(selected_documents)
            quality=self.build_answer_quality(refusal, sources, selected_documents, verify_claims=False)
            evidence_panel = self.build_evidence_panel(retrived_document, selected_documents)
            return refusal, None, None, sources, risk_assessment, confidence, quality, disclaimer, evidence_panel

        documnets_prompts="\n".join([
                self._render_document_prompt(idx, doc)
            for idx,doc in enumerate(selected_documents)
        ])
        
        footer_prompt=self.template_parser.get("rag","footer_prompt")

        if risk_assessment["risk_level"] == "needs_caution":
            footer_prompt = (
                "SAFETY NOTE: The user appears to be asking about a specific person's "
                "symptoms. Your answer MUST begin by clearly stating that you cannot "
                "provide a diagnosis or prescribe medication without a clinical "
                "assessment, and that they should consult a qualified healthcare "
                "professional. Then you may share the relevant general guideline "
                "information from the documents above, with citations.\n\n"
                + footer_prompt
            )
        
        chat_history = [
        self.generation_client.construct_prompt(
            prompt=system_prompt,
            role=self.generation_client.enums.SYSTEM.value,
                        )
                    ]

        full_prompt = "\n\n".join(filter(None, [
            documnets_prompts,
            conversation_block,
            f"## User Question:\n{query}",
            footer_prompt,
        ]))

        answer=await self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        if risk_assessment["risk_level"] == "needs_caution" and answer:
            answer = (
                "I can't diagnose this person or prescribe medication without a "
                "clinical assessment. Please consult a qualified healthcare "
                "professional. Here is general information from the guidelines:\n\n"
            ) + answer

        sources=self._build_sources(selected_documents) if include_sources else []
        quality=self.build_answer_quality(answer or "", sources, selected_documents, verify_claims=verify_claims)

        max_regenerations = get_settings().ANSWER_MAX_VERIFICATION_REGENERATIONS
        if verify_claims and answer and max_regenerations > 0:
            for attempt in range(1, max_regenerations + 1):
                if quality["citation_faithfulness"] >= 1.0 and quality["unsupported_claim_rate"] == 0.0:
                    break

                correction_footer = (
                    "IMPORTANT: Your previous answer draft failed automated verification: "
                    "every factual claim must carry the citation [Document Name, p. PAGE] using the "
                    "EXACT document names from the '## Document Name' headers above. "
                    "Rewrite the answer from scratch, using ONLY the documents above, cite every claim, "
                    "and do not add pleasantries or restate the question."
                )
                chat_history = [
                    self.generation_client.construct_prompt(
                        prompt=system_prompt,
                        role=self.generation_client.enums.SYSTEM.value,
                    )
                ]
                full_prompt = "\n\n".join(filter(None, [
                    documnets_prompts,
                    conversation_block,
                    f"## User Question:\n{query}",
                    correction_footer,
                ]))
                retry_answer = await self.generation_client.generate_text(
                    prompt=full_prompt,
                    chat_history=chat_history
                )
                if not retry_answer:
                    break
                answer = retry_answer
                if risk_assessment["risk_level"] == "needs_caution":
                    answer = (
                        "I can't diagnose this person or prescribe medication without a "
                        "clinical assessment. Please consult a qualified healthcare "
                        "professional. Here is general information from the guidelines:\n\n"
                    ) + answer
                quality=self.build_answer_quality(answer, sources, selected_documents, verify_claims=verify_claims)

        evidence_panel = self.build_evidence_panel(retrived_document, selected_documents)
        return answer , full_prompt , chat_history, sources, risk_assessment, confidence, quality, disclaimer, evidence_panel

    CONVERSATION_CONTEXT_TURNS = 3
    CONVERSATION_ANSWER_CHARS = 400

    def _recent_conversation_turns(self, conversation_history: list):
        if not conversation_history:
            return []
        return [
            turn for turn in conversation_history[-self.CONVERSATION_CONTEXT_TURNS:]
            if (turn.get("question") or "").strip()
        ]

    def _build_retrieval_query(self, query: str, conversation_history: list):
        """Follow-up questions ("what about someone older?") carry no topic
        of their own, so prior turns are folded into the retrieval text —
        otherwise the vector/keyword search has nothing asthma-related to
        match against."""
        turns = self._recent_conversation_turns(conversation_history)
        if not turns:
            return query

        context = " ".join((turn.get("question") or "").strip() for turn in turns)
        return f"{context} {query}".strip()

    def _build_conversation_block(self, conversation_history: list):
        turns = self._recent_conversation_turns(conversation_history)
        if not turns:
            return ""

        lines = ["## Conversation so far (for context on follow-up questions):"]
        for turn in turns:
            question = (turn.get("question") or "").strip()
            answer = (turn.get("answer") or "").strip()
            lines.append(f"User: {question}")
            if answer:
                lines.append(f"Assistant: {answer[:self.CONVERSATION_ANSWER_CHARS]}")
        lines.append(
            "The user's new question below may be a follow-up referring back to this "
            "conversation (e.g. a pronoun, an age group, or an implied topic) — resolve "
            "that reference using the conversation above, then answer strictly from the "
            "documents."
        )
        return "\n".join(lines)

    def _parse_chunk_metadata(self, metadata):
        if not metadata:
            return {}

        if isinstance(metadata, dict):
            return metadata

        try:
            return json.loads(metadata)
        except Exception:
            return {}

    def expand_query(self, query: str):
        text=query or ""
        lowered=text.lower()
        expansions=[]
        rules=[
            (r"\bmart\b|maintenance[-\s]?and[-\s]?reliever", "maintenance-and-reliever therapy ICS-formoterol budesonide formoterol as-needed single-inhaler combination"),
            (r"\bair\b|anti[-\s]?inflammatory reliever", "anti-inflammatory reliever low dose ICS-formoterol as-needed"),
            (r"\bics\b|inhaled corticosteroid", "inhaled corticosteroid controller preventer corticosteroid"),
            (r"\bsaba\b|short[-\s]?acting beta", "short-acting beta agonist reliever salbutamol albuterol SABA-only"),
            (r"\blaba\b|long[-\s]?acting beta", "long-acting beta agonist formoterol salmeterol"),
            (r"\baction plan\b|self[-\s]?management", "written asthma action plan personalised action plan monitor symptoms seek medical care"),
            (r"\bobesity\b|overweight|weight", "weight reduction 5-10% 5–10% improve asthma control obesity comorbidity"),
            (r"\binitial\b|newly diagnosed|first treatment", "initial pharmacological treatment low-dose ICS/formoterol as-needed AIR therapy"),
            (r"\bstep 1\b|step one", "Step 1 preferred treatment adults adolescents low dose ICS-formoterol"),
            (r"\bdiagnos", "confirm asthma diagnosis spirometry bronchodilator reversibility FeNO eosinophil variability"),
            (r"\bexercise\b|bronchospasm", "exercise-induced bronchospasm before exercise SABA LTRA cromolyn NHLBI"),
            (r"\bgina\b", "GINA 2026 Global Strategy for Asthma Management and Prevention GINA 2026 Summary Guide for Asthma Management and Prevention"),
            (r"\bnice\b|ng80|bts", "NICE Asthma: diagnosis monitoring and chronic asthma management NG80"),
            (r"\bnhlbi\b|naepp|epr4|epr[-\s]?4", "Asthma Quick Reference Guide 2020 NAEPP EPR-4 focused updates NHLBI"),
            (r"step[-\s]?down|stepped?\s+down|de-?escalat|reduc\w*\s+(treatment|therapy|dose)", 
            "stepping down maintenance therapy reduce controller dose de-escalation well-controlled asthma minimum effective treatment"),
            (r"non[-\s]?pharmacolog|smoking cessation|vaccination|immunisation|obesity|weight reduction",
            "non-pharmacological strategies smoking cessation physical activity weight reduction vaccination obesity self-management GINA guideline recommendations"),
            (r"risk.*(exacerb|poor outcome|severe)|increased risk|identify.*risk|at risk",
            "risk factors exacerbations severe exacerbation poor lung function FEV1 intubation past year emergency visit"),
            (r"preferred treatment|step.?1.*persistent|persistent.*step.?1",
            "preferred treatment Step 1 persistent asthma low-dose ICS SABA as needed NHLBI stepwise approach"),
        ]

        for pattern, expansion in rules:
            if re.search(pattern, lowered, re.IGNORECASE):
                expansions.append(expansion)

        if not expansions:
            return text

        expanded_terms=" ".join(expansions)
        return f"{text}\n\nExpanded retrieval terms: {expanded_terms}"

    async def _build_rerank_candidates(self, collection_name: str, query: str, vector: list,
                                       limit: int, metadata_filter: dict = None,
                                       score_threshold: float = None,
                                       fallback_results: list = None):
        settings=get_settings()
        candidate_groups=[]

        if fallback_results:
            candidate_groups.append(fallback_results)

        try:
            vector_results=await self.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=vector,
                limit=limit,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter,
            )
            if vector_results:
                candidate_groups.append(vector_results)
        except NotImplementedError:
            pass

        try:
            keyword_results=await self.vectordb_client.search_by_keyword(
                collection_name=collection_name,
                query=query,
                limit=limit,
                metadata_filter=metadata_filter,
            )
            if keyword_results:
                candidate_groups.append(keyword_results)
        except NotImplementedError:
            pass

        try:
            hybrid_results=await self.vectordb_client.search_hybrid(
                collection_name=collection_name,
                vector=vector,
                query=query,
                limit=limit,
                metadata_filter=metadata_filter,
                rrf_k=settings.HYBRID_RRF_K,
            )
            if hybrid_results:
                candidate_groups.append(hybrid_results)
        except NotImplementedError:
            pass

        return self._dedupe_documents(candidate_groups)[:limit]

    @staticmethod
    def _dedupe_documents(candidate_groups: list):
        candidates=[]
        seen=set()

        for group in candidate_groups:
            for doc in group or []:
                if isinstance(doc, dict):
                    chunk_id=doc.get("chunk_id")
                    text=doc.get("text")
                else:
                    chunk_id=getattr(doc, "chunk_id", None)
                    text=getattr(doc, "text", None)
                key=chunk_id or text
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(doc)

        return candidates

    def _is_personal_symptom_query(self, query: str) -> bool:
        """Rule-based guardrail: detect when the user asks about the symptoms of
        a specific person (self or family) so the answer can refuse to diagnose.
        Patterns are sourced from safety_config.json (patterns.*)."""
        text = query or ""
        patterns = (get_safety_config().get("patterns") or {})

        person_ref = self._compiled_named_pattern("person_ref", patterns)
        first_person = self._compiled_named_pattern("first_person", patterns)
        strong_symptom = self._compiled_named_pattern("strong_symptom", patterns)
        health_topic = self._compiled_named_pattern("health_topic", patterns)

        has_person = bool(person_ref.search(text))
        has_first = bool(first_person.search(text))
        has_strong = bool(strong_symptom.search(text))
        has_topic = bool(health_topic.search(text))

        return (has_person and has_topic) or (has_first and has_strong)

    _ANIMAL_TERMS = ("dog", "cat", "pet", "animal", "horse", "bird")

    def _is_animal_patient_query(self, query: str) -> bool:
        """True when the query asks about an animal's own health (animal as
        patient), e.g. 'My cat is wheezing'.  Returns False for questions
        where the animal is an allergen/trigger context, e.g. 'Can pet
        dander trigger asthma?' — those are in-scope clinical questions."""
        text = (query or "").lower()
        has_animal = any(
            re.search(r"\b" + term + r"\b", text) for term in self._ANIMAL_TERMS
        )
        if not has_animal:
            return False
        possessives = ("my ", "our ", "his ", "her ")
        return any(p + a in text for p in possessives for a in self._ANIMAL_TERMS)

    _compiled_cache = {}

    def _compiled_named_pattern(self, name: str, patterns: dict):
        entry = patterns.get(name)
        if not entry:
            return re.compile(r"(?!)")
        cache_key = (name, self._config_signature())
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]
        compiled = compile_rule_pattern(entry.get("patterns"))
        self._compiled_cache[cache_key] = compiled
        return compiled

    @staticmethod
    def _config_signature():
        return config_signature()

    def _compiled_rule_pattern(self, name: str, rule: dict):
        patterns = rule.get("patterns")
        if not patterns:
            return None
        cache_key = (name, self._config_signature())
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]
        compiled = compile_rule_pattern(patterns)
        self._compiled_cache[cache_key] = compiled
        return compiled

    def _rule_matches(self, rule: dict, lowered: str, text: str) -> bool:
        pattern = self._compiled_rule_pattern(rule.get("name"), rule)
        if pattern is not None and not pattern.search(lowered):
            return False
        patterns = (get_safety_config().get("patterns") or {})
        for veto_name in rule.get("when_not") or []:
            if self._compiled_named_pattern(veto_name, patterns).search(lowered):
                return False
        for require_name in rule.get("require") or []:
            if not self._compiled_named_pattern(require_name, patterns).search(lowered):
                return False
        if rule.get("auxiliary"):
            handler = getattr(self, rule["auxiliary"], None)
            if handler is None or not handler(text):
                return False
        return True

    def classify_input_risk(self, query: str):
        text = (query or "").strip()
        lowered = text.lower()

        if not text:
            return self._risk(RiskLevel.REFUSE_REDIRECT.value, SafetyReason.EMPTY_QUERY.value)

        # Rules are evaluated in order; the first match wins. Adding a new rule
        # (or keywords/patterns to an existing one) is done in
        # src/config/safety_config.json, no code changes required.
        config = get_safety_config()
        for rule in config.get("risk_rules") or []:
            if self._rule_matches(rule, lowered, text):
                return self._risk(rule.get("risk_level", RiskLevel.ALLOWED.value), rule.get("reason"))

        return self._risk(RiskLevel.ALLOWED.value, SafetyReason.WITHIN_ASTHMA_GUIDELINE_SCOPE.value)

    @staticmethod
    def _risk(risk_level: str, reason: str):
        return {
            "risk_level": risk_level,
            "reason": reason,
        }

    def _build_refusal_answer(self, risk_assessment: dict):
        config = get_safety_config()
        refusals = config.get("refusals") or {}
        default = config.get("refusal_default") or (
            "I can only answer clinical questions that are within the indexed asthma "
            "guideline scope and supported by retrieved evidence."
        )
        return refusals.get(risk_assessment.get("reason"), default)

    def _build_confidence(self, retrieved_documents: list, selected_documents: list, question: str = ""):
        settings=get_settings()
        scores=[
            doc.get("score")
            for doc in retrieved_documents or []
            if isinstance(doc.get("score"), (int, float))
        ]
        top_score=max(scores) if scores else None
        evidence_count=sum(
            1 for doc in selected_documents or []
            if self._has_official_evidence_metadata(doc)
        )
        generation_allowed=evidence_count >= settings.ANSWER_MIN_EVIDENCE_COUNT

        if (
            settings.ANSWER_MIN_TOP_SCORE > 0
            and top_score is not None
            and top_score < settings.ANSWER_MIN_TOP_SCORE
        ):
            generation_allowed=False

        # Dosing-query guardrail: numeric, unit, or frequency questions are
        # blocked unless the retrieved evidence carries numeric content, so we
        # never answer with a made-up dose.
        if generation_allowed and self._is_numeric_or_dosing_query(question):
            evidence_numeric=any(
                self._numeric_values(" ".join(
                    str(part) for part in [
                        doc.get("text"),
                        (doc.get("metadata") or {}).get("excerpt") or (doc.get("metadata") or {}).get("snippet") or "",
                    ] if part
                ))
                for doc in selected_documents or []
            )
            if not evidence_numeric:
                generation_allowed=False

        if not generation_allowed:
            level="insufficient"
            reason="insufficient_official_evidence"
        elif top_score is None:
            level="medium" if evidence_count >= 2 else "low"
            reason="no_numeric_score"
        elif top_score >= 0.65:
            level="high"
            reason="strong_retrieval_score"
        elif top_score >= 0.30:
            level="medium"
            reason="moderate_retrieval_score"
        else:
            level="low"
            reason="weak_retrieval_score"

        return {
            "confidence_level": level,
            "top_score": top_score,
            "evidence_count": evidence_count,
            "generation_allowed": generation_allowed,
            "reason": reason,
        }

    @staticmethod
    def _build_confidence_summary(confidence: dict):
        if not confidence.get("generation_allowed"):
            return "Insufficient — generation blocked due to insufficient official evidence"
        
        level = confidence.get("confidence_level", "low").capitalize()
        score = confidence.get("top_score")
        score_str = f"{score:.2f}" if score is not None else "N/A"
        reason = confidence.get("reason", "").replace("_", " ")
        evidence_count = confidence.get("evidence_count", 0)
        
        return f"{level} — {reason} ({score_str}), {evidence_count} official evidence chunks"

    @staticmethod
    def _has_official_evidence_metadata(doc: dict):
        metadata=doc.get("metadata") or {}
        document_name=(metadata.get("document_name") or "").lower()
        org=(metadata.get("org") or "").lower()
        source_url=(metadata.get("source_url") or "").lower()
        page_number=metadata.get("page_number")
        official_markers=("nice", "gina", "nhlbi", "naepp", "who", "cdc", "uspstf")
        _official_re=re.compile(r'(?:^|[\s_/]|\.)(?:' + '|'.join(re.escape(m) for m in official_markers) + r')(?:[\s_/]|\.|$)', re.IGNORECASE)
        combined=" ".join([document_name, org, source_url])
        has_source=bool(_official_re.search(combined))
        return has_source and page_number not in (None, "", 0)

    def _render_document_prompt(self, idx:int, doc:dict):
        metadata=doc.get("metadata") or {}
        return self.template_parser.get("rag","document_prompt",{
            "doc_num": idx+1,
            "doc_name": metadata.get("document_name") or metadata.get("file_name") or "Unknown",
            "page_number": metadata.get("page_number") or "",
            "section_title": metadata.get("section_title") or "",
            "source_url": metadata.get("source_url") or "",
            "chunk_text": self.generation_client.process_text(doc.get("text") or ""),
        })

    def _select_documents_for_prompt(self, retrived_document: list, max_documents: int,
                                    max_context_chars: int, reserved_chars: int = 0):
        budget = max_context_chars - reserved_chars
        selected_documents = []
        used_context_chars = 0
        seen_documents = set()

        for doc in retrived_document:
            metadata = doc.get("metadata") or {}
            dedupe_key = (metadata.get("asset_id"), metadata.get("chunk_order"), doc.get("text"))
            if dedupe_key in seen_documents:
                continue

            text = doc.get("text") or ""
            next_context_size = used_context_chars + len(text) + DOC_HEADER_OVERHEAD_CHARS

            if selected_documents and next_context_size > budget:
                continue

            seen_documents.add(dedupe_key)
            selected_documents.append(doc)
            used_context_chars += len(text) + DOC_HEADER_OVERHEAD_CHARS

            if len(selected_documents) >= max_documents:
                break

        return selected_documents
    def _build_sources(self, selected_documents:list):
        sources=[]

        for idx, doc in enumerate(selected_documents):
            metadata=doc.get("metadata") or {}
            text=doc.get("text") or ""
            sources.append({
                "doc_num": idx + 1,
                "chunk_id": doc.get("chunk_id"),
                "file_name": metadata.get("file_name") or metadata.get("file_id"),
                "document_name": metadata.get("document_name") or metadata.get("file_name"),
                "asset_id": metadata.get("asset_id"),
                "chunk_order": metadata.get("chunk_order"),
                "page_number": metadata.get("page_number"),
                "section_title": metadata.get("section_title"),
                "source_url": metadata.get("source_url"),
                "score": doc.get("score"),
                "text": text[:300]
            })

        return sources

    def build_evidence_panel(self, retrieved_documents: list, selected_documents: list):
        retrieved_count = len(retrieved_documents) if retrieved_documents else 0
        selected_count = len(selected_documents) if selected_documents else 0
        
        unique_docs = set()
        pages = []
        chunks = []
        
        for idx, doc in enumerate(selected_documents or []):
            metadata = doc.get("metadata") or {}
            doc_name = metadata.get("document_name") or metadata.get("file_name") or "Unknown"
            page_num = metadata.get("page_number")
            
            unique_docs.add(doc_name)
            if page_num is not None:
                try:
                    pages.append(int(page_num))
                except ValueError:
                    pass
                
            chunks.append({
                "rank": idx + 1,
                "chunk_id": doc.get("chunk_id"),
                "relevance_score": doc.get("score"),
                "document_name": doc_name,
                "page_number": page_num,
                "section_title": metadata.get("section_title") or "",
                "source_url": metadata.get("source_url") or "",
                "text_preview": (doc.get("text") or "")[:300],
                "is_official_source": self._has_official_evidence_metadata(doc)
            })
            
        page_range = {}
        if pages:
            page_range = {"min": min(pages), "max": max(pages)}
            
        return {
            "total_retrieved": retrieved_count,
            "total_selected": selected_count,
            "retrieval_coverage": {
                "documents": list(unique_docs),
                "unique_documents": len(unique_docs),
                "page_range": page_range
            },
            "chunks": chunks
        }

    def verify_citations(self, answer: str, sources: list):
        citations=[]
        citation_pattern=re.compile(
            r"[\[\(](?P<document>[^\[\]()]+?)[,:]\s*(?:pp?\.?|pages?|pg\.?)\s*"
            r"(?P<page>\d+)(?:\s*[-,]\s*\d+)?(?:,\s*(?P<section>[^\[\]]+?))?[\]\)]",
            re.IGNORECASE,
        )

        # Pre-process: split combined citations like [Doc1, p. 1, Doc2, p. 2]
        # into separate [Doc1, p. 1] and [Doc2, p. 2] brackets.
        def _split_combined_citations(text):
            def _split_match(m):
                inner = m.group(0)[1:-1]  # strip [ ]
                # Split on document boundaries: look for known doc names followed by page refs
                parts = re.split(r'(?=(?:GINA|NICE|NHLBI|WHO|CDC|USPSTF)(?:[_\s]|$))', inner)
                parts = [p.strip().strip(',').strip() for p in parts if p.strip().strip(',').strip()]
                if len(parts) <= 1:
                    return m.group(0)
                return ']['.join(f'[{p}]' for p in parts)
            return re.sub(r'\[[^\[\]]{20,}\]', _split_match, text)

        processed_answer = _split_combined_citations(answer or "")

        for match in citation_pattern.finditer(processed_answer):
            document=match.group("document").strip()
            page=int(match.group("page"))
            section=(match.group("section") or "").strip()
            matched_source=self._find_cited_source(document, page, section, sources)
            citations.append({
                "citation": match.group(0),
                "document": document,
                "page_number": page,
                "section_title": section,
                "supported": matched_source is not None,
                "source_chunk_id": matched_source.get("chunk_id") if matched_source else None,
            })

        return citations

    def build_answer_quality(self, answer: str, sources: list, selected_documents: list,
                             verify_claims: bool = True):
        citations=self.verify_citations(answer, sources)
        citation_total=len(citations)
        supported_total=sum(1 for citation in citations if citation.get("supported"))
        citation_faithfulness=(
            supported_total / citation_total
            if citation_total > 0
            else 1.0 if (not answer or not verify_claims) else 0.0
        )
        unsupported_claims=(
            self.detect_unsupported_claims(answer, selected_documents, citations)
            if verify_claims
            else []
        )
        claims_checked=len(self._extract_claims(answer)) if verify_claims else 0
        unsupported_claim_rate=(
            len(unsupported_claims) / claims_checked
            if claims_checked > 0
            else 0.0
        )

        return {
            "citations": citations,
            "citation_faithfulness": round(citation_faithfulness, 3),
            "unsupported_claims": unsupported_claims,
            "unsupported_claim_rate": round(unsupported_claim_rate, 3),
            "claims_checked": claims_checked,
        }

    def detect_unsupported_claims(self, answer: str, selected_documents: list, citations: list):
        unsupported=[]
        cited_chunk_ids={
            citation.get("source_chunk_id")
            for citation in citations
            if citation.get("supported") and citation.get("source_chunk_id") is not None
        }

        for claim in self._extract_claims(answer):
            claim_citations=self.verify_citations(claim, self._build_sources(selected_documents))
            supported_citations=[citation for citation in claim_citations if citation.get("supported")]
            evidence_text=self._evidence_text_for_claim(selected_documents, supported_citations, cited_chunk_ids)
            support_score=self._claim_support_score(claim, evidence_text)
            # Arabic claims against English evidence cannot be compared via
            # token overlap. Skip the deterministic verdict for those; their
            # support is carried by citation faithfulness (the citations still
            # carry the exact English document names and page numbers).
            if self._contains_arabic(claim) and not self._contains_arabic(evidence_text):
                continue
            if support_score < 0.15:
                unsupported.append({
                    "claim": claim,
                    "reason": "weak_evidence_overlap" if claim_citations else "uncited_and_weak_evidence_overlap",
                    "support_score": round(support_score, 3),
                })

        return unsupported

    def _extract_claims(self, answer: str):
        text=(answer or "").strip()
        if not text:
            return []

        skip_patterns=re.compile(
            r"\b(i can'?t diagnose|consult a qualified|emergency medical|"
            r"local emergency services|here is general information|i do not have enough)\b",
            re.IGNORECASE,
        )

        # Mask citation brackets so the period in "p. 12]" never splits a claim,
        # and keep line breaks so each bullet stays its own claim.
        def _mask_line(snippet):
            citations=[]
            def _replace(match):
                citations.append(match.group(0))
                return f"\uE000CIT{len(citations)}\uE001"
            return re.sub(r"\[[^\[\]]+\]", _replace, snippet), citations

        def _unmask(fragment, citations):
            def _restore(match):
                idx=int(match.group(1))-1
                if 0 <= idx < len(citations):
                    return citations[idx]
                return match.group(0)
            return re.sub(r"\uE000CIT(\d+)\uE001", _restore, fragment)

        claims=[]
        for raw_line in re.split(r"\n+", text):
            line=raw_line.strip()
            line=re.sub(r"^[-*•]\s+", "", line)
            if not line:
                continue
            masked_line, citations=_mask_line(line)
            for fragment in re.split(r"(?<=[.!?؟])\s+", masked_line):
                claim=_unmask(fragment.strip(), citations)
                if len(claim.split()) < 5:
                    continue
                if skip_patterns.search(claim):
                    continue
                claims.append(claim)

        return claims

    def _evidence_text_for_claim(self, selected_documents: list, supported_citations: list,
                                 fallback_chunk_ids: set):
        chunk_ids={
            citation.get("source_chunk_id")
            for citation in supported_citations
            if citation.get("source_chunk_id") is not None
        }
        if not chunk_ids:
            chunk_ids=fallback_chunk_ids

        texts=[]
        for doc in selected_documents or []:
            if chunk_ids and doc.get("chunk_id") not in chunk_ids:
                continue
            texts.append(doc.get("text") or "")

        if not texts:
            texts=[doc.get("text") or "" for doc in selected_documents or []]

        return " ".join(texts)

    def _claim_support_score(self, claim: str, evidence_text: str):
        claim_tokens=self._meaningful_tokens(claim)
        if not claim_tokens:
            return 0.0

        evidence_tokens=set(self._meaningful_tokens(evidence_text))
        if not evidence_tokens:
            return 0.0

        overlap=sum(1 for token in claim_tokens if token in evidence_tokens)
        token_score=overlap / len(claim_tokens)

        # Numeric/unit agreement: a claim carrying figures (doses, ranges,
        # frequencies, percentages) is only strongly supported when the
        # evidence carries the same values.
        claim_values=self._numeric_values(claim)
        if claim_values:
            evidence_values=self._numeric_values(evidence_text)
            matched_values=sum(1 for value in claim_values if value in evidence_values)
            numeric_ratio=matched_values / len(claim_values)
            return 0.55 * token_score + 0.45 * numeric_ratio

        return token_score

    @staticmethod
    def _numeric_values(text: str):
        """Normalize numbers so '90%'==='90 %'==='90%' and '2.5'!=='25'."""
        normalized=[]
        for token in re.findall(r"\d[\d,]*\.?\d*\s*%?|\d+", (text or "").replace(",", "")):
            token=token.replace(" ", "")
            if token not in normalized:
                normalized.append(token)
        return set(normalized)

    @staticmethod
    def _is_numeric_or_dosing_query(question: str):
        q=(question or "").strip()
        if not q:
            return False
        unit=re.compile(
            r"\b(mg|mcg|ug|ml|mcg/day|mg/day|times\s+a\s+day|twice\s+a\s+day|"
            r"once\s+a\s+day|puffs?|tablets?|doses?|dosage|dose|frequency|"
            r"dosages?|per\s+day|per\s+week|daily)\b",
            re.IGNORECASE,
        )
        quantity=re.compile(
            r"\b(how\s+many|how\s+much|how\s+often|what\s+dose|what\s+dosage|"
            r"dose\s+of|dosage\s+of)\b",
            re.IGNORECASE,
        )
        has_unit=bool(unit.search(q))
        has_digit=bool(re.search(r"\d", q))
        has_quantity=bool(quantity.search(q))
        return has_unit and (has_digit or has_quantity)

    @staticmethod
    def _meaningful_tokens(text: str):
        stopwords={
            "about", "after", "also", "based", "before", "being", "between",
            "could", "every", "from", "have", "into", "more", "must", "only",
            "should", "that", "their", "there", "these", "this", "those",
            "when", "where", "which", "with", "within", "would", "your",
            "document", "documents", "guideline", "guidelines", "source",
        }
        arabic_stopwords={
            "وال", "في", "من", "إلى", "الى", "على", "عن", "مع", "هذا", "هذه",
            "ذلك", "التي", "الذي", "أن", "إن", "ما", "لم", "لن", "هل", "كل",
            "بعض", "عند", "لدى", "بعد", "قبل", "حتى", "أو", "ولا", "لكن",
            "كان", "كانت", "هي", "هو", "هم", "هن", "كما", "قد", "ثم", "حيث",
            "هناك", "إذا", "اذا", "ليس", "ليست", "أثناء", "خلال", "بين",
        }
        raw=text or ""
        # Strip Arabic diacritics/harakat so inflected forms still match.
        raw=re.sub(r"[\u064B-\u065F\u0670]", "", raw)
        lowered=raw.lower()

        tokens=re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", lowered)
        # Arabic word tokens (Arabic script), including hamza variants.
        for word in re.findall(r"[\u0621-\u064A\u0671-\u06D3][\u0621-\u064A\u0671-\u06D3\-]*", raw):
            word=word.strip("-")
            if len(word) >= 2 and word not in arabic_stopwords:
                tokens.append(word)

        return [token for token in tokens if token not in stopwords]

    @staticmethod
    def _contains_arabic(text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text or ""))

    @staticmethod
    def _clinical_disclaimer():
        return (
            "This system supports clinical guideline review and does not replace "
            "clinical judgment, diagnosis, emergency care, or advice from a qualified "
"healthcare professional."
        )

    ORG_ALIASES = None

    @classmethod
    def _get_org_aliases(cls):
        if cls.ORG_ALIASES is None:
            cls.ORG_ALIASES = (get_safety_config().get("org_aliases") or {})
        return cls.ORG_ALIASES

    CITATION_STOP_TOKENS = {
        "the", "a", "an", "and", "or", "of", "for", "in", "on", "to",
        "with", "at", "by", "is", "are", "was", "be", "as", "pp", "p",
    }

    @classmethod
    def _doc_tokens(cls, text: str):
        return {
            token
            for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if token not in cls.CITATION_STOP_TOKENS
        }

    @classmethod
    def _citation_doc_score(cls, document: str, source_doc: str, source_url: str):
        """0.0 = no match, >=0.6 = fuzzy token-overlap match, 1.0 = exact match."""
        normalized_doc = document.lower()
        normalized_source = source_doc.lower()
        if normalized_doc in normalized_source or normalized_source in normalized_doc:
            return 1.0

        citation_tokens = cls._doc_tokens(document)
        source_tokens = cls._doc_tokens(source_doc)
        if not citation_tokens:
            return 0.0

        haystack = f"{normalized_source} {(source_url or '').lower()}"
        for token in citation_tokens:
            for org, markers in cls._get_org_aliases().items():
                if token == org and any(marker in haystack for marker in markers):
                    return 1.0

        overlap = len(citation_tokens & source_tokens)
        if overlap == 0:
            return 0.0
        return overlap / min(len(citation_tokens), len(source_tokens))

    @classmethod
    def _normalize_section(cls, text: str):
        normalized=re.sub(
            r"^(section|sec|heading)\s*[:.#-]?\s*",
            "",
            (text or "").lower().strip(),
        )
        normalized=re.sub(r"\s+", " ", normalized)
        return normalized.strip(" .,;:-()[]\t")

    @classmethod
    def _sections_match(cls, cited_section: str, source_section: str, source_text: str = ""):
        cited=cls._normalize_section(cited_section)
        source=cls._normalize_section(source_section)
        if not cited:
            return True
        if cited in source or source in cited:
            return True
        # Numbered refs like "1.2.1" rarely appear in the extracted PDF
        # section_title; accept only when the chunk body carries the parent
        # section marker (e.g. "1.2") that the cited number descends from.
        match=re.match(r"(\d+(?:\.\d+)+)", cited)
        if not match:
            return False
        number=match.group(1)
        if number in (source_text or ""):
            return True
        parent=number.rsplit(".", 1)[0]
        return bool(parent and parent in (source_text or ""))

    @classmethod
    def _find_cited_source(cls, document: str, page_number: int, section: str, sources: list):
        best_match = None
        best_score = 0.0

        for source in sources or []:
            source_doc = (source.get("document_name") or source.get("file_name") or "")
            source_section = (source.get("section_title") or "")
            source_page = source.get("page_number")
            try:
                source_page = int(source_page)
            except (TypeError, ValueError):
                continue

            page_matches = source_page == page_number
            section_matches = cls._sections_match(
                section, source_section, source.get("text") or ""
            )

            if not (page_matches and section_matches):
                continue

            score = cls._citation_doc_score(
                document, source_doc, source.get("source_url") or ""
            )
            if score > best_score:
                best_score = score
                best_match = source

        if best_score >= 0.6:
            return best_match
        return None
        
