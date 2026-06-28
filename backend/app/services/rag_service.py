import math
import re
import time
from statistics import mean, median
from threading import Lock
from typing import Any, Dict, Generator, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.logging import logger
from app.core.config import settings
from app.services.llm_manager import LLMManager
from app.services.map_intelligence_service import MapIntelligenceService
from app.services.vector_store import VectorStoreManager
from app.services.filter_engine import FilterEngine

class RAGService:
    """
    The Maestro. Orchestrates the flow between User Query -> Filter Extraction ->
    Vector Retrieval -> Strict Filtering -> Padding Logic -> LLM Chat Generation.
    """
    def __init__(self):
        self.llm_manager = LLMManager()
        self.vector_store = VectorStoreManager()
        self.map_intelligence = MapIntelligenceService()
        self.filter_engine = FilterEngine(self.llm_manager)
        
        # In-memory session tracking (Could be upgraded to Redis for production)
        self.sessions = {}
        self.chat_history = {}
        self.interaction_history: Dict[str, List[Dict[str, Any]]] = {}
        self.recommendation_cache: Dict[str, Tuple[float, List[Document]]] = {}

    def get_recommendation(self, query: str, session_id: str = None) -> Tuple[str, List[Document]]:
        """Handles conversational interaction, processing filters and returning contextual replies."""
        if self._is_greeting(query):
            response_text = self._build_greeting_response(query)
            if session_id:
                if session_id not in self.sessions:
                    self.sessions[session_id] = {}
                    self.chat_history[session_id] = []
                self.chat_history[session_id].append(HumanMessage(content=query))
                self.chat_history[session_id].append(AIMessage(content=response_text))
            return response_text, []

        if not self.vector_store.vectorstore:
            return "System is initializing database, please hold on!", []

        history: List = []
        filters: Dict = {}

        # 1. Session setup
        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = {}
                self.chat_history[session_id] = []
            history = self.chat_history[session_id]

        # 2. Feature Extraction
        new_filters = self.filter_engine.extract_filters(
            query,
            self.vector_store.available_locations,
            self.vector_store.available_services,
        )
        if not isinstance(new_filters, dict):
            logger.warning("Filter engine returned non-dict payload. Falling back to empty filters.")
            new_filters = {}
        new_filters = self._sanitize_filters(new_filters)

        # 3. Session Memory Merging
        if session_id:
            for k, v in new_filters.items():
                if v is not None:
                    self.sessions[session_id][k] = v
            filters = self.sessions[session_id]
        else:
            filters = new_filters

        stats_intent = self._is_inventory_stats_intent(query, filters)
        scoring_intent = self._is_scoring_explanation_intent(query)
        coverage_intent = self._is_coverage_intent(query, filters)
        if stats_intent or scoring_intent or coverage_intent:
            response_parts: List[str] = []
            if stats_intent:
                stats_filters = new_filters if self._has_active_filters(new_filters) else filters
                response_parts.append(
                    self._build_inventory_stats_response(query=query, filters=stats_filters)
                )
            if coverage_intent:
                coverage_filters = new_filters if self._has_active_filters(new_filters) else filters
                response_parts.append(
                    self._build_coverage_response(query=query, filters=coverage_filters)
                )
            if scoring_intent:
                response_parts.append(self._build_scoring_explanation_response())

            response_text = "\n\n".join([part.strip() for part in response_parts if part.strip()]).strip()
            if session_id:
                self.chat_history[session_id].append(HumanMessage(content=query))
                self.chat_history[session_id].append(AIMessage(content=response_text))
            return response_text, []

        analysis_intent = self._is_analysis_intent(query, filters)
        if analysis_intent:
            analysis_result = self.analyze_market(query=query, explicit_filters=filters)
            response_text = self._build_analysis_chat_response(analysis_result)
            analysis_docs = self._collect_analysis_docs(analysis_result)
            if session_id:
                self.chat_history[session_id].append(HumanMessage(content=query))
                self.chat_history[session_id].append(AIMessage(content=response_text))
            return response_text, analysis_docs

        # Check if it's a property-related query (anti-hallucination guard)
        has_meaningful_filters = any(
            v is not None and v != "" and v != [] and v != {}
            for v in new_filters.values()
        )
        is_property_query = has_meaningful_filters or self._is_property_keyword_query(query)
        
        # Extra guard: if query matches casual/chitchat patterns, never show properties
        if self._is_casual_chitchat(query):
            is_property_query = False
        
        if not is_property_query:
            # Conversational response for non-property queries
            response_text, _ = self._generate_conversational_response(query, history)
            if session_id:
                self.chat_history[session_id].append(HumanMessage(content=query))
                self.chat_history[session_id].append(AIMessage(content=response_text))
            return response_text, []
            
        logger.info(f"Active Merged Filters applied: {filters}")

        # 4. Vector Retrieval & Hard Constraints Application
        search_query = self._build_effective_search_query(query, filters)
        raw_docs = self.vector_store.retrieve(search_query, k=settings.chat_retrieval_k)
        filtered_docs = self._expand_strict_matches(raw_docs, filters)
        
        # 5. Fallback Routing & Padding
        search_status, final_docs = self._enforce_padding_logic(
            search_query,
            filters,
            filtered_docs,
            raw_docs,
            max_results=settings.chat_result_limit,
        )
        listing_intent = self._normalize_listing_intent(filters.get("listing_intent"))
        if listing_intent and not final_docs:
            response_text = self._build_listing_intent_unavailable_response(listing_intent, filters)
            if session_id:
                self.chat_history[session_id].append(HumanMessage(content=query))
                self.chat_history[session_id].append(AIMessage(content=response_text))
            return response_text, []
        final_docs = self._rank_recommendations(final_docs, filters)

        # 6. Language Model Generation
        is_arabic = bool(re.search(r"[\u0600-\u06FF]", query))
        if settings.fast_property_responses and not is_arabic:
            return self._generate_fast_property_response(query, search_status, final_docs, history)

        response_text, generated_docs = self._generate_response(query, search_status, final_docs, history)
        
        return response_text, generated_docs

    def search_properties(self, query: Optional[str] = None, explicit_filters: Dict = None) -> Tuple[Dict, List[Document]]:
        """Headless search execution for direct API/UI querying without conversational fluff."""
        if not self.vector_store.vectorstore:
            return {}, []
            
        filters = dict(explicit_filters or {})
        filters = self._sanitize_filters(filters)
        if query:
            extracted = self.filter_engine.extract_filters(
                query,
                self.vector_store.available_locations,
                self.vector_store.available_services,
            )
            extracted = self._sanitize_filters(extracted)
            # Merge extracted into explicit but prioritize explicit UI inputs
            for k, v in extracted.items():
                if k not in filters or filters[k] is None:
                    filters[k] = v
        
        search_query = self._build_effective_search_query(query, filters)
        raw_docs = self.vector_store.retrieve(search_query, k=settings.search_retrieval_k)
        filtered_docs = self._expand_strict_matches(raw_docs, filters)
        
        _, final_docs = self._enforce_padding_logic(
            search_query,
            filters,
            filtered_docs,
            raw_docs,
            max_results=settings.search_result_limit,
        )
        final_docs = self._rank_recommendations(final_docs, filters)
        return filters, final_docs

    def _get_region(self, location_str: str) -> str:
        loc = str(location_str or "").lower()
        if any(w in loc for w in ["alex", "اسكندرية", "إسكندرية", "سموحة", "smouha", "كامب شيزار", "caesar"]):
            return "Alexandria"
        if any(w in loc for w in ["zayed", "زايد", "اكتوبر", "october", "جيزة", "giza"]):
            return "Zayed"
        if any(w in loc for w in ["dahab", "دهب"]):
            return "Dahab"
        if any(w in loc for w in ["hurghada", "غردقة", "red sea", "البحر الاحمر"]):
            return "Hurghada"
        if any(w in loc for w in ["mansoura", "منصورة"]):
            return "Mansoura"
        if any(w in loc for w in ["cairo", "new cairo", "settlement", "قاهرة", "تجمع", "رحاب", "rehab", "مدينتي", "madinaty"]):
            return "Cairo"
        return "Other"

    def recommend_similar(self, description: str, k: Optional[int] = None) -> List[Document]:
        """Provides direct semantic equivalents to a given property description with heuristic ranking."""
        limit = self._result_limit(k, settings.recommendation_result_limit, 100)
        
        # 1. Try to find the exact target property in our database to get precise metadata
        target_doc = None
        for doc in self.vector_store.all_docs_list:
            doc_desc = doc.metadata.get("property_desc") or doc.metadata.get("description") or ""
            doc_title = doc.metadata.get("property_name") or doc.metadata.get("title") or ""
            if (description.strip() and 
                (description.strip() in doc_desc.strip() or 
                 doc_desc.strip() in description.strip() or 
                 doc_title.strip() in description.strip())):
                target_doc = doc
                break

        # 2. Extract/infer filters from the description or target doc
        target_id = None
        target_loc = ""
        target_type = ""
        target_intent = ""
        target_price = 0.0
        target_beds = 0.0

        if target_doc:
            target_id = self._doc_property_id(target_doc)
            target_loc = target_doc.metadata.get("location") or ""
            target_type = target_doc.metadata.get("type") or ""
            target_intent = self._doc_listing_intent(target_doc) or ""
            target_price = self._safe_float(target_doc.metadata.get("price"))
            target_beds = self._safe_float(target_doc.metadata.get("bedrooms"))
        else:
            try:
                extracted = self.filter_engine.extract_filters(
                    description,
                    self.vector_store.available_locations,
                    self.vector_store.available_services,
                )
                target_loc = extracted.get("location") or ""
                target_type = extracted.get("property_type") or ""
                target_intent = extracted.get("listing_intent") or ""
                target_price = self._safe_float(extracted.get("max_price") or extracted.get("min_price"))
                target_beds = self._safe_float(extracted.get("min_bedrooms"))
            except Exception as e:
                logger.warning(f"Failed to extract filters for recommendation: {e}")

        # 3. Filter other docs to find candidates
        target_region = self._get_region(target_loc)
        target_type_clean = target_type.lower() if target_type else ""
        target_intent_clean = target_intent.lower() if target_intent else ""

        # Level-based candidate gathering
        candidates = []
        
        # Level 1: same region + same type + same intent
        for doc in self.vector_store.all_docs_list:
            doc_id = self._doc_property_id(doc)
            if target_id and doc_id == target_id:
                continue
            loc = doc.metadata.get("location", "")
            doc_type = str(doc.metadata.get("type", "")).lower()
            doc_intent = self._doc_listing_intent(doc) or ""
            if (self._get_region(loc) == target_region and 
                (target_type_clean in doc_type or doc_type in target_type_clean) and 
                doc_intent == target_intent_clean):
                candidates.append(doc)

        # Level 2: same region + same intent
        if len(candidates) < limit:
            for doc in self.vector_store.all_docs_list:
                doc_id = self._doc_property_id(doc)
                if target_id and doc_id == target_id:
                    continue
                if doc in candidates:
                    continue
                loc = doc.metadata.get("location", "")
                doc_intent = self._doc_listing_intent(doc) or ""
                if self._get_region(loc) == target_region and doc_intent == target_intent_clean:
                    candidates.append(doc)

        # Level 3: same region
        if len(candidates) < limit:
            for doc in self.vector_store.all_docs_list:
                doc_id = self._doc_property_id(doc)
                if target_id and doc_id == target_id:
                    continue
                if doc in candidates:
                    continue
                loc = doc.metadata.get("location", "")
                if self._get_region(loc) == target_region:
                    candidates.append(doc)

        # Level 4: global semantic backup
        if len(candidates) < limit:
            sim_docs = self.vector_store.similarity_search(description, k=limit + 10)
            for doc in sim_docs:
                doc_id = self._doc_property_id(doc)
                if target_id and doc_id == target_id:
                    continue
                if doc in candidates:
                    continue
                candidates.append(doc)
                if len(candidates) >= limit + 10:
                    break

        # 4. Score candidates with a heuristic
        scored_candidates = []
        for doc in candidates:
            score = 0.0
            loc = doc.metadata.get("location", "")
            doc_type = str(doc.metadata.get("type", "")).lower()
            doc_intent = self._doc_listing_intent(doc) or ""
            doc_price = self._safe_float(doc.metadata.get("price"))

            # Region match (weight: 100)
            if self._get_region(loc) == target_region:
                score += 100.0
            
            # Type match (weight: 50)
            if target_type_clean and (target_type_clean in doc_type or doc_type in target_type_clean):
                score += 50.0

            # Intent match (weight: 30)
            if target_intent_clean and target_intent_clean == doc_intent:
                score += 30.0

            # Price closeness (weight: 20)
            if target_price > 0 and doc_price > 0:
                price_diff = abs(doc_price - target_price) / target_price
                price_score = max(0.0, 1.0 - price_diff)
                score += price_score * 20.0

            # Bedrooms closeness (weight: 10)
            doc_beds = self._safe_float(doc.metadata.get("bedrooms"))
            if target_beds > 0 and doc_beds > 0:
                beds_diff = abs(doc_beds - target_beds)
                beds_score = max(0.0, 1.0 - (beds_diff / 5.0))
                score += beds_score * 10.0

            scored_candidates.append((score, doc))

        # Sort descending by score and return top K
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        final_recommendations = [doc for _, doc in scored_candidates[:limit]]
        return final_recommendations

    def record_property_interaction(
        self,
        session_id: str,
        property_id: int,
        event_type: str = "click",
    ) -> List[int]:
        """Stores lightweight interest signals for anonymous session recommendations."""
        normalized_session = str(session_id or "").strip()
        normalized_property_id = self._safe_int_value(property_id)
        if not normalized_session or not normalized_property_id:
            return []

        event = {
            "property_id": normalized_property_id,
            "event_type": str(event_type or "click").strip().lower() or "click",
            "timestamp": time.time(),
        }
        events = self.interaction_history.setdefault(normalized_session, [])
        events.append(event)

        max_events = 50
        if len(events) > max_events:
            del events[:-max_events]

        self._invalidate_session_recommendation_cache(normalized_session)
        return self.get_session_property_ids(normalized_session)

    def get_session_property_ids(self, session_id: str) -> List[int]:
        seen = set()
        ordered_ids: List[int] = []
        for event in self.interaction_history.get(str(session_id or "").strip(), []):
            property_id = self._safe_int_value(event.get("property_id"))
            if property_id and property_id not in seen:
                ordered_ids.append(property_id)
                seen.add(property_id)
        return ordered_ids

    def recommend_from_interactions(
        self,
        session_id: str,
        property_ids: Optional[List[int]] = None,
        limit: int = 5,
    ) -> List[Document]:
        """Builds recommendations from clicked/favorited properties with TTL caching."""
        normalized_session = str(session_id or "").strip()
        explicit_ids = [self._safe_int_value(item) for item in (property_ids or [])]
        explicit_ids = [item for item in explicit_ids if item]
        seed_ids = self._unique_ordered(self.get_session_property_ids(normalized_session) + explicit_ids)
        if not seed_ids:
            return []

        limit = self._result_limit(limit, settings.recommendation_result_limit, 100)
        cache_key = f"{normalized_session}:{','.join(map(str, seed_ids))}:{limit}"
        cached = self.recommendation_cache.get(cache_key)
        if cached and time.time() - cached[0] <= settings.interaction_cache_ttl_sec:
            return cached[1]

        seed_docs = self.vector_store.get_docs_by_property_ids(seed_ids)
        if not seed_docs:
            return []

        seed_text = "\n\n".join(
            [
                self._recommendation_seed_text(doc)
                for doc in seed_docs[-8:]
            ]
        )
        candidates = self.vector_store.similarity_search(seed_text, k=limit + len(seed_ids) + 15)
        excluded = set(seed_ids)
        recommendations: List[Document] = []
        for doc in candidates:
            doc_id = self._doc_property_id(doc)
            if doc_id in excluded:
                continue
            recommendations.append(doc)
            if len(recommendations) >= limit:
                break

        self.recommendation_cache[cache_key] = (time.time(), recommendations)
        self._trim_recommendation_cache()
        return recommendations

    def _build_effective_search_query(
        self,
        query: Optional[str],
        filters: Optional[Dict[str, Any]],
        fallback: str = "real estate in Egypt",
    ) -> str:
        """
        Keeps natural language as the source of truth for retrieval.
        Explicit fields are folded back into the query text instead of replacing
        it with a dict-like string, so address/location/title signals stay inside
        one semantic search query.
        """
        parts: List[str] = []
        base_query = str(query or "").strip()
        if base_query:
            parts.append(base_query)

        safe_filters = filters if isinstance(filters, dict) else {}
        field_labels = {
            "location": "Location",
            "address": "Address",
            "title": "Title",
            "property_type": "Type",
            "listing_intent": "Listing intent",
            "min_price": "Minimum price",
            "max_price": "Maximum price",
            "min_bedrooms": "Minimum bedrooms",
            "max_bedrooms": "Maximum bedrooms",
        }
        for key, label in field_labels.items():
            value = safe_filters.get(key)
            if value in (None, "", [], {}):
                continue
            parts.append(f"{label}: {value}")

        desired_services = self._normalize_services(safe_filters.get("desired_services", []))
        if desired_services:
            parts.append(f"Nearby services: {', '.join(desired_services)}")

        return "\n".join(parts).strip() or fallback

    def analyze_market(self, query: Optional[str] = None, explicit_filters: Dict = None) -> Dict[str, Any]:
        """
        Generates numerical market insights from matched inventory rather than conversational output.
        """
        if not self.vector_store.vectorstore:
            return {
                "insight": "لا تتوفر قاعدة بيانات التحليل حاليًا، حاول مرة أخرى بعد اكتمال التهيئة.",
                "filters_used": {},
                "match_scope": "none",
                "total_candidates": 0,
                "matched_count": 0,
                "stats": self._compute_market_stats([]),
                "top_locations": [],
                "top_property_types": [],
                "buy_decision": {
                    "decision": "insufficient_data",
                    "headline": "بيانات غير كافية",
                    "confidence": 0.0,
                    "reasons": ["قاعدة البيانات غير متاحة الآن، لذلك لا يمكن إصدار قرار شراء."],
                },
                "better_option_found": False,
                "better_option_reason": "لا يمكن تحديد بديل أفضل لأن قاعدة البيانات غير متاحة.",
                "better_option_doc": None,
                "sample_docs": [],
            }

        filters = dict(explicit_filters or {})
        filters = self._sanitize_filters(filters)
        if query:
            extracted = self.filter_engine.extract_filters(
                query,
                self.vector_store.available_locations,
                self.vector_store.available_services,
            )
            extracted = self._sanitize_filters(extracted)
            if isinstance(extracted, dict):
                for k, v in extracted.items():
                    if k not in filters or filters[k] is None:
                        filters[k] = v

        search_query = self._build_effective_search_query(query, filters, fallback="real estate market in Egypt")
        raw_docs = self.vector_store.retrieve(search_query, k=250)
        strict_docs = self._apply_exact_filters(raw_docs, filters)

        if strict_docs:
            analysis_docs = strict_docs
            match_scope = "strict"
        elif raw_docs:
            requested_location = str(filters.get("location") or "").strip().lower()
            location_docs = self._filter_docs_by_location(raw_docs, requested_location)
            if location_docs:
                analysis_docs = location_docs[:60]
                match_scope = "location_fallback"
            else:
                analysis_docs = raw_docs[:60]
                match_scope = "semantic_fallback"
        else:
            analysis_docs = []
            match_scope = "none"

        analysis_docs = self._rank_recommendations(analysis_docs, filters)

        stats = self._compute_market_stats(analysis_docs)
        top_locations = self._compute_segment_stats(analysis_docs, segment_key="location")
        top_property_types = self._compute_segment_stats(analysis_docs, segment_key="type")
        buy_decision = self._compute_buy_decision(analysis_docs, stats, filters, match_scope)
        better_option = self._identify_better_option(analysis_docs, stats, filters)

        return {
            "insight": self._build_market_insight(stats, top_locations, top_property_types, filters, match_scope),
            "filters_used": filters,
            "match_scope": match_scope,
            "total_candidates": len(raw_docs),
            "matched_count": len(analysis_docs),
            "stats": stats,
            "top_locations": top_locations,
            "top_property_types": top_property_types,
            "buy_decision": buy_decision,
            "better_option_found": better_option["found"],
            "better_option_reason": better_option["reason"],
            "better_option_doc": better_option["doc"],
            "sample_docs": analysis_docs[:5],
        }

    def _filter_docs_by_location(self, docs: List[Document], requested_location: str) -> List[Document]:
        if not requested_location:
            return []

        tokens = self._location_match_tokens(requested_location)
        matched: List[Document] = []
        for doc in docs:
            doc_location = str(doc.metadata.get("location", "")).lower()
            if any(token in doc_location for token in tokens):
                matched.append(doc)
        return matched

    def _location_match_tokens(self, requested_location: str) -> List[str]:
        if not requested_location:
            return []

        normalized = requested_location.lower()
        tokens: List[str] = [normalized]

        if "5th settlement" in normalized:
            tokens.extend(["the 5th settlement", "5th settlement", "new cairo", "settlement"])
        if "1st settlement" in normalized:
            tokens.extend(["the 1st settlement", "1st settlement", "new cairo", "settlement"])
        if "new cairo" in normalized:
            tokens.extend(["new cairo", "settlement", "the 5th settlement", "the 1st settlement"])
        if "sheikh zayed" in normalized or normalized == "zayed":
            tokens.extend(["sheikh zayed", "zayed"])
        if "october" in normalized:
            tokens.extend(["6th of october", "october"])
        if "north coast" in normalized:
            tokens.extend(["north coast", "sahel", "sidi abdelrahman"])
        if "new capital" in normalized:
            tokens.extend(["new capital", "administrative capital"])

        unique_tokens: List[str] = []
        for token in tokens:
            if token and token not in unique_tokens:
                unique_tokens.append(token)
        return unique_tokens

    @staticmethod
    def _has_active_filters(filters: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(filters, dict):
            return False
        return any(value not in (None, "", [], {}) for value in filters.values())

    @staticmethod
    def _is_greeting(query: str) -> bool:
        normalized = re.sub(r"[\s،,.!؟?]+", " ", str(query or "").strip().lower()).strip()
        if not normalized:
            return False
        greeting_phrases = {
            "اهلا",
            "أهلا",
            "اهلا بيك",
            "أهلا بيك",
            "السلام عليكم",
            "سلام عليكم",
            "ازيك",
            "إزيك",
            "هاي",
            "هالو",
            "hello",
            "hi",
            "hey",
        }
        return normalized in {phrase.lower() for phrase in greeting_phrases}

    @staticmethod
    def _build_greeting_response(query: str) -> str:
        is_arabic_query = bool(re.search(r"[\u0600-\u06FF]", query))
        if is_arabic_query:
            return (
                "أهلاً بيك! أنا AqarAI، أقدر أساعدك تلاقي شقة أو عقار مناسب. "
                "اكتبلي المنطقة والميزانية وعدد الغرف، وأنا أطلعلك أفضل الاختيارات المتاحة."
            )
        else:
            return (
                "Hello! I am AqarAI, your smart real estate consultant. "
                "Please tell me the area, budget, and number of bedrooms, and I will find the best options for you."
            )

    def _is_inventory_stats_intent(self, query: str, filters: Optional[Dict[str, Any]] = None) -> bool:
        raw_query = str(query or "").strip().lower()
        if not raw_query:
            return False

        count_tokens = [
            "count",
            "how many",
            "number of",
            "stats",
            "statistics",
            "عدد",
            "كام",
            "كم",
            "احصائيات",
            "إحصائيات",
            "اجمالي",
            "إجمالي",
        ]
        entity_tokens = [
            "apartment",
            "apartments",
            "property",
            "properties",
            "unit",
            "units",
            "شقة",
            "شقق",
            "عقار",
            "عقارات",
            "وحدة",
            "وحدات",
            "فيلا",
            "فلل",
        ]

        has_count_token = any(token in raw_query for token in count_tokens)
        if not has_count_token:
            return False

        has_entity_token = any(token in raw_query for token in entity_tokens)
        has_filter_context = self._has_active_filters(filters)
        return has_entity_token or has_filter_context

    def _is_coverage_intent(self, query: str, filters: Optional[Dict[str, Any]] = None) -> bool:
        raw_query = str(query or "").strip().lower()
        if not raw_query:
            return False

        coverage_tokens = [
            "مناطق ايه",
            "مناطق إيه",
            "مناطق",
            "فين",
            "اماكن ايه",
            "أماكن إيه",
            "available locations",
            "locations",
            "areas",
            "where do you have",
        ]
        property_tokens = [
            "شقة",
            "شقق",
            "عقار",
            "عقارات",
            "property",
            "properties",
            "apartment",
            "villa",
            "عندكو",
            "عندك",
        ]

        has_coverage_token = any(token in raw_query for token in coverage_tokens)
        if not has_coverage_token:
            return False

        has_property_token = any(token in raw_query for token in property_tokens)
        has_filter_context = self._has_active_filters(filters)
        return has_property_token or has_filter_context

    def _is_property_keyword_query(self, query: str) -> bool:
        raw_query = str(query or "").strip().lower()
        if not raw_query:
            return False

        # Must contain at least one property-specific keyword
        property_tokens = [
            "apartment",
            "apartments",
            "villa",
            "duplex",
            "chalet",
            "townhouse",
            "property",
            "real estate",
            "شقة",
            "شقق",
            "فيلا",
            "دوبلكس",
            "شاليه",
            "عقار",
            "عقارات",
            "للإيجار",
            "للايجار",
            "إيجار",
            "ايجار",
            "للبيع",
            "شراء",
            "بيع",
            "غرف",
            "غرفة",
            "متر",
            "كمبوند",
            "compound",
        ]
        return any(token in raw_query for token in property_tokens)

    @staticmethod
    def _is_casual_chitchat(query: str) -> bool:
        """Detects casual/social queries that should NEVER trigger property results."""
        normalized = re.sub(r"[\s،,.!؟?]+", " ", str(query or "").strip().lower()).strip()
        if not normalized:
            return False

        # Exact match phrases that are definitively NOT about properties
        chitchat_exact = {
            "ازيك", "إزيك", "ازيكم", "عامل ايه", "عامل إيه",
            "الحمد لله", "تمام", "شكرا", "شكراً", "thanks", "thank you",
            "مين انت", "مين أنت", "انت مين", "أنت مين",
            "بتعمل ايه", "بتعمل إيه", "ايه ده", "إيه ده",
            "يعني ايه", "يعني إيه", "ok", "اوك", "أوك", "حسنا",
            "ماشي", "تمام كدة", "اه", "أه", "لا", "مش عاوز",
            "باي", "bye", "مع السلامة", "سلام",
            "ايه الاخبار", "إيه الأخبار", "اخبارك ايه", "أخبارك إيه",
            "كسم الحر", "الجو حر", "الطقس", "weather",
            "what is your name", "who are you", "how are you",
            "ايه اسمك", "إيه اسمك", "اسمك ايه", "اسمك إيه",
            "good", "nice", "cool", "great",
        }
        if normalized in chitchat_exact:
            return True

        # Greeting-prefix patterns: if query STARTS with a greeting, it's chitchat
        # even if it contains property-adjacent words like "يا عقاري"
        greeting_prefixes = [
            "ازيك", "إزيك", "اهلا", "أهلا", "هاي", "هالو",
            "hello", "hi ", "hey ", "مرحبا", "يا هلا",
            "صباح الخير", "مساء الخير", "السلام عليكم", "سلام عليكم",
        ]
        for prefix in greeting_prefixes:
            if normalized.startswith(prefix):
                return True

        # Pattern-based: very short queries with no property keywords
        if len(normalized.split()) <= 3:
            property_hints = {
                "شقة", "شقق", "فيلا", "عقار ", "عقارات", "كمبوند",
                "compound", "apartment", "villa", "duplex", "chalet",
                "للبيع", "للايجار", "للإيجار", "إيجار", "ايجار",
                "شراء", "بيع", "غرف", "غرفة", "متر",
                "تجمع", "زايد", "اكتوبر", "ساحل", "العاصمة",
                "ابحث", "دور", "عاوز شقة", "عايز شقة",
            }
            if not any(hint in normalized for hint in property_hints):
                return True

        return False

    def _is_scoring_explanation_intent(self, query: str) -> bool:
        raw_query = str(query or "").strip().lower()
        if not raw_query:
            return False

        explain_tokens = [
            "score",
            "scoring",
            "rating",
            "recommendation score",
            "التقييم",
            "تقييم",
            "بيتحسب",
            "بناء على ايه",
            "بناء علي ايه",
            "بناءا على ايه",
            "بناءًا على ايه",
            "على اي اساس",
            "معايير",
            "ليه ده افضل",
            "ازاي بتقيم",
        ]
        return any(token in raw_query for token in explain_tokens)

    def _build_scoring_explanation_response(self) -> str:
        return (
            "التقييم عندنا مبني على داتا مشروعنا الداخلية، مش تقدير عشوائي:\n"
            "- `35%` القرب من المنطقة المطلوبة (Distance Score).\n"
            "- `25%` توفر الخدمات القريبة المطلوبة (Service Score).\n"
            "- `25%` القيمة السعرية = سعر المتر مقارنة بمتوسط النتائج (Value Score).\n"
            "- `15%` مدى مناسبة السعر للميزانية لو متوفرة (Budget Score).\n"
            "- Bonus `+0.05` لو نوع العقار مطابق لنوعك المطلوب.\n"
            "الخدمات بتيجي من الداتا الأساسية، ومع التفعيل الحي ممكن تتدعم من Map API.\n"
            "مثال: تقييم `0.47` يعني العقار حقق تقريبًا `47%` من أفضل Score ممكن حسب المعايير دي."
        )

    def _build_coverage_response(self, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        docs = self._get_inventory_docs_for_stats()
        if not docs:
            return "قاعدة بيانات المنصة غير متاحة حاليًا، مقدرش أحدد المناطق المتاحة دلوقتي."

        effective_filters = self._sanitize_filters(dict(filters or {}))
        inferred_type = self._infer_property_type_from_query(query)
        if inferred_type and not effective_filters.get("property_type"):
            effective_filters["property_type"] = inferred_type
        inferred_intent = self._infer_listing_intent_from_query(query)
        if inferred_intent and not effective_filters.get("listing_intent"):
            effective_filters["listing_intent"] = inferred_intent

        filtered_docs = self._apply_exact_filters(docs, effective_filters) if self._has_active_filters(effective_filters) else docs
        if not filtered_docs:
            return "مفيش مناطق مطابقة للشروط الحالية في قاعدة بياناتنا."

        grouped = self._compute_segment_stats(filtered_docs, segment_key="location", top_k=12)
        lines: List[str] = [
            "المناطق المتاحة عندنا في قاعدة بيانات الموقع حاليًا:",
            f"- إجمالي النتائج المطابقة: {len(filtered_docs)}.",
        ]

        property_type_value = effective_filters.get("property_type")
        listing_intent_value = self._normalize_listing_intent(effective_filters.get("listing_intent"))
        if property_type_value:
            lines.append(f"- نوع العقار: {self._property_type_label(property_type_value)}.")
        if listing_intent_value:
            lines.append(f"- نوع العرض: {'إيجار' if listing_intent_value == 'rent' else 'بيع'}.")

        lines.append("- أهم المناطق:")
        for idx, item in enumerate(grouped, start=1):
            lines.append(f"{idx}. {item['name']} ({item['count']})")

        return "\n".join(lines)

    def _build_listing_intent_unavailable_response(self, listing_intent: str, filters: Optional[Dict[str, Any]] = None) -> str:
        docs = self._get_inventory_docs_for_stats()
        rent_count = 0
        buy_count = 0
        for doc in docs:
            intent = self._doc_listing_intent(doc)
            if intent == "rent":
                rent_count += 1
            elif intent == "buy":
                buy_count += 1

        filters = self._sanitize_filters(dict(filters or {}))
        location = filters.get("location")
        ptype = self._property_type_label(filters.get("property_type")) if filters.get("property_type") else None

        if listing_intent == "rent":
            lines = [
                "حاليًا مفيش وحدات إيجار في قاعدة بياناتنا الحالية، فمقدرش أرشح إيجار بدقة من الداتا.",
                f"- إجمالي الإيجار المتاح في الداتا الآن: {rent_count}.",
                f"- إجمالي وحدات البيع المتاحة: {buy_count}.",
            ]
            if location:
                lines.append(f"- المنطقة المطلوبة: {location}.")
            if ptype:
                lines.append(f"- نوع العقار المطلوب: {ptype}.")
            lines.append("لو تحب، أقدر أعرض لك أفضل بدائل البيع بنفس المنطقة أو أول ما تضيفوا داتا إيجار هتظهر تلقائيًا.")
            return "\n".join(lines)

        return "لا توجد نتائج مطابقة للفلاتر الحالية."

    def _build_inventory_stats_response(self, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        docs = self._get_inventory_docs_for_stats()
        if not docs:
            return "قاعدة بيانات المنصة غير متاحة حاليًا، مقدرش أطلع إحصائيات دقيقة دلوقتي."

        effective_filters = self._sanitize_filters(dict(filters or {}))
        inferred_type = self._infer_property_type_from_query(query)
        if inferred_type and not effective_filters.get("property_type"):
            effective_filters["property_type"] = inferred_type
        inferred_intent = self._infer_listing_intent_from_query(query)
        if inferred_intent and not effective_filters.get("listing_intent"):
            effective_filters["listing_intent"] = inferred_intent

        filtered_docs = self._apply_exact_filters(docs, effective_filters) if self._has_active_filters(effective_filters) else docs
        stats = self._compute_market_stats(filtered_docs)
        top_locations = self._compute_segment_stats(filtered_docs, segment_key="location", top_k=3)

        total_count = len(docs)
        matched_count = len(filtered_docs)
        location_value = effective_filters.get("location")
        property_type_value = effective_filters.get("property_type")
        listing_intent_value = self._normalize_listing_intent(effective_filters.get("listing_intent"))

        lines = [
            "الإحصائيات دي من قاعدة بيانات موقعنا الحالية فقط (مش من أي منصة خارجية):",
            f"- إجمالي العقارات المفهرسة حاليًا: {total_count}.",
            f"- عدد النتائج المطابقة لطلبك: {matched_count}.",
        ]

        if location_value:
            lines.append(f"- النطاق المطلوب: {location_value}.")
        if property_type_value:
            lines.append(f"- نوع العقار: {self._property_type_label(property_type_value)}.")
        if listing_intent_value:
            lines.append(f"- نوع العرض المطلوب: {'إيجار' if listing_intent_value == 'rent' else 'بيع'}.")

        if matched_count == 0:
            lines.append("- مفيش نتائج مطابقة بنفس الشروط الحالية؛ وسّع المنطقة أو عدّل الفلاتر.")
            return "\n".join(lines)

        lines.append(f"- متوسط السعر: {stats.get('avg_price', 0.0):,.0f} جنيه.")
        lines.append(f"- وسيط السعر: {stats.get('median_price', 0.0):,.0f} جنيه.")
        lines.append(f"- متوسط سعر المتر: {stats.get('avg_price_per_sqm', 0.0):,.0f} جنيه.")

        if top_locations:
            top_bits = [f"{item['name']} ({item['count']})" for item in top_locations]
            lines.append(f"- أعلى المناطق ظهورًا: {', '.join(top_bits)}.")

        return "\n".join(lines)

    def _get_inventory_docs_for_stats(self) -> List[Document]:
        vector_store = getattr(self, "vector_store", None)
        if vector_store is None:
            return []

        docs: List[Document] = []
        if getattr(vector_store, "all_docs_list", None):
            docs = list(vector_store.all_docs_list)
        elif getattr(vector_store, "vectorstore", None) and hasattr(vector_store.vectorstore, "docstore"):
            store_dict = getattr(vector_store.vectorstore.docstore, "_dict", {})
            docs = list(store_dict.values())

        if docs:
            return vector_store._enrich_docs(docs)

        property_lookup = getattr(vector_store, "property_lookup", {}) or {}
        fallback_docs: List[Document] = []
        for url, row in property_lookup.items():
            fallback_docs.append(
                Document(
                    page_content=f"Description: {row.get('title') or 'Property Listing'}",
                    metadata={
                        "url": url,
                        "title": row.get("title"),
                        "location": row.get("location"),
                        "type": row.get("type"),
                        "price": row.get("price"),
                        "bedrooms": row.get("bedrooms"),
                        "bathrooms": row.get("bathrooms"),
                        "size": row.get("size"),
                        "lat": row.get("lat"),
                        "lon": row.get("lon"),
                        "nearby_services": row.get("nearby_services", []),
                    },
                )
            )
        return fallback_docs

    def _infer_property_type_from_query(self, query: str) -> Optional[str]:
        raw_query = str(query or "").lower()
        if not raw_query:
            return None

        mapping = {
            "apartment": ["apartment", "apartments", "flat", "شقة", "شقق"],
            "villa": ["villa", "villas", "فيلا", "فلل"],
            "duplex": ["duplex", "دوبلكس"],
            "chalet": ["chalet", "شاليه", "شاليهات"],
            "townhouse": ["townhouse", "تاون هاوس"],
        }
        for canonical, tokens in mapping.items():
            if any(token in raw_query for token in tokens):
                return canonical
        return None

    def _infer_listing_intent_from_query(self, query: str) -> Optional[str]:
        raw_query = str(query or "").lower()
        if not raw_query:
            return None

        rent_tokens = ["rent", "rental", "lease", "للإيجار", "للايجار", "إيجار", "ايجار"]
        buy_tokens = ["buy", "sale", "for sale", "للبيع", "شراء", "بيع", "تمليك"]
        if any(token in raw_query for token in rent_tokens):
            return "rent"
        if any(token in raw_query for token in buy_tokens):
            return "buy"
        return None

    @staticmethod
    def _property_type_label(property_type: Any) -> str:
        token = str(property_type or "").strip().lower()
        labels = {
            "apartment": "Apartment / شقة",
            "villa": "Villa / فيلا",
            "duplex": "Duplex / دوبلكس",
            "chalet": "Chalet / شاليه",
            "townhouse": "Townhouse / تاون هاوس",
        }
        return labels.get(token, token or "غير محدد")

    def _is_analysis_intent(self, query: str, filters: Optional[Dict[str, Any]] = None) -> bool:
        raw_query = str(query or "").strip().lower()
        if not raw_query:
            return False

        analysis_tokens = [
            "analysis",
            "analyze",
            "buy",
            "invest",
            "worth",
            "تحليل",
            "حلل",
            "اناليسز",
            "اشتري",
            "أشتري",
            "اشترى",
            "شراء",
            "استثمار",
            "قرار",
            "ولا لا",
            "ينفع",
        ]
        market_tokens = [
            "property",
            "real estate",
            "apartment",
            "villa",
            "compound",
            "عقار",
            "عقاري",
            "شقة",
            "فيلا",
            "كمبوند",
            "سوق",
            "منطقة",
            "تجمع",
            "زايد",
            "العاصمة",
        ]

        has_analysis_token = any(token in raw_query for token in analysis_tokens)
        if not has_analysis_token:
            return False

        has_market_token = any(token in raw_query for token in market_tokens)
        has_filter_context = any(
            value not in (None, "", [], {})
            for value in (filters or {}).values()
        )
        return has_market_token or has_filter_context

    def _collect_analysis_docs(self, analysis_result: Dict[str, Any], max_docs: int = 5) -> List[Document]:
        docs: List[Document] = []
        seen: set[str] = set()

        candidates: List[Any] = [analysis_result.get("better_option_doc")]
        candidates.extend(list(analysis_result.get("sample_docs", []) or []))

        for candidate in candidates:
            if not isinstance(candidate, Document):
                continue
            meta = candidate.metadata or {}
            key = str(meta.get("url") or meta.get("title") or id(candidate))
            if key in seen:
                continue
            seen.add(key)
            docs.append(candidate)
            if len(docs) >= max_docs:
                break
        return docs

    def _build_analysis_chat_response(self, analysis_result: Dict[str, Any]) -> str:
        insight = str(analysis_result.get("insight") or "").strip()
        matched_count = int(self._safe_float(analysis_result.get("matched_count")))
        total_candidates = int(self._safe_float(analysis_result.get("total_candidates")))

        decision = analysis_result.get("buy_decision") or {}
        headline = str(decision.get("headline") or "قرار الشراء غير واضح حاليًا.").strip()
        confidence = self._safe_float(decision.get("confidence"))
        reasons = [
            str(reason).strip()
            for reason in (decision.get("reasons") or [])
            if str(reason).strip()
        ]

        lines: List[str] = []
        if insight:
            lines.append(insight)
        lines.append(f"قرار الشراء: {headline} (ثقة {confidence * 100:.0f}%).")
        if total_candidates > 0 or matched_count > 0:
            lines.append(f"تم التحليل على {matched_count} نتيجة مناسبة من أصل {total_candidates}.")

        if reasons:
            lines.append("أهم الأسباب:")
            for reason in reasons[:4]:
                lines.append(f"- {reason}")

        better_option_found = bool(analysis_result.get("better_option_found"))
        better_option_reason = str(analysis_result.get("better_option_reason") or "").strip()
        better_doc = analysis_result.get("better_option_doc")
        if better_option_found and isinstance(better_doc, Document):
            meta = better_doc.metadata or {}
            title = str(meta.get("title") or "عقار مرشح").strip()
            location = str(meta.get("location") or "غير محدد").strip()
            price = self._safe_float(meta.get("price"))
            price_text = f"{price:,.0f} جنيه" if price > 0 else "السعر غير متاح"
            lines.append(f"أفضل بديل حاليًا: {title} في {location} بسعر {price_text}.")
            if better_option_reason:
                lines.append(f"سبب الترشيح: {better_option_reason}")
        elif better_option_reason:
            lines.append(f"ملاحظة البديل: {better_option_reason}")

        return "\n".join(lines).strip()

    def _apply_exact_filters(self, docs: List[Document], filters: Dict) -> List[Document]:
        """Applies rigorous constraints to the raw vector results."""
        filtered = docs
        
        def safe_float(val):
            try: return float(val)
            except: return 0.0

        if filters.get('location'):
            target_loc = filters['location'].lower()
            filtered = [d for d in filtered if target_loc in d.metadata.get('location', '').lower()]

        if filters.get('min_price') is not None:
            min_p = safe_float(filters['min_price'])
            filtered = [d for d in filtered if safe_float(d.metadata.get('price', 0)) >= min_p]

        if filters.get('min_bedrooms') is not None:
            min_b = safe_float(filters['min_bedrooms'])
            filtered = [d for d in filtered if safe_float(d.metadata.get('bedrooms', 0)) >= min_b]

        if filters.get('max_bedrooms') is not None:
            max_b = safe_float(filters['max_bedrooms'])
            filtered = [d for d in filtered if safe_float(d.metadata.get('bedrooms', 0)) <= max_b]
        
        if filters.get('max_price') is not None:
            max_p = safe_float(filters['max_price'])
            # 10% budget wiggle room
            filtered = [d for d in filtered if safe_float(d.metadata.get('price', 0)) <= max_p * 1.1]

        if filters.get('property_type'):
            ptype = filters['property_type'].lower()
            filtered = [d for d in filtered if ptype in d.metadata.get('type', '').lower() or ptype in d.metadata.get('title', '').lower()]

        requested_intent = self._normalize_listing_intent(filters.get("listing_intent"))
        if requested_intent:
            filtered = [
                d for d in filtered
                if self._doc_listing_intent(d) == requested_intent
            ]

        desired_services = set(self._normalize_services(filters.get("desired_services", [])))
        if desired_services and filtered:
            service_matched = []
            for doc in filtered:
                available = set(self._normalize_services(doc.metadata.get("nearby_services", [])))
                if desired_services.intersection(available):
                    service_matched.append(doc)
            if service_matched:
                filtered = service_matched

        return filtered

    def _expand_strict_matches(self, raw_docs: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """
        Applies exact filters to semantic results, then expands from the full in-memory
        index when filters are present. This keeps query retrieval semantic, while not
        losing good exact matches just because they were outside the first vector hits.
        """
        filtered_docs = self._apply_exact_filters(raw_docs, filters)
        if not self._has_active_filters(filters):
            return filtered_docs

        all_docs = getattr(self.vector_store, "all_docs_list", []) or []
        if not all_docs:
            return filtered_docs

        strict_pool = self._apply_exact_filters(all_docs, filters)
        try:
            strict_pool = self.vector_store._enrich_docs(strict_pool)
        except Exception:
            pass

        return self._merge_unique_docs(filtered_docs, strict_pool)

    def _merge_unique_docs(self, *doc_groups: List[Document]) -> List[Document]:
        merged: List[Document] = []
        seen = set()
        for docs in doc_groups:
            for doc in docs or []:
                key = self._doc_identity(doc)
                if key in seen:
                    continue
                merged.append(doc)
                seen.add(key)
        return merged

    def _doc_identity(self, doc: Document) -> str:
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        return str(
            meta.get("url")
            or meta.get("property_id")
            or meta.get("id")
            or meta.get("title")
            or id(doc)
        )

    def _sanitize_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(filters, dict):
            return {}

        cleaned = dict(filters)
        cleaned["location"] = self._normalize_location_filter(cleaned.get("location"))
        cleaned["listing_intent"] = self._normalize_listing_intent(cleaned.get("listing_intent"))
        cleaned["property_type"] = self._normalize_optional_text_filter(cleaned.get("property_type"))
        for key in ("min_price", "max_price"):
            cleaned[key] = self._normalize_positive_number_filter(cleaned.get(key))
        for key in ("min_bedrooms", "max_bedrooms"):
            value = self._normalize_positive_number_filter(cleaned.get(key))
            cleaned[key] = int(value) if value is not None else None
        services = cleaned.get("desired_services")
        if isinstance(services, str):
            services = [s.strip() for s in services.split(",") if s.strip()]
        if isinstance(services, list):
            services = [
                item
                for item in services
                if self._normalize_optional_text_filter(item) is not None
            ]
            cleaned["desired_services"] = self._normalize_services(services)
        elif services is not None:
            cleaned["desired_services"] = []
        return cleaned

    def _normalize_location_filter(self, location: Any) -> Any:
        if location is None:
            return None

        raw = str(location).strip()
        if not raw:
            return None
        raw_lower = raw.lower()
        if raw_lower in {"string", "none", "null", "undefined", "optional"}:
            return None

        aliases = {
            "القاهرة": "Cairo",
            "cairo": "Cairo",
            "التجمع": "New Cairo",
            "القاهرة الجديدة": "New Cairo",
            "التجمع الخامس": "The 5th Settlement",
            "التجمع الاول": "The 1st Settlement",
            "الشيخ زايد": "Sheikh Zayed",
            "زايد": "Sheikh Zayed",
            "اكتوبر": "October",
            "٦ اكتوبر": "October",
            "الساحل": "North Coast",
            "الساحل الشمالي": "North Coast",
            "العاصمة الادارية": "New Capital",
            "العاصمة": "New Capital",
            "new cairo": "New Cairo",
            "the 5th settlement": "The 5th Settlement",
            "the 1st settlement": "The 1st Settlement",
            "sheikh zayed": "Sheikh Zayed",
            "6th of october": "October",
            "october": "October",
            "north coast": "North Coast",
            "new capital": "New Capital",
        }

        for token, canonical in aliases.items():
            if token in raw_lower:
                return canonical

        available_locations = getattr(getattr(self, "vector_store", None), "available_locations", set()) or set()
        for candidate in available_locations:
            candidate_text = str(candidate).strip()
            if not candidate_text:
                continue
            candidate_lower = candidate_text.lower()
            if raw_lower == candidate_lower:
                return candidate_text

        return raw

    @staticmethod
    def _normalize_optional_text_filter(value: Any) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        if raw.lower() in {"string", "none", "null", "undefined", "optional"}:
            return None
        return raw

    @staticmethod
    def _normalize_positive_number_filter(value: Any) -> Optional[float]:
        if value in (None, "", [], {}):
            return None
        try:
            parsed = float(value)
        except Exception:
            return None
        if parsed <= 0:
            return None
        return parsed

    def _normalize_listing_intent(self, listing_intent: Any) -> Optional[str]:
        if listing_intent is None:
            return None
        raw = str(listing_intent).strip().lower()
        if not raw:
            return None

        if raw in {"rent", "rental", "lease", "ايجار", "إيجار", "للايجار", "للإيجار"}:
            return "rent"
        if raw in {"buy", "sale", "sell", "شراء", "بيع", "للبيع", "تمليك"}:
            return "buy"
        return None

    def _doc_listing_intent(self, doc: Document) -> str:
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        explicit = self._normalize_listing_intent(meta.get("listing_intent"))
        if explicit:
            return explicit

        url = str(meta.get("url", "")).lower()
        if "/rent/" in url:
            return "rent"
        if "/buy/" in url:
            return "buy"
        return "unknown"

    def _rank_recommendations(self, docs: List[Document], filters: Dict) -> List[Document]:
        """
        Re-ranks documents by combining affordability, value, distance, and nearby services.
        """
        if not docs:
            return []

        reference = self._resolve_reference_point(docs, filters)
        budget = self._safe_float(filters.get("max_price"))
        desired_services = set(self._normalize_services(filters.get("desired_services", [])))
        median_ppsqm = self._median_price_per_sqm(docs)

        scored_docs = []
        max_live_service_docs = self._get_max_live_service_docs()
        for idx, doc in enumerate(docs):
            price = self._safe_float(doc.metadata.get("price"))
            size = self._safe_float(doc.metadata.get("size"))
            lat, lon = self._extract_lat_lon(doc)
            ppsqm = (price / size) if price > 0 and size > 0 else 0.0

            distance_km = None
            if reference and lat is not None and lon is not None:
                distance_km = self._haversine_km(reference[0], reference[1], lat, lon)
                doc.metadata["distance_km"] = round(distance_km, 2)

            services = self._normalize_services(doc.metadata.get("nearby_services", []))
            if desired_services and idx < max_live_service_docs and lat is not None and lon is not None:
                api_services = self._get_live_services(lat, lon)
                if api_services:
                    services = self._normalize_services(services + api_services)
            doc.metadata["nearby_services"] = services

            distance_score = 0.45 if distance_km is None else max(0.0, 1 - min(distance_km, 25.0) / 25.0)

            if desired_services:
                service_matches = desired_services.intersection(set(services))
                service_score = len(service_matches) / len(desired_services)
                doc.metadata["service_match_count"] = len(service_matches)
            else:
                service_score = min(len(services) / 5.0, 1.0)

            if budget > 0 and price > 0:
                if price <= budget:
                    budget_score = 1.0
                elif price <= budget * 1.1:
                    budget_score = 0.7
                else:
                    budget_score = max(0.0, 1 - ((price - budget) / budget))
            else:
                budget_score = 0.5

            if median_ppsqm > 0 and ppsqm > 0:
                if ppsqm <= median_ppsqm * 0.9:
                    value_score = 1.0
                else:
                    value_score = max(0.0, 1 - ((ppsqm - median_ppsqm) / median_ppsqm))
            else:
                value_score = 0.5

            score = (
                (0.35 * distance_score)
                + (0.25 * service_score)
                + (0.25 * value_score)
                + (0.15 * budget_score)
            )

            requested_type = (filters.get("property_type") or "").strip().lower()
            actual_type = str(doc.metadata.get("type", "")).lower()
            if requested_type and requested_type in actual_type:
                score += 0.05

            doc.metadata["recommendation_score"] = round(score, 3)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs]

    def _resolve_reference_point(
        self, docs: List[Document], filters: Dict
    ) -> Optional[Tuple[float, float]]:
        requested_location = str(filters.get("location", "") or "").lower().strip()

        prioritized: List[Tuple[float, float]] = []
        fallback: List[Tuple[float, float]] = []
        for doc in docs:
            lat, lon = self._extract_lat_lon(doc)
            if lat is None or lon is None:
                continue
            fallback.append((lat, lon))
            doc_location = str(doc.metadata.get("location", "")).lower()
            if requested_location and requested_location in doc_location:
                prioritized.append((lat, lon))

        pool = prioritized if prioritized else fallback
        if pool:
            lat_center = sum(p[0] for p in pool) / len(pool)
            lon_center = sum(p[1] for p in pool) / len(pool)
            return lat_center, lon_center

        return self._get_requested_location_center(requested_location)

    def _get_requested_location_center(
        self, requested_location: str
    ) -> Optional[Tuple[float, float]]:
        if not requested_location:
            return None
        map_service = getattr(self, "map_intelligence", None)
        if map_service is None:
            return None
        try:
            return map_service.geocode_area_center(requested_location)
        except Exception as e:
            logger.debug(f"Location center API fallback due to error: {e}")
            return None

    def _get_live_services(self, lat: float, lon: float) -> List[str]:
        map_service = getattr(self, "map_intelligence", None)
        if map_service is None:
            return []
        try:
            return map_service.get_nearby_services(lat, lon)
        except Exception as e:
            logger.debug(f"Nearby services API fallback due to error: {e}")
            return []

    def _get_max_live_service_docs(self) -> int:
        map_service = getattr(self, "map_intelligence", None)
        if map_service is None:
            return 0
        try:
            return int(getattr(map_service, "max_docs_per_rank", 0) or 0)
        except Exception:
            return 0

    def _extract_lat_lon(self, doc: Document) -> Tuple[Optional[float], Optional[float]]:
        lat = self._safe_float(
            doc.metadata.get("lat")
            or doc.metadata.get("latitude")
        )
        lon = self._safe_float(
            doc.metadata.get("lon")
            or doc.metadata.get("longitude")
        )
        if lat == 0.0 or lon == 0.0:
            return None, None
        return lat, lon

    def _median_price_per_sqm(self, docs: List[Document]) -> float:
        values: List[float] = []
        for doc in docs:
            price = self._safe_float(doc.metadata.get("price"))
            size = self._safe_float(doc.metadata.get("size"))
            if price > 0 and size > 0:
                values.append(price / size)
        return median(values) if values else 0.0

    def _normalize_services(self, services: Any) -> List[str]:
        if services is None:
            return []
        if isinstance(services, list):
            raw_items = services
        elif isinstance(services, str):
            raw_items = [s.strip() for s in services.split(",") if s.strip()]
        else:
            raw_items = []

        alias_map = {
            "school": "schools",
            "schools": "schools",
            "hospital": "hospitals",
            "clinic": "hospitals",
            "hospitals": "hospitals",
            "transport": "transport",
            "metro": "transport",
            "mall": "commercial_area",
            "commercial": "commercial_area",
            "commercial_area": "commercial_area",
            "security": "security",
            "club": "club_house",
            "clubhouse": "club_house",
            "club_house": "club_house",
            "green": "green_spaces",
            "green_spaces": "green_spaces",
            "swimming_pool": "swimming_pool",
            "pool": "swimming_pool",
        }

        normalized = []
        for item in raw_items:
            token = str(item).strip().lower().replace(" ", "_")
            token = alias_map.get(token, token)
            if token and token not in normalized:
                normalized.append(token)
        return normalized

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius_km = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * (math.sin(d_lambda / 2) ** 2)
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius_km * c

    def _compute_market_stats(self, docs: List[Document]) -> Dict[str, Any]:
        prices: List[float] = []
        sizes: List[float] = []
        price_per_sqm: List[float] = []
        bedrooms: List[float] = []
        bathrooms: List[float] = []

        for doc in docs:
            meta = doc.metadata
            price = self._safe_float(meta.get("price"))
            size = self._safe_float(meta.get("size"))
            beds = self._safe_float(meta.get("bedrooms"))
            baths = self._safe_float(meta.get("bathrooms"))

            if price > 0:
                prices.append(price)
            if size > 0:
                sizes.append(size)
            if price > 0 and size > 0:
                price_per_sqm.append(price / size)
            if beds > 0:
                bedrooms.append(beds)
            if baths > 0:
                bathrooms.append(baths)

        return {
            "count": len(docs),
            "avg_price": round(mean(prices), 2) if prices else 0.0,
            "median_price": round(median(prices), 2) if prices else 0.0,
            "min_price": round(min(prices), 2) if prices else 0.0,
            "max_price": round(max(prices), 2) if prices else 0.0,
            "avg_size_sqm": round(mean(sizes), 2) if sizes else 0.0,
            "avg_price_per_sqm": round(mean(price_per_sqm), 2) if price_per_sqm else 0.0,
            "avg_bedrooms": round(mean(bedrooms), 2) if bedrooms else 0.0,
            "avg_bathrooms": round(mean(bathrooms), 2) if bathrooms else 0.0,
        }

    def _compute_segment_stats(self, docs: List[Document], segment_key: str, top_k: int = 5) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Document]] = {}

        for doc in docs:
            value = doc.metadata.get(segment_key)
            if segment_key == "type" and value is None:
                value = doc.metadata.get("property_type")
            name = str(value or "Unknown").strip()
            grouped.setdefault(name, []).append(doc)

        segments: List[Dict[str, Any]] = []
        for name, seg_docs in grouped.items():
            prices: List[float] = []
            ppsqm: List[float] = []

            for doc in seg_docs:
                price = self._safe_float(doc.metadata.get("price"))
                size = self._safe_float(doc.metadata.get("size"))
                if price > 0:
                    prices.append(price)
                if price > 0 and size > 0:
                    ppsqm.append(price / size)

            segments.append(
                {
                    "name": name,
                    "count": len(seg_docs),
                    "avg_price": round(mean(prices), 2) if prices else 0.0,
                    "median_price": round(median(prices), 2) if prices else 0.0,
                    "avg_price_per_sqm": round(mean(ppsqm), 2) if ppsqm else 0.0,
                }
            )

        segments.sort(key=lambda item: item["count"], reverse=True)
        return segments[:top_k]

    def _build_market_insight(
        self,
        stats: Dict[str, Any],
        top_locations: List[Dict[str, Any]],
        top_property_types: List[Dict[str, Any]],
        filters: Dict[str, Any],
        match_scope: str,
    ) -> str:
        if stats["count"] == 0:
            return "ملقتش بيانات كفاية للتحليل بالمدخلات الحالية."

        lead = (
            "التحليل مبني على أقرب نتائج دلالية لأن مفيش تطابق كامل."
            if match_scope == "semantic_fallback"
            else (
                "التحليل حافظ على نفس المنطقة المطلوبة مع تخفيف بعض القيود."
                if match_scope == "location_fallback"
                else "التحليل مبني على نتائج متطابقة مع طلبك."
            )
        )

        top_loc = top_locations[0]["name"] if top_locations else "غير محدد"
        top_type = top_property_types[0]["name"] if top_property_types else "غير محدد"
        filtered_scope = f"الفلاتر المستخدمة: {filters}." if filters else "بدون فلاتر صريحة."
        desired_services = filters.get("desired_services", [])
        services_hint = (
            f" الأولوية الخدمية المطلوبة: {', '.join(desired_services)}."
            if desired_services
            else ""
        )

        return (
            f"{lead} تم تحليل {stats['count']} عقار، متوسط السعر {stats['avg_price']:,.0f} جنيه "
            f"ووسيط السعر {stats['median_price']:,.0f} جنيه، ومتوسط سعر المتر {stats['avg_price_per_sqm']:,.0f} جنيه. "
            f"أكثر منطقة ظهورًا: {top_loc}، وأكثر نوع متكرر: {top_type}. {filtered_scope}{services_hint}"
        )

    def _compute_buy_decision(
        self,
        docs: List[Document],
        stats: Dict[str, Any],
        filters: Dict[str, Any],
        match_scope: str,
    ) -> Dict[str, Any]:
        """
        Produces a practical buy/no-buy signal using affordability and value indicators.
        """
        count = len(docs)
        if count < 3 or stats.get("avg_price", 0.0) <= 0:
            return {
                "decision": "insufficient_data",
                "headline": "بيانات غير كافية لاتخاذ قرار شراء",
                "confidence": 0.35,
                "reasons": [
                    "عدد العقارات المطابقة قليل جدًا للتحليل الموثوق.",
                    "يفضل توسيع الفلاتر أو تعديل المنطقة لرفع دقة القرار.",
                ],
            }

        budget = self._safe_float(filters.get("max_price"))
        avg_ppsqm = self._safe_float(stats.get("avg_price_per_sqm"))

        priced_docs = [d for d in docs if self._safe_float(d.metadata.get("price")) > 0]
        if not priced_docs:
            return {
                "decision": "insufficient_data",
                "headline": "بيانات السعر غير مكتملة",
                "confidence": 0.35,
                "reasons": ["لا توجد أسعار كافية داخل النتائج لاتخاذ قرار شراء."],
            }

        affordable_docs = priced_docs
        if budget > 0:
            affordable_docs = [
                d for d in priced_docs
                if self._safe_float(d.metadata.get("price")) <= budget * 1.05
            ]

        value_docs = []
        if avg_ppsqm > 0:
            for doc in priced_docs:
                price = self._safe_float(doc.metadata.get("price"))
                size = self._safe_float(doc.metadata.get("size"))
                if size <= 0:
                    continue
                if (price / size) <= avg_ppsqm * 0.95:
                    value_docs.append(doc)

        affordable_ratio = len(affordable_docs) / len(priced_docs)
        value_ratio = (len(value_docs) / len(priced_docs)) if priced_docs else 0.0

        reasons: List[str] = []
        if budget > 0:
            reasons.append(f"{len(affordable_docs)} من {len(priced_docs)} عقار ضمن أو قريب من ميزانيتك.")
        else:
            reasons.append("لم يتم تحديد ميزانية، لذلك التوصية مبنية على القيمة السوقية وسعر المتر.")
        reasons.append(
            f"متوسط سعر المتر في النتائج {avg_ppsqm:,.0f} جنيه، ونسبة الفرص الأقل من المتوسط {value_ratio * 100:.0f}%."
            if avg_ppsqm > 0
            else "متوسط سعر المتر غير متاح بدقة كافية في النتائج."
        )

        top_doc = docs[0] if docs else None
        if top_doc:
            top_distance = top_doc.metadata.get("distance_km")
            if isinstance(top_distance, (int, float)):
                reasons.append(f"أقوى ترشيح يبعد تقريبًا {top_distance:.1f} كم عن مركز الطلب.")
            top_services = self._normalize_services(top_doc.metadata.get("nearby_services", []))
            if top_services:
                reasons.append(f"الخدمات المتوفرة حول الترشيح الأفضل: {', '.join(top_services[:4])}.")

        if match_scope == "semantic_fallback":
            reasons.append("القرار مبني على أقرب نتائج دلالية لأن التطابق الدقيق محدود.")
        elif match_scope == "location_fallback":
            reasons.append("القرار مبني على نفس المنطقة المطلوبة مع تخفيف بعض القيود (مثل الميزانية/عدد الغرف).")
        else:
            reasons.append("القرار مبني على نتائج متطابقة مع الفلاتر المطلوبة.")

        if budget > 0:
            if affordable_ratio >= 0.5 and value_ratio >= 0.3:
                decision = "buy_now"
                headline = "مؤشرات الشراء إيجابية حاليًا"
                confidence = 0.83
            elif affordable_ratio >= 0.25:
                decision = "wait_or_negotiate"
                headline = "الأفضل تفاوض قوي أو انتظار فرصة أحسن"
                confidence = 0.72
            else:
                decision = "not_now"
                headline = "غير مناسب للشراء حاليًا بالميزانية الحالية"
                confidence = 0.87
        else:
            if value_ratio >= 0.45:
                decision = "buy_now"
                headline = "الأسعار الظاهرة فيها فرص شراء جيدة"
                confidence = 0.75
            elif value_ratio >= 0.25:
                decision = "wait_or_negotiate"
                headline = "في فرص متوسطة، قارن وتفاوض قبل الشراء"
                confidence = 0.66
            else:
                decision = "wait_or_negotiate"
                headline = "الأفضل الانتظار أو إعادة تحديد معاييرك"
                confidence = 0.62

        count_boost = min(0.08, count / 200)
        confidence = round(min(0.95, confidence + count_boost), 2)

        return {
            "decision": decision,
            "headline": headline,
            "confidence": confidence,
            "reasons": reasons,
        }

    def _identify_better_option(
        self,
        docs: List[Document],
        stats: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Picks a standout listing from ranked docs and explains why it is a stronger option.
        """
        if not docs:
            return {
                "found": False,
                "reason": "لا توجد نتائج كافية لاختيار بديل أفضل.",
                "doc": None,
            }

        candidate = docs[0]
        price = self._safe_float(candidate.metadata.get("price"))
        size = self._safe_float(candidate.metadata.get("size"))
        ppsqm = (price / size) if price > 0 and size > 0 else 0.0
        avg_ppsqm = self._safe_float(stats.get("avg_price_per_sqm"))
        budget = self._safe_float(filters.get("max_price"))
        distance = candidate.metadata.get("distance_km")
        services = self._normalize_services(candidate.metadata.get("nearby_services", []))

        signals = 0
        reasons: List[str] = []

        if budget > 0 and price > 0 and price <= budget * 1.05:
            signals += 1
            reasons.append("ضمن نطاق الميزانية أو قريب جدًا منه.")
        elif budget > 0 and price > budget * 1.05:
            reasons.append("سعره أعلى من الميزانية المحددة.")

        if avg_ppsqm > 0 and ppsqm > 0 and ppsqm <= avg_ppsqm * 0.95:
            signals += 1
            reasons.append("سعر المتر أقل من متوسط السوق في النتائج.")
        elif ppsqm > 0 and avg_ppsqm > 0:
            reasons.append("سعر المتر قريب من متوسط السوق.")

        if isinstance(distance, (int, float)) and distance <= 6:
            signals += 1
            reasons.append(f"قريب جغرافيًا من محور الطلب (حوالي {distance:.1f} كم).")
        elif isinstance(distance, (int, float)):
            reasons.append(f"المسافة متوسطة (حوالي {distance:.1f} كم).")

        if len(services) >= 2:
            signals += 1
            reasons.append("متوفر حوله خدمات متعددة.")

        found = signals >= 2
        if not reasons:
            reasons.append("لا توجد مؤشرات تفضيل كافية مقارنة بباقي النتائج.")

        return {
            "found": found,
            "reason": " ".join(reasons),
            "doc": candidate if found else None,
        }

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _safe_int_value(value: Any) -> int:
        try:
            return int(float(value))
        except Exception:
            return 0

    @staticmethod
    def _unique_ordered(values: List[int]) -> List[int]:
        seen = set()
        unique: List[int] = []
        for value in values:
            if value and value not in seen:
                unique.append(value)
                seen.add(value)
        return unique

    def _doc_property_id(self, doc: Document) -> int:
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        return self._safe_int_value(meta.get("property_id") or meta.get("id"))

    @staticmethod
    def _recommendation_seed_text(doc: Document) -> str:
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        return (
            f"Title: {meta.get('title', '')}\n"
            f"Location: {meta.get('location', '')}\n"
            f"Type: {meta.get('type', '')}\n"
            f"Price: {meta.get('price', '')}\n"
            f"Bedrooms: {meta.get('bedrooms', '')}\n"
            f"Size: {meta.get('size', '')}\n"
            f"{doc.page_content}"
        )

    def _invalidate_session_recommendation_cache(self, session_id: str):
        prefix = f"{session_id}:"
        for key in list(self.recommendation_cache.keys()):
            if key.startswith(prefix):
                self.recommendation_cache.pop(key, None)

    def _trim_recommendation_cache(self):
        max_entries = 500
        if len(self.recommendation_cache) <= max_entries:
            return
        sorted_items = sorted(self.recommendation_cache.items(), key=lambda item: item[1][0])
        for key, _ in sorted_items[: len(self.recommendation_cache) - max_entries]:
            self.recommendation_cache.pop(key, None)

    def _enforce_padding_logic(self, query, filters, filtered_docs, raw_docs, max_results: Optional[int] = None):
        """
        Calculates search status and returns the best matching result set.
        Enforces strict geographical boundaries when backing off constraints.
        """
        result_limit = self._result_limit(max_results, settings.chat_result_limit, settings.search_retrieval_k)
        search_status = "Excellent Match: We found properties that match exactly your request."
        requested_intent = self._normalize_listing_intent(filters.get("listing_intent"))
        
        if len(filtered_docs) == 0:
            if requested_intent:
                available_with_intent = [d for d in raw_docs if self._doc_listing_intent(d) == requested_intent]
                if not available_with_intent:
                    intent_label = "rental" if requested_intent == "rent" else "sale"
                    search_status = f"Alert: No {intent_label} listings are available in the current database for your request."
                    return search_status, []

            requested_loc = filters.get('location')
            if requested_loc:
                # If a location was requested, we MUST stay within that location boundary.
                # If we have no matches for that location, we return an empty list and alert the user.
                target_padding_loc = requested_loc
                filtered_docs = [d for d in raw_docs if target_padding_loc.lower() in d.metadata.get('location', '').lower()]
                if not filtered_docs:
                    all_docs = getattr(self.vector_store, 'all_docs_list', []) or []
                    filtered_docs = [d for d in all_docs if target_padding_loc.lower() in d.metadata.get('location', '').lower()]
                
                if not filtered_docs:
                    search_status = f"Alert: You requested ({requested_loc}) which is currently unavailable."
                    return search_status, []
                
                # If we found properties in that location but they didn't match other constraints,
                # we show these as closest alternatives in the same location.
                search_status = f"Alert: The exact specifications (rooms/price) in {requested_loc} are unavailable. Presenting these alternatives in the same location."
                filtered_docs = filtered_docs[:result_limit]
            else:
                search_status = "Alert: The exact specifications (rooms/price) are unavailable. Present these closest semantic alternatives instead."
                if requested_intent:
                    filtered_docs = [d for d in raw_docs if self._doc_listing_intent(d) == requested_intent]
                
                # Absolute Fallback: Global semantic search
                if not filtered_docs:
                    filtered_docs = [
                        d for d in raw_docs
                        if (not requested_intent) or self._doc_listing_intent(d) == requested_intent
                    ][:result_limit]
                else:
                    filtered_docs = filtered_docs[:result_limit]
                
        else:
            filtered_docs = filtered_docs[:result_limit]
            
        return search_status, filtered_docs

    @staticmethod
    def _result_limit(value: Any, default: int, hard_max: int) -> int:
        try:
            parsed = int(value or default)
        except Exception:
            parsed = default
        return max(1, min(parsed, hard_max))

    def _format_price_text_en(self, value: Any) -> str:
        price = self._safe_float(value)
        if price <= 0:
            return "Unspecified Price"
        if price >= 1_000_000:
            amount = price / 1_000_000
            text = f"{amount:.1f}".rstrip("0").rstrip(".")
            return f"{text} Million EGP"
        return f"{price:,.0f} EGP"

    def _generate_fast_property_response(self, query, search_status, final_docs, history=None):
        """Builds a low-latency response from ranked docs matching the query's language (Arabic or English)."""
        if history is None:
            history = []

        is_arabic_query = bool(re.search(r"[\u0600-\u06FF]", query))

        if not final_docs:
            if is_arabic_query:
                content = "للأسف مش لاقي نتائج مناسبة للطلب ده حاليًا في قاعدة البيانات 😕\nجرّب توسّع المنطقة أو الميزانية شوية وأنا هساعدك."
            else:
                content = "Unfortunately, I couldn't find matching properties in the database right now 😕\nTry widening the location or budget, and I'll help you search."
            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=content))
            return content, []

        status_text = str(search_status or "")
        if is_arabic_query:
            if status_text.startswith("Excellent Match"):
                opening = "إليك أفضل النتائج المطابقة لطلبك 🏠✨"
            elif status_text.startswith("Partial Match"):
                opening = "لقيتلك نتائج قريبة جدًا من طلبك، بص عليها 👇"
            else:
                opening = "مفيش تطابق كامل بنفس الشروط، بس دي أقرب بدائل ليك 🔍"
        else:
            if status_text.startswith("Excellent Match"):
                opening = "Here are the best matching properties for your request 🏠✨"
            elif status_text.startswith("Partial Match"):
                opening = "I found some very close options for you, take a look 👇"
            else:
                opening = "No exact match found, but here are the closest alternatives 🔍"

        summary_count = self._result_limit(
            settings.fast_response_summary_items,
            5,
            max(1, len(final_docs)),
        )
        lines = [opening, ""]
        for idx, doc in enumerate(final_docs[:summary_count], start=1):
            meta = doc.metadata if isinstance(doc.metadata, dict) else {}
            title = str(meta.get("title") or ("عقار مناسب" if is_arabic_query else "Suitable Property")).strip()
            location = str(meta.get("location") or ("منطقة غير محددة" if is_arabic_query else "Unspecified Location")).strip()
            bedrooms = self._safe_int_value(meta.get("bedrooms"))
            size = self._safe_int_value(meta.get("size"))
            specs = []

            if is_arabic_query:
                price = self._format_price_text(meta.get("price"))
                if bedrooms:
                    specs.append(f"{bedrooms} غرف")
                if size:
                    specs.append(f"{size} م²")
                spec_text = f" ({' • '.join(specs)})" if specs else ""
                lines.append(f"{idx}. **{title}** — {location}\n   💰 {price}{spec_text}")
            else:
                price = self._format_price_text_en(meta.get("price"))
                if bedrooms:
                    specs.append(f"{bedrooms} BR")
                if size:
                    specs.append(f"{size} sqm")
                spec_text = f" ({' • '.join(specs)})" if specs else ""
                lines.append(f"{idx}. **{title}** — {location}\n   💰 {price}{spec_text}")

        best_doc = final_docs[0]
        best_meta = best_doc.metadata if isinstance(best_doc.metadata, dict) else {}
        best_title = str(best_meta.get("title") or ("أول اختيار" if is_arabic_query else "Top Pick")).strip()
        services = self._normalize_services(best_meta.get("nearby_services", []))
        lines.append("")
        if is_arabic_query:
            if services:
                lines.append(f"⭐ أقوى ترشيح: **{best_title}** — خدمات قريبة: {', '.join(services[:3])}")
            else:
                lines.append(f"⭐ أقوى ترشيح: **{best_title}** — الأقرب لشروطك بين النتائج")
        else:
            if services:
                lines.append(f"⭐ Top Recommendation: **{best_title}** — Nearby services: {', '.join(services[:3])}")
            else:
                lines.append(f"⭐ Top Recommendation: **{best_title}** — The closest to your requirements")

        if len(final_docs) > summary_count:
            if is_arabic_query:
                lines.append(f"\n📋 إجمالي {len(final_docs)} نتيجة متاحة، مرتبين بالأفضل.")
            else:
                lines.append(f"\n📋 Total of {len(final_docs)} results available, ordered by preference.")

        content = "\n".join(lines).strip()
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=content))
        return content, final_docs

    def _format_price_text(self, value: Any) -> str:
        price = self._safe_float(value)
        if price <= 0:
            return "سعر غير محدد"
        if price >= 1_000_000:
            amount = price / 1_000_000
            text = f"{amount:.1f}".rstrip("0").rstrip(".")
            return f"{text} مليون جنيه"
        return f"{price:,.0f} جنيه"

    def _generate_response(self, query, search_status, final_docs, history=None):
        """Generates conversational text with deep Chat History Memory."""
        if history is None:
            history = []
        try:
            llm = self.llm_manager.get_llm()
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            return "آسف، الخدمة الذكية غير متاحة حاليًا. حاول مرة تانية بعد شوية.", []
        
        template = """
        You are "AqarAI", a premium Real Estate AI Consultant.
        
        LANGUAGE RULE:
        - Detect the language of [User Query].
        - If the query is in Arabic (or Egyptian Arabic), respond in elegant, confident Egyptian Arabic dialect. Translate any English property names, descriptions, or locations to Egyptian Arabic (e.g. explain/translate the English titles into Arabic).
        - If the query is in English, respond in professional, friendly English. Translate any Arabic property names, descriptions, or locations to English.

        [Search Status]: {search_status}
        [Properties]: {context}
        [User Query]: {question}

        STRICT RULES:
        1. Start with a warm, polished opening (e.g. "إليك أفضل النتائج لطلبك 🏠" or "لقيتلك اختيارات ممتازة 👇" in Arabic, or "Here are the best options for your request 🏠" in English).
        2. Summarize the top 2-3 properties attractively — mention title, location, price, and key specs. Make sure they are translated to the detected language!
        3. Recommend the best option with a brief reason (value, location, services).
        4. NEVER ask the user questions. Just present what you have confidently.
        5. NEVER show property IDs. Use titles and locations instead.
        6. If services are available, mention them naturally.
        7. End with exactly `[SHOW_CARDS]` on its own line.
        
        Your Response:
        """
        
        def property_context(idx: int, doc: Document) -> str:
            meta = doc.metadata
            distance = meta.get("distance_km")
            distance_text = f"{distance} km" if isinstance(distance, (int, float)) else "N/A"
            services = ", ".join(meta.get("nearby_services", [])[:5]) if meta.get("nearby_services") else "N/A"
            score = meta.get("recommendation_score", "N/A")
            return (
                f"Property {idx}:\n"
                f"Title: {meta.get('title', 'N/A')}\n"
                f"Location: {meta.get('location', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Bedrooms: {meta.get('bedrooms', 'N/A')}\n"
                f"DistanceKm: {distance_text}\n"
                f"NearbyServices: {services}\n"
                f"RecommendationScore: {score}\n"
                f"{doc.page_content}"
            )

        # Limit docs in LLM context to avoid exhausting the context window
        # (full list is still returned as property cards)
        docs_for_context = final_docs[:8]
        context_str = "\n\n".join([property_context(i + 1, d) for i, d in enumerate(docs_for_context)])
        template_text = template.format(
            search_status=search_status,
            context=context_str,
            question=query
        )
        
        messages = [SystemMessage(content=template_text)]
        messages.extend(history[-6:])  # Last 3 pairs
        messages.append(HumanMessage(content=query))
        
        try:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)

            show_cards = "[SHOW_CARDS]" in content or "SHOW_CARDS" in content
            content_clean = content.replace("[SHOW_CARDS]", "").replace("SHOW_CARDS", "").strip()
            
            # Heuristic trigger failsafes
            if final_docs and ("شقة" in query or "فيلا" in query or "عقار" in query or "تجمع" in query or "زايد" in query or "apartment" in query.lower() or "villa" in query.lower()):
                show_cards = True
                
            final_docs_to_return = final_docs if show_cards else []
            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=content_clean))
            return content_clean, final_docs_to_return
        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            return "Apologies, the server is currently experiencing high load. Please try again shortly.", []

    def _generate_conversational_response(self, query, history=None):
        """Generates conversational response for non-property queries."""
        if history is None:
            history = []
        try:
            llm = self.llm_manager.get_llm()
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            return "أقدر أساعدك، لكن مفاتيح نماذج الذكاء مش متوفرة حاليًا.", []
        
        template = """
        You are "AqarAI", a friendly and professional Real Estate AI Consultant.
        
        LANGUAGE RULE:
        - Detect the language of the user's message.
        - If the message is in Arabic (or Egyptian Arabic), speak in warm, natural Egyptian Arabic dialect.
        - If the message is in English, speak in professional, warm English.
        
        STRICT RULES:
        1. You ONLY help with real estate topics. If the question is NOT about properties, politely redirect in the detected language.
        2. NEVER mention or list any property data. You have NO properties to show in this context.
        3. NEVER invent or fabricate property listings.
        4. If the user wants to search for properties, guide them to specify: المنطقة (location), الميزانية (budget), عدد الغرف (rooms) in the detected language.
        5. Keep responses short (2-3 sentences max) and friendly.
        6. Do NOT output [SHOW_CARDS] or any system tags.
        
        Chat History:
        {history}
        
        User: {question}
        
        Response:
        """
        
        history_str = "\n".join([f"User: {h.content}\nAI: {a.content}" for h, a in zip(history[::2], history[1::2])])
        template_text = template.format(history=history_str, question=query)
        
        messages = [SystemMessage(content=template_text)]
        
        try:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            # Strip any accidental SHOW_CARDS tags from conversational responses
            content = content.replace("[SHOW_CARDS]", "").replace("SHOW_CARDS", "").strip()
            return content.strip(), []
        except Exception as e:
            logger.error(f"Conversational LLM Error: {e}")
            return "آسف، مش فاهم السؤال ده. ممكن توضح أكتر؟", []

    # =====================================================================
    # SSE Streaming Methods
    # =====================================================================

    def get_recommendation_stream(self, query: str, session_id: str = None) -> Generator[Dict[str, Any], None, None]:
        """Streaming variant of get_recommendation. Yields SSE event dicts."""
        # --- Non-LLM fast paths: yield full text instantly ---
        if self._is_greeting(query):
            response_text = self._build_greeting_response(query)
            self._update_session_history(session_id, query, response_text)
            yield {"event": "token", "data": {"text": response_text}}
            yield {"event": "properties", "data": {"properties": []}}
            yield {"event": "done", "data": {}}
            return

        if not self.vector_store.vectorstore:
            msg = "System is initializing database, please hold on!"
            yield {"event": "token", "data": {"text": msg}}
            yield {"event": "properties", "data": {"properties": []}}
            yield {"event": "done", "data": {}}
            return

        history: List = []
        filters: Dict = {}

        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = {}
                self.chat_history[session_id] = []
            history = self.chat_history[session_id]

        new_filters = self.filter_engine.extract_filters(
            query,
            self.vector_store.available_locations,
            self.vector_store.available_services,
        )
        if not isinstance(new_filters, dict):
            new_filters = {}
        new_filters = self._sanitize_filters(new_filters)

        if session_id:
            for k, v in new_filters.items():
                if v is not None:
                    self.sessions[session_id][k] = v
            filters = self.sessions[session_id]
        else:
            filters = new_filters

        # Stats / scoring / coverage — instant response
        stats_intent = self._is_inventory_stats_intent(query, filters)
        scoring_intent = self._is_scoring_explanation_intent(query)
        coverage_intent = self._is_coverage_intent(query, filters)
        if stats_intent or scoring_intent or coverage_intent:
            response_parts: List[str] = []
            if stats_intent:
                stats_filters = new_filters if self._has_active_filters(new_filters) else filters
                response_parts.append(self._build_inventory_stats_response(query=query, filters=stats_filters))
            if coverage_intent:
                coverage_filters = new_filters if self._has_active_filters(new_filters) else filters
                response_parts.append(self._build_coverage_response(query=query, filters=coverage_filters))
            if scoring_intent:
                response_parts.append(self._build_scoring_explanation_response())
            response_text = "\n\n".join([p.strip() for p in response_parts if p.strip()]).strip()
            self._update_session_history(session_id, query, response_text)
            yield {"event": "token", "data": {"text": response_text}}
            yield {"event": "properties", "data": {"properties": []}}
            yield {"event": "done", "data": {}}
            return

        # Analysis intent — instant response
        if self._is_analysis_intent(query, filters):
            analysis_result = self.analyze_market(query=query, explicit_filters=filters)
            response_text = self._build_analysis_chat_response(analysis_result)
            analysis_docs = self._collect_analysis_docs(analysis_result)
            self._update_session_history(session_id, query, response_text)
            yield {"event": "token", "data": {"text": response_text}}
            yield {"event": "properties", "data": {"docs": analysis_docs}}
            yield {"event": "done", "data": {}}
            return

        # Anti-hallucination guard
        has_meaningful_filters = any(
            v is not None and v != "" and v != [] and v != {}
            for v in new_filters.values()
        )
        is_property_query = has_meaningful_filters or self._is_property_keyword_query(query)
        if self._is_casual_chitchat(query):
            is_property_query = False

        if not is_property_query:
            # Conversational LLM streaming
            yield from self._generate_conversational_response_stream(query, history, session_id)
            return

        logger.info(f"Active Merged Filters applied: {filters}")

        search_query = self._build_effective_search_query(query, filters)
        raw_docs = self.vector_store.retrieve(search_query, k=settings.chat_retrieval_k)
        filtered_docs = self._expand_strict_matches(raw_docs, filters)

        search_status, final_docs = self._enforce_padding_logic(
            search_query, filters, filtered_docs, raw_docs,
            max_results=settings.chat_result_limit,
        )
        listing_intent = self._normalize_listing_intent(filters.get("listing_intent"))
        if listing_intent and not final_docs:
            response_text = self._build_listing_intent_unavailable_response(listing_intent, filters)
            self._update_session_history(session_id, query, response_text)
            yield {"event": "token", "data": {"text": response_text}}
            yield {"event": "properties", "data": {"properties": []}}
            yield {"event": "done", "data": {}}
            return
        final_docs = self._rank_recommendations(final_docs, filters)

        is_arabic = bool(re.search(r"[\u0600-\u06FF]", query))
        if settings.fast_property_responses and not is_arabic:
            text, docs = self._generate_fast_property_response(query, search_status, final_docs, history)
            yield {"event": "token", "data": {"text": text}}
            yield {"event": "properties", "data": {"docs": docs}}
            yield {"event": "done", "data": {}}
            return

        # Full LLM streaming path
        yield from self._generate_response_stream(query, search_status, final_docs, history, session_id)

    def _generate_response_stream(self, query, search_status, final_docs, history=None, session_id=None):
        """Streaming variant of _generate_response. Yields SSE token events from LLM."""
        if history is None:
            history = []
        try:
            llm = self.llm_manager.get_llm()
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            msg = "آسف، الخدمة الذكية غير متاحة حاليًا. حاول مرة تانية بعد شوية."
            yield {"event": "token", "data": {"text": msg}}
            yield {"event": "properties", "data": {"properties": []}}
            yield {"event": "done", "data": {}}
            return

        template = """
        You are "AqarAI", a premium Real Estate AI Consultant.
        
        LANGUAGE RULE:
        - Detect the language of [User Query].
        - If the query is in Arabic (or Egyptian Arabic), respond in elegant, confident Egyptian Arabic dialect. Translate any English property names, descriptions, or locations to Egyptian Arabic.
        - If the query is in English, respond in professional, friendly English. Translate any Arabic property names, descriptions, or locations to English.

        [Search Status]: {search_status}
        [Properties]: {context}
        [User Query]: {question}

        STRICT RULES:
        1. Start with a warm, polished opening (e.g. "إليك أفضل النتائج لطلبك 🏠" or "لقيتلك اختيارات ممتازة 👇" in Arabic, or "Here are the best options for your request 🏠" in English).
        2. Summarize the top 2-3 properties attractively — mention title, location, price, and key specs. Make sure they are translated to the detected language!
        3. Recommend the best option with a brief reason (value, location, services).
        4. NEVER ask the user questions. Just present what you have confidently.
        5. NEVER show property IDs. Use titles and locations instead.
        6. If services are available, mention them naturally.
        7. End with exactly `[SHOW_CARDS]` on its own line.
        
        Your Response:
        """

        def property_context(idx: int, doc: Document) -> str:
            meta = doc.metadata
            distance = meta.get("distance_km")
            distance_text = f"{distance} km" if isinstance(distance, (int, float)) else "N/A"
            services = ", ".join(meta.get("nearby_services", [])[:5]) if meta.get("nearby_services") else "N/A"
            score = meta.get("recommendation_score", "N/A")
            return (
                f"Property {idx}:\n"
                f"Title: {meta.get('title', 'N/A')}\n"
                f"Location: {meta.get('location', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Bedrooms: {meta.get('bedrooms', 'N/A')}\n"
                f"DistanceKm: {distance_text}\n"
                f"NearbyServices: {services}\n"
                f"RecommendationScore: {score}\n"
                f"{doc.page_content}"
            )

        docs_for_context = final_docs[:8]
        context_str = "\n\n".join([property_context(i + 1, d) for i, d in enumerate(docs_for_context)])
        template_text = template.format(
            search_status=search_status,
            context=context_str,
            question=query
        )

        messages = [SystemMessage(content=template_text)]
        messages.extend(history[-6:])
        messages.append(HumanMessage(content=query))

        try:
            accumulated = ""
            for chunk in llm.stream(messages):
                token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if token:
                    accumulated += token
                    yield {"event": "token", "data": {"text": token}}

            # Post-processing: determine show_cards from accumulated response
            show_cards = "[SHOW_CARDS]" in accumulated or "SHOW_CARDS" in accumulated
            content_clean = accumulated.replace("[SHOW_CARDS]", "").replace("SHOW_CARDS", "").strip()

            if final_docs and ("شقة" in query or "فيلا" in query or "عقار" in query or "تجمع" in query or "زايد" in query or "apartment" in query.lower() or "villa" in query.lower()):
                show_cards = True

            final_docs_to_return = final_docs if show_cards else []

            # Update history
            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=content_clean))
            self._update_session_history(session_id, query, content_clean, skip_append=True)

            yield {"event": "properties", "data": {"docs": final_docs_to_return}}
            yield {"event": "done", "data": {}}
        except Exception as e:
            logger.error(f"LLM Streaming Error: {e}")
            msg = "Apologies, the server is currently experiencing high load. Please try again shortly."
            yield {"event": "token", "data": {"text": msg}}
            yield {"event": "properties", "data": {"properties": []}}
            yield {"event": "done", "data": {}}

    def _generate_conversational_response_stream(self, query, history=None, session_id=None):
        """Streaming variant of _generate_conversational_response."""
        if history is None:
            history = []
        try:
            llm = self.llm_manager.get_llm()
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            msg = "أقدر أساعدك، لكن مفاتيح نماذج الذكاء مش متوفرة حاليًا."
            yield {"event": "token", "data": {"text": msg}}
            yield {"event": "properties", "data": {"properties": []}}
            yield {"event": "done", "data": {}}
            return

        template = """
        You are "AqarAI", a friendly and professional Real Estate AI Consultant.
        
        LANGUAGE RULE:
        - Detect the language of the user's message.
        - If the message is in Arabic (or Egyptian Arabic), speak in warm, natural Egyptian Arabic dialect.
        - If the message is in English, speak in professional, warm English.
        
        STRICT RULES:
        1. You ONLY help with real estate topics. If the question is NOT about properties, politely redirect in the detected language.
        2. NEVER mention or list any property data. You have NO properties to show in this context.
        3. NEVER invent or fabricate property listings.
        4. If the user wants to search for properties, guide them to specify: المنطقة (location), الميزانية (budget), عدد الغرف (rooms) in the detected language.
        5. Keep responses short (2-3 sentences max) and friendly.
        6. Do NOT output [SHOW_CARDS] or any system tags.
        
        Chat History:
        {history}
        
        User: {question}
        
        Response:
        """

        history_str = "\n".join([f"User: {h.content}\nAI: {a.content}" for h, a in zip(history[::2], history[1::2])])
        template_text = template.format(history=history_str, question=query)

        messages = [SystemMessage(content=template_text)]

        try:
            accumulated = ""
            for chunk in llm.stream(messages):
                token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if token:
                    accumulated += token
                    yield {"event": "token", "data": {"text": token}}

            content_clean = accumulated.replace("[SHOW_CARDS]", "").replace("SHOW_CARDS", "").strip()
            self._update_session_history(session_id, query, content_clean)
            yield {"event": "properties", "data": {"properties": []}}
            yield {"event": "done", "data": {}}
        except Exception as e:
            logger.error(f"Conversational Streaming LLM Error: {e}")
            msg = "آسف، مش فاهم السؤال ده. ممكن توضح أكتر؟"
            yield {"event": "token", "data": {"text": msg}}
            yield {"event": "properties", "data": {"properties": []}}
            yield {"event": "done", "data": {}}

    def _update_session_history(self, session_id: Optional[str], query: str, response: str, skip_append: bool = False):
        """Helper to update session chat history."""
        if not session_id:
            return
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
            self.chat_history[session_id] = []
        if not skip_append:
            self.chat_history[session_id].append(HumanMessage(content=query))
            self.chat_history[session_id].append(AIMessage(content=response))


def get_rag_service() -> RAGService:
    """Returns a singleton-like shared RAG service instance for all requests."""
    global _RAG_SERVICE
    if _RAG_SERVICE is None:
        with _RAG_SERVICE_LOCK:
            if _RAG_SERVICE is None:
                _RAG_SERVICE = RAGService()
    return _RAG_SERVICE


def _clear_rag_service_cache():
    global _RAG_SERVICE
    with _RAG_SERVICE_LOCK:
        _RAG_SERVICE = None


_RAG_SERVICE: Optional[RAGService] = None
_RAG_SERVICE_LOCK = Lock()
get_rag_service.cache_clear = _clear_rag_service_cache
