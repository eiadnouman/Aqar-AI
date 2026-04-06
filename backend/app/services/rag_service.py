from functools import lru_cache
import math
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.logging import logger
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

    def get_recommendation(self, query: str, session_id: str = None) -> Tuple[str, List[Document]]:
        """Handles conversational interaction, processing filters and returning contextual replies."""
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

        # Check if it's a property-related query
        is_property_query = any(v is not None for v in new_filters.values()) or self._is_property_keyword_query(query)
        
        if not is_property_query:
            # Conversational response for non-property queries
            response_text, _ = self._generate_conversational_response(query, history)
            if session_id:
                self.chat_history[session_id].append(HumanMessage(content=query))
                self.chat_history[session_id].append(AIMessage(content=response_text))
            return response_text, []
            
        logger.info(f"Active Merged Filters applied: {filters}")

        # 4. Vector Retrieval & Hard Constraints Application
        raw_docs = self.vector_store.retrieve(query)
        filtered_docs = self._apply_exact_filters(raw_docs, filters)
        
        # 5. Fallback Routing & Padding
        search_status, final_docs = self._enforce_padding_logic(query, filters, filtered_docs, raw_docs)
        listing_intent = self._normalize_listing_intent(filters.get("listing_intent"))
        if listing_intent and not final_docs:
            response_text = self._build_listing_intent_unavailable_response(listing_intent, filters)
            if session_id:
                self.chat_history[session_id].append(HumanMessage(content=query))
                self.chat_history[session_id].append(AIMessage(content=response_text))
            return response_text, []
        final_docs = self._rank_recommendations(final_docs, filters)

        # 6. Language Model Generation
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
        
        search_query = query or str(filters)
        raw_docs = self.vector_store.retrieve(search_query)
        filtered_docs = self._apply_exact_filters(raw_docs, filters)
        
        _, final_docs = self._enforce_padding_logic(search_query, filters, filtered_docs, raw_docs)
        final_docs = self._rank_recommendations(final_docs, filters)
        return filters, final_docs

    def recommend_similar(self, description: str, k: int = 5) -> List[Document]:
        """Provides direct semantic equivalents to a given property description."""
        return self.vector_store.similarity_search(description, k=k)

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

        search_query = query or str(filters) or "real estate market in Egypt"
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

        tokens = [
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
        ]
        return any(token in raw_query for token in tokens)

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

    def _sanitize_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(filters, dict):
            return {}

        cleaned = dict(filters)
        cleaned["location"] = self._normalize_location_filter(cleaned.get("location"))
        cleaned["listing_intent"] = self._normalize_listing_intent(cleaned.get("listing_intent"))
        services = cleaned.get("desired_services")
        if isinstance(services, str):
            services = [s.strip() for s in services.split(",") if s.strip()]
        if isinstance(services, list):
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
            if idx < max_live_service_docs and lat is not None and lon is not None:
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
        requested_center = self._get_requested_location_center(requested_location)
        if requested_center is not None:
            return requested_center

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
        if not pool:
            return None

        lat_center = sum(p[0] for p in pool) / len(pool)
        lon_center = sum(p[1] for p in pool) / len(pool)
        return lat_center, lon_center

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

    def _enforce_padding_logic(self, query, filters, filtered_docs, raw_docs):
        """
        Calculates search status and conditionally pads results.
        Enforces strict geographical boundaries when backing off constraints.
        """
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
            if requested_loc and not any(requested_loc.lower() in d.metadata.get('location', '').lower() for d in raw_docs):
                 search_status = f"Alert: You requested ({requested_loc}) which is currently unavailable. Ask the user to review these alternatives in different locations."
            else:
                 search_status = "Alert: The exact specifications (rooms/price) are unavailable. Present these closest semantic alternatives instead."
                 
            # Priority Fallback: Attempt to hold the location boundary if possible
            requested_loc = filters.get('location')
            target_padding_loc = requested_loc
            
            if target_padding_loc:
                filtered_docs = [d for d in raw_docs if target_padding_loc.lower() in d.metadata.get('location', '').lower()]
            elif requested_intent:
                filtered_docs = [d for d in raw_docs if self._doc_listing_intent(d) == requested_intent]
            
            # Absolute Fallback: Global semantic search
            if not filtered_docs:
                filtered_docs = [
                    d for d in raw_docs
                    if (not requested_intent) or self._doc_listing_intent(d) == requested_intent
                ][:5]
            else:
                filtered_docs = filtered_docs[:5]
                
        elif len(filtered_docs) < 5:
             search_status = "Partial Match: We found a few matching properties. Display these along with semantic alternatives in the same region."
             existing_urls = {d.metadata.get('url') for d in filtered_docs}
             
             requested_loc = filters.get('location')
             
             padding_pool = []
             # Pad exclusively inside the requested geo-fence
             if requested_loc:
                 for d in raw_docs:
                     d_loc = d.metadata.get('location', '').lower()
                     if requested_loc.lower() in d_loc:
                         padding_pool.append(d)
             else:
                 # If no specific location was requested, pad from anywhere
                 padding_pool = raw_docs
                 if requested_intent:
                     padding_pool = [d for d in padding_pool if self._doc_listing_intent(d) == requested_intent]
             
             for doc in padding_pool:
                 if len(filtered_docs) >= 5: break
                 if doc.metadata.get('url') not in existing_urls:
                     filtered_docs.append(doc)
                     existing_urls.add(doc.metadata.get('url'))
        else:
            filtered_docs = filtered_docs[:5]
            
        return search_status, filtered_docs

    def _generate_response(self, query, search_status, final_docs, history=None):
        """Generates conversational text with deep Chat History Memory."""
        if history is None:
            history = []
        try:
            llm = self.llm_manager.get_llm()
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            return "آسف، الخدمة الذكية غير متاحة حاليًا. حاول مرة تانية بعد شوية.", []
        
        # Translated to English while keeping the persona and output style consistent
        template = """
        You are "AqarAI", a highly intelligent, direct, and practical Real Estate Consultant and Broker.
        You act like a trusted adviser who knows the market, negotiates firmly, and presents the best deals clearly.
        Your mission is to provide rapid, direct answers with expert broker insight and no unnecessary talk.

        [Actual Database Search Status Context]: 
        {search_status}

        [Available Properties Context for Display]:
        {context}

        User Request: 
        {question}

        System Instructions (Follow these strictly):
        1. **Context Adherence**: Phrase your opening response exactly matching the [Search Status Context].
           - If "Excellent Match" or "Partial Match": Confidently present the properties immediately ("Based on your request, here are the best available properties:").
           - If "Alert": Apologize politely, state the reason defined in the alert, and present the alternatives.
        2. **Do Not Interrogate**: NEVER ask the user questions (e.g., "What is your budget?" or "How many rooms?"). Just display the inventory.
        3. **Tone**: Speak in an elegant, confident, and direct Egyptian Arabic dialect. Summarize properties attractively without listing numerical IDs.
        4. **Best Option Guidance**: If one property has clearly better score/value/location proximity, explicitly recommend it as the best option and explain why briefly.
        5. **Nearby Services Awareness**: Mention key nearby services when available (e.g., schools, hospitals, mall, transport).
        6. **Media Rendering**:
           - **CRITICAL**: You MUST append exactly `[SHOW_CARDS]` on a new standalone line at the very end of your response to trigger UI rendering.
        
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

        context_str = "\n\n".join([property_context(i + 1, d) for i, d in enumerate(final_docs)])
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
            if final_docs and ("شقة" in query or "فيلا" in query or "عقار" in query or "تجمع" in query or "زايد" in query):
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
        You are "AqarAI", a skilled Real Estate Consultant and Broker.
        You can chat about real estate topics, answer general questions, and act as an adviser who knows the market well.
        Keep responses in Egyptian Arabic dialect, be helpful and trustworthy.
        If the user asks about properties, guide them to specify location, budget, and preferences clearly.
        Do not show property cards for conversational responses.
        
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
            return content.strip(), []
        except Exception as e:
            logger.error(f"Conversational LLM Error: {e}")
            return "آسف، مش فاهم السؤال ده. ممكن توضح أكتر؟", []


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Returns a singleton-like shared RAG service instance for all requests."""
    return RAGService()
