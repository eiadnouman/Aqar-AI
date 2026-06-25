import re
from typing import Any, Dict, Optional, Set
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.core.logging import logger
from app.core.config import settings


class SearchFilters(BaseModel):
    """Rigorous Schema enforced natively by the LLM via Tool Calling."""
    location: Optional[str] = Field(None, description="Match the user's location request. If it matches one of the 'Available Locations', use that exact name. If it is a location not in the list (e.g., 'بورسعيد' or 'Port Said'), extract/translate the location name and output it anyway so we know what specific location was requested.")
    min_price: Optional[float] = Field(None, description="Maximum price limitation extracted as a numeric value.")
    max_price: Optional[float] = Field(None, description="Maximum budget tolerance extracted dynamically.")
    min_bedrooms: Optional[int] = Field(None, description="Minimum logical bedrooms required.")
    max_bedrooms: Optional[int] = Field(None, description="Maximum conceptual bedrooms requested.")
    property_type: Optional[str] = Field(None, description="Categorization (e.g., apartment, villa, duplex).")
    listing_intent: Optional[str] = Field(
        None,
        description="Intent of listing type. Must be exactly 'rent' for rental requests and 'buy' for purchase requests.",
    )
    desired_services: Optional[list[str]] = Field(
        None,
        description="List of requested nearby services (e.g., security, schools, hospitals, transport, commercial_area, club_house, green_spaces).",
    )

class FilterEngine:
    """
    Engine responsible for extracting structured filters from natural language queries.
    Utilizes LLMs for dynamic extraction, falling back to Regex if LLM is unavailable or errors out.
    """
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    def extract_filters(
        self,
        query: str,
        available_locations: set,
        available_services: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Extracts filters using LLM, then backfills missing fields with regex heuristics."""
        regex_filters = self._extract_filters_regex(query)
        if settings.fast_filter_extraction and self._should_use_fast_regex(query, regex_filters):
            return regex_filters

        # We always attempt LLM Tool Calling extraction first for precision.
        try:
            llm = self.llm_manager.get_llm()
            llm_filters = self._extract_filters_llm(query, llm, available_locations, available_services or set())
            merged = dict(llm_filters or {})
            for key, value in regex_filters.items():
                if key not in merged or merged.get(key) in (None, "", [], {}):
                    merged[key] = value
            return merged
        except Exception as e:
            logger.warning(f"Native Function Calling failed: {e}. Falling back to Regex constraints.")
            return regex_filters

    def _extract_filters_llm(
        self,
        query: str,
        llm: Any,
        available_locations: set,
        available_services: Set[str],
    ) -> Dict[str, Any]:
        """Uses native Function Calling to construct the Pydantic schema automatically."""
        if llm is None or not hasattr(llm, "with_structured_output"):
            raise RuntimeError("Structured output is not supported by the active LLM backend.")

        locations_str = ", ".join(sorted(list(available_locations))) if available_locations else "No specific locations indexed."
        services_str = ", ".join(sorted(list(available_services))) if available_services else "No specific services indexed."

        template = """
        You are an advanced Real Estate Filter Analyzer.

        Available Locations DB Context:
        [{locations}]

        Available Services Tags Context:
        [{services}]

        Your job is to perfectly map the user's conversational query to the filter fields.
        Pay extreme attention to regional sub-districts and numeric names (like The 1st Settlement vs 5th Settlement).
        Detect listing intent precisely:
        - If user asks about renting/lease/إيجار -> listing_intent='rent'
        - If user asks about buying/sale/شراء/للبيع -> listing_intent='buy'
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("user", "{query}")
        ])
        
        structured_llm = llm.with_structured_output(SearchFilters)
        chain = prompt | structured_llm
        
        extracted: SearchFilters = chain.invoke(
            {"query": query, "locations": locations_str, "services": services_str}
        )
        result = extracted.model_dump(exclude_none=True)
        logger.info(f"Tool Calling Extracted Filters: {result}")
        return result

    def _extract_filters_regex(self, query: str) -> Dict[str, Any]:
        """Extracts structured filters from query text using strict Regular Expressions."""
        filters = {}
        
        # Edge cases for Arabic conversational bedroom counts
        if "غرفتين" in query or "2 bedrooms" in query or "2 rooms" in query:
            filters['min_bedrooms'] = 2
            filters['max_bedrooms'] = 2 
        
        # Standard bedroom matching
        bed_match = re.search(r'(\d+)\s*(?:ghorfa|oda|bedrooms?|rooms?|beds?|غرف|نوم)', query.lower())
        if bed_match:
            val = int(bed_match.group(1))
            filters['min_bedrooms'] = val
            filters['max_bedrooms'] = val + 1

        # Budget extraction (Millions vs standard numbers)
        million_match = re.search(r'(\d+)\s*(?:m|million|millions|مليون)', query.lower())
        if million_match:
            amount = float(million_match.group(1)) * 1_000_000
            filters['max_price'] = amount
        else:
            num_match = re.search(r'(\d{6,})', query.replace(',', ''))
            if num_match:
                filters['max_price'] = float(num_match.group(1))

        rent_tokens = ["rent", "rental", "lease", "للإيجار", "للايجار", "ايجار", "إيجار"]
        buy_tokens = ["buy", "sale", "for sale", "للبيع", "شراء", "تمليك", "بيع"]
        q_lower = query.lower()
        if any(token in q_lower for token in rent_tokens):
            filters["listing_intent"] = "rent"
        elif any(token in q_lower for token in buy_tokens):
            filters["listing_intent"] = "buy"

        property_type_map = {
            "apartment": ["شقة", "شقق", "apartment", "apartments", "flat", "flats"],
            "villa": ["فيلا", "فيلات", "villa", "villas"],
            "duplex": ["دوبلكس", "duplex"],
            "studio": ["ستوديو", "studio"],
            "chalet": ["شاليه", "شاليهات", "chalet", "chalets"],
        }
        for canonical, keywords in property_type_map.items():
            if any(token in q_lower for token in keywords):
                filters["property_type"] = canonical
                break

        # Basic Location Mapping for Regex Fallback
        location_map = {
            "القاهرة": "Cairo", "قاهرة": "Cairo",
            "اول": "The 1st Settlement", "تجمع اول": "The 1st Settlement",
            "خامس": "The 5th Settlement", "تجمع خامس": "The 5th Settlement",
            "تجمع": "New Cairo", "زايد": "Sheikh Zayed", "اكتوبر": "October",
            "ساحل": "North Coast", "العاصمة": "New Capital",
        }
        for ar, en in location_map.items():
            if ar in query:
                filters['location'] = en
                break

        service_map = {
            "schools": ["school", "schools", "مدرسة", "مدارس", "جامعة"],
            "hospitals": ["hospital", "clinic", "medical", "مستشفى", "عيادة"],
            "transport": ["metro", "transport", "مواصلات", "مترو", "طريق", "محور"],
            "commercial_area": ["mall", "shopping", "commercial", "مول", "تجاري", "خدمات"],
            "security": ["security", "حراسة", "أمن", "امن"],
            "green_spaces": ["garden", "park", "green", "حديقة", "حدائق", "مساحات خضراء"],
            "club_house": ["club", "clubhouse", "نادي", "كلوب هاوس"],
        }
        matched_services = []
        for canonical, keywords in service_map.items():
            if any(token in query.lower() for token in keywords):
                matched_services.append(canonical)
        if matched_services:
            filters["desired_services"] = matched_services
                
        return filters

    @staticmethod
    def _has_actionable_regex_filters(filters: Dict[str, Any]) -> bool:
        return any(
            filters.get(key) not in (None, "", [], {})
            for key in (
                "location",
                "min_price",
                "max_price",
                "min_bedrooms",
                "max_bedrooms",
                "property_type",
                "listing_intent",
                "desired_services",
            )
        )

    def _should_use_fast_regex(self, query: str, filters: Dict[str, Any]) -> bool:
        if not self._has_actionable_regex_filters(filters):
            return False
        if filters.get("location"):
            return True

        q_lower = str(query or "").lower()
        location_cues = (" في ", " فى ", " بمنطقة", " داخل ", " in ")
        if any(cue in f" {q_lower} " for cue in location_cues):
            return False

        return True
