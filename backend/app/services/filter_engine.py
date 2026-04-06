import re
from typing import Any, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.core.logging import logger


class SearchFilters(BaseModel):
    """Rigorous Schema enforced natively by the LLM via Tool Calling."""
    location: Optional[str] = Field(None, description="Match the user's location request to the EXACT English name in the 'Available Locations' list. Pay extremely close attention to numbers (e.g., 'التجمع الاول' maps to 'The 1st Settlement', 'التجمع الخامس' maps to 'The 5th Settlement'). If the exact settlement number isn't specified, use 'New Cairo'. If not found, output null.")
    min_price: Optional[float] = Field(None, description="Maximum price limitation extracted as a numeric value.")
    max_price: Optional[float] = Field(None, description="Maximum budget tolerance extracted dynamically.")
    min_bedrooms: Optional[int] = Field(None, description="Minimum logical bedrooms required.")
    max_bedrooms: Optional[int] = Field(None, description="Maximum conceptual bedrooms requested.")
    property_type: Optional[str] = Field(None, description="Categorization (e.g., apartment, villa, duplex).")

class FilterEngine:
    """
    Engine responsible for extracting structured filters from natural language queries.
    Utilizes LLMs for dynamic extraction, falling back to Regex if LLM is unavailable or errors out.
    """
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    def extract_filters(self, query: str, available_locations: set) -> Dict[str, Any]:
        """Extracts filters using LLM (if capable) or Regex fallback."""
        # We always attempt LLM Tool Calling extraction first for unparalleled precision
        try:
            llm = self.llm_manager.get_llm()
            return self._extract_filters_llm(query, llm, available_locations)
        except Exception as e:
            logger.warning(f"Native Function Calling failed: {e}. Falling back to Regex constraints.")
            return self._extract_filters_regex(query)

    def _extract_filters_llm(self, query: str, llm: Any, available_locations: set) -> Dict[str, Any]:
        """Uses native Function Calling to construct the Pydantic schema automatically."""
        if llm is None or not hasattr(llm, "with_structured_output"):
            raise RuntimeError("Structured output is not supported by the active LLM backend.")

        locations_str = ", ".join(sorted(list(available_locations))) if available_locations else "No specific locations indexed."

        template = """
        You are an advanced Real Estate Filter Analyzer.

        Available Locations DB Context:
        [{locations}]

        Your job is to perfectly map the user's conversational query to the filter fields.
        Pay extreme attention to regional sub-districts and numeric names (like The 1st Settlement vs 5th Settlement).
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("user", "{query}")
        ])
        
        structured_llm = llm.with_structured_output(SearchFilters)
        chain = prompt | structured_llm
        
        extracted: SearchFilters = chain.invoke({"query": query, "locations": locations_str})
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

        # Basic Location Mapping for Regex Fallback
        location_map = {
            "اول": "The 1st Settlement", "تجمع اول": "The 1st Settlement",
            "خامس": "The 5th Settlement", "تجمع خامس": "The 5th Settlement",
            "تجمع": "New Cairo", "زايد": "Sheikh Zayed", "اكتوبر": "October",
            "ساحل": "North Coast", "العاصمة": "New Capital",
        }
        for ar, en in location_map.items():
            if ar in query:
                filters['location'] = en
                break
                
        return filters
