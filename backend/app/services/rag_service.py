from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.logging import logger
from app.services.llm_manager import LLMManager
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
        new_filters = self.filter_engine.extract_filters(query, self.vector_store.available_locations)
        if not isinstance(new_filters, dict):
            logger.warning("Filter engine returned non-dict payload. Falling back to empty filters.")
            new_filters = {}

        # 3. Session Memory Merging
        if session_id:
            for k, v in new_filters.items():
                if v is not None:
                    self.sessions[session_id][k] = v
            filters = self.sessions[session_id]
        else:
            filters = new_filters

        # Check if it's a property-related query
        is_property_query = any(v is not None for v in new_filters.values())
        
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

        # 6. Language Model Generation
        response_text, generated_docs = self._generate_response(query, search_status, final_docs, history)
        
        return response_text, generated_docs

    def search_properties(self, query: Optional[str] = None, explicit_filters: Dict = None) -> Tuple[Dict, List[Document]]:
        """Headless search execution for direct API/UI querying without conversational fluff."""
        if not self.vector_store.vectorstore:
            return {}, []
            
        filters = dict(explicit_filters or {})
        if query:
            extracted = self.filter_engine.extract_filters(query, self.vector_store.available_locations)
            # Merge extracted into explicit but prioritize explicit UI inputs
            for k, v in extracted.items():
                if k not in filters or filters[k] is None:
                    filters[k] = v
        
        search_query = query or str(filters)
        raw_docs = self.vector_store.retrieve(search_query)
        filtered_docs = self._apply_exact_filters(raw_docs, filters)
        
        _, final_docs = self._enforce_padding_logic(search_query, filters, filtered_docs, raw_docs)
        return filters, final_docs

    def recommend_similar(self, description: str, k: int = 5) -> List[Document]:
        """Provides direct semantic equivalents to a given property description."""
        return self.vector_store.similarity_search(description, k=k)

    def _apply_exact_filters(self, docs: List[Document], filters: Dict) -> List[Document]:
        """Applies rigorous constraints to the raw vector results."""
        filtered = docs
        
        def safe_float(val):
            try: return float(val)
            except: return 0.0

        if filters.get('location'):
            target_loc = filters['location'].lower()
            filtered = [d for d in filtered if target_loc in d.metadata.get('location', '').lower()]

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

        return filtered

    def _enforce_padding_logic(self, query, filters, filtered_docs, raw_docs):
        """
        Calculates search status and conditionally pads results.
        Enforces strict geographical boundaries when backing off constraints.
        """
        search_status = "Excellent Match: We found properties that match exactly your request."
        
        if len(filtered_docs) == 0:
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
            
            # Absolute Fallback: Global semantic search
            if not filtered_docs:
                filtered_docs = raw_docs[:5]
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
        4. **Media Rendering**:
           - **CRITICAL**: You MUST append exactly `[SHOW_CARDS]` on a new standalone line at the very end of your response to trigger UI rendering.
        
        Your Response:
        """
        
        context_str = "\n\n".join([f"Property {i+1}:\n{d.page_content}" for i, d in enumerate(final_docs)])
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
