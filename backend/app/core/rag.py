import os
import csv
import re
import logging
from typing import List, Dict, Optional, Any, Tuple

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing Groq, handle missing dependency gracefully
try:
    from langchain_groq import ChatGroq
except ImportError:
    logger.warning("langchain_groq not found. Groq models will be unavailable.")
    ChatGroq = None

load_dotenv()

class RealEstateRAG:
    """
    Core RAG Engine for Real Estate Recommendations.
    Handles data indexing, retrieval, and LLM generation.
    """

    def __init__(self, index_path: str = "data/faiss_index_cloud"):
        # Use an robust path resolution strategy for production
        # In Railway, the app runs from the root directory (/app)
        base_dir = os.path.abspath(os.getcwd())
        default_path = os.path.join(base_dir, "data", "faiss_index_cloud")
        self.index_path = os.getenv("FAISS_INDEX_PATH", default_path)
        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        if not self.hf_token:
            logger.error("HUGGINGFACEHUB_API_TOKEN is missing in .env")
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is required.")

        # 1. Setup Embeddings
        logger.info("Initializing Embeddings Engine (Local)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # 2. Setup LLM
        self.llm = self._initialize_llm()
        
        self.available_locations = set()
        self.sessions = {} # Session Filter Memory
        self.vectorstore = None

        # 3. Load Vector Store
        self._load_index()

    # Updated generate_recommendation to use memory
    def generate_recommendation(self, query: str, session_id: str = None) -> Tuple[str, List[Document]]:
        """
        Generates recommendation. Uses session_id to maintain filter state.
        """
        if not self.vectorstore:
            return "النظام بيجهز الداتا والدليل العقاري... ثواني وراجعلك!", []

        # 1. Extract Filters from Current Query
        new_filters = self._extract_filters(query)
        logger.info(f"New Filters: {new_filters}")

        # 2. Merge with Session Filters
        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = {}
            
            # Update session with new NON-NULL filters
            for k, v in new_filters.items():
                if v is not None:
                    self.sessions[session_id][k] = v
            
            final_filters = self.sessions[session_id]
            logger.info(f"Merged Filters (Session {session_id}): {final_filters}")
        else:
            final_filters = new_filters

        # 3. Retrieve
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 100})
        raw_docs = retriever.invoke(query)
        
        # 4. Apply Filters
        filtered_docs = self._apply_filters(raw_docs, final_filters)
        
        # Handle "No Result" with Fallback
        msg_prefix = ""
        if not filtered_docs:
            logger.warning("No properties matched strict filters. Showing closest vector matches.")
            # If we have location in filters, force fallback there
            loc_fallback = final_filters.get('location', '')
            if loc_fallback:
                 # Try to filter raw_docs by location only as fallback
                 filtered_docs = [d for d in raw_docs if loc_fallback.lower() in d.metadata.get('location', '').lower()]
            
            if not filtered_docs:
                 filtered_docs = raw_docs[:5] # Last resort
            else:
                 filtered_docs = filtered_docs[:5]

            msg_prefix = "ملقتش عقارات بالمواصفات دي بالظبط (ممكن بسبب السعر)، بس دي أقرب حاجات ليك:\n\n"
        else:
            filtered_docs = filtered_docs[:5]
        
        final_docs = filtered_docs

        # 5. Construct Prompt (Pass effective filters context to LLM)
        filter_context = f"Active Filters: {json.dumps(final_filters, ensure_ascii=False)}"
        
        template = """
        انت "AqarAI"، مستشار عقاري خبير.
        
        {filter_context}

        البيانات المتاحة (العقارات):
        {context}

        رسالة العميل: 
        {question}

        تعليمات الرد (مهمة جداً):
        1. **شخصيتك**: عامية مصرية شيك (Formal Friendly).
        2. **مهمتك**: لو العميل طلب حاجة محددة (زي "شقة في التجمع")، وانت لقيت عقارات مناسبة في "البيانات المتاحة"، يبقى اعرضها عليه فوراً وقوله "لقيتلك شقق ممتازة".
        3. **إظهار العقارات**: 
             - لو العقارات المتاحة مناسبة لطلب العميل: اكتب `[SHOW_CARDS]`.
             - لو العقارات مش مناسبة أو العميل لسه بيسأل سؤال عام: لا تكتبها.
        
        ردك:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": lambda x: self._format_docs_to_string(final_docs), 
             "question": RunnablePassthrough(),
             "filter_context": lambda x: filter_context}
            | prompt
            | self.llm
        )

        # Execute
        try:
            response = chain.invoke(query)
            content = response.content if hasattr(response, 'content') else str(response)
            
            if msg_prefix:
                content = msg_prefix + content

            if final_docs and "[SHOW_CARDS]" not in content:
                content += "\n\n[SHOW_CARDS]"
                
            return content, final_docs
        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            return "معلش حصل مشكلة بسيطة. ممكن نحاول تاني؟", []

    def _initialize_llm(self):
        """Initializes the LLM, prioritizing Groq Llama 3 with fallback to Flan-T5."""
        if self.groq_api_key and ChatGroq:
            try:
                logger.info("Connecting to Groq (Llama 3.3)...")
                llm = ChatGroq(
                    temperature=0.7,
                    model_name="llama-3.3-70b-versatile",
                    groq_api_key=self.groq_api_key.strip()
                )
                logger.info("Connected to Groq successfully.")
                return llm
            except Exception as e:
                logger.error(f"Groq connection failed: {e}. Falling back to HuggingFace.")
        
        logger.info("Using HuggingFace Fallback Model.")
        return self._init_hf_fallback()

    def _init_hf_fallback(self):
        """Fallback to Mixtral via HuggingFace API."""
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        logger.info(f"Initializing Fallback Model: {repo_id}")
        return HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.1,
            huggingfacehub_api_token=self.hf_token
        )

    def _load_index(self):
        """Loads the FAISS index from disk if it exists and extracts unique locations."""
        if os.path.exists(self.index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector Database loaded successfully.")
                
                # Dynamic Location Extraction
                try:
                    # Access the in-memory docstore to find all unique locations
                    if hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
                        all_docs = self.vectorstore.docstore._dict.values()
                        for doc in all_docs:
                            loc = doc.metadata.get('location')
                            if loc:
                                self.available_locations.add(loc)
                        logger.info(f"Dynamic Locations Found: {list(self.available_locations)[:5]}... (Total: {len(self.available_locations)})")
                except Exception as e:
                    logger.warning(f"Could not extract dynamic locations: {e}")
                    
            except Exception as e:
                logger.error(f"Failed to load vector index: {e}")
        else:
            logger.error(f"FAISS Index Path NOT FOUND at: {os.path.abspath(self.index_path)}")

    def load_and_index_data(self, csv_file: str, batch_size: int = 1000):
        """
        Reads CSV, processes properties, and builds/updates the vector index.
        
        Args:
            csv_file (str): Path to the source CSV file.
            batch_size (int): Number of records to process per batch.
        """
        logger.info(f"Starting data indexing from {csv_file}...")
        documents = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                
                for i, row in enumerate(reader):
                    # Pass the zero-based index for image mapping (property_{i+1})
                    content = self._format_doc_content(row)
                    meta = self._extract_metadata(row, i)
                    
                    documents.append(Document(page_content=content, metadata=meta))
                    count += 1
                    
                    if len(documents) >= batch_size:
                        self._process_batch(documents)
                        documents = []
                        logger.info(f"Indexed {count} properties...")
                
                if documents:
                    self._process_batch(documents)
                    
            logger.info(f"Indexing complete. Total properties: {count}")
            
        except Exception as e:
            logger.error(f"Data indexing failed: {e}")

    def _format_doc_content(self, row: Dict) -> str:
        """Formats CSV row into a readable text document."""
        return (
            f"Type: {row.get('property_type', 'Unknown')}\n"
            f"Title: {row.get('title', 'No Title')}\n"
            f"Location: {row.get('location_full_name', 'Unknown')}\n"
            f"Price: {row.get('price_value', '0')} {row.get('price_currency', 'EGP')}\n"
            f"Size: {row.get('size_value', '0')} {row.get('size_unit', 'sqm')}\n"
            f"Bedrooms: {row.get('bedrooms', '0')} | Bathrooms: {row.get('bathrooms', '0')}\n"
            f"Description: {row.get('description', '')}"
        )

    def _extract_metadata(self, row: Dict, row_index: int) -> Dict:
        """Extracts and cleans metadata for the document."""
        def clean_num(value):
            try:
                if not value or value == 'null': return 0
                return float(str(value).replace(',', ''))
            except ValueError:
                return 0

        # Image Mapping Logic: property_{row_index + 1}/1.jpg
        # We assume the images are in 'data/images/images/' relative to project root
        property_id = row_index + 1
        image_path = f"images/images/property_{property_id}/1.jpg"
        # Generate a local URL routing to the frontend's property view
        local_url = f"/property/{property_id}"

        return {
            "price": clean_num(row.get('price_value')),
            "size": clean_num(row.get('size_value')),
            "location": row.get('location_full_name', ''),
            "type": row.get('property_type', ''),
            "url": row.get('url', '#'),
            "bedrooms": clean_num(row.get('bedrooms')),
            "bathrooms": clean_num(row.get('bathrooms')),
            "lat": clean_num(row.get('lat')),
            "lon": clean_num(row.get('lon')),
            "image": image_path,
            "title": row.get('title', '')
        }

    def _process_batch(self, docs: List[Document]):
        """Helper to add documents to vectorstore and save."""
        if not self.vectorstore:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)
        self.vectorstore.save_local(self.index_path)

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Performs a similarity search."""
        if not self.vectorstore:
            logger.warning("Search attempted but VectorStore is not loaded.")
            return []
        return self.vectorstore.similarity_search(query, k=k)

    def generate_recommendation(self, query: str, session_id: str = None) -> Tuple[str, List[Document]]:
        """
        Generates a persona-driven recommendation based on the query.
        Uses session_id to persist filters.
        """
        if not self.vectorstore:
            return "النظام بيجهز الداتا والدليل العقاري... ثواني وراجعلك!", []

        # 1. Extract Filters
        new_filters = self._extract_filters(query)
        logger.info(f"New Filters: {new_filters}")

        # 2. Merge with Session Filters
        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = {}
            for k, v in new_filters.items():
                if v is not None:
                    self.sessions[session_id][k] = v
            filters = self.sessions[session_id]
            logger.info(f"Merged Filters (Session {session_id}): {filters}")
        else:
            filters = new_filters

        # 2. Retrieve (Fetch more to allow for filtering)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 100})
        raw_docs = retriever.invoke(query)
        
        # 3. Apply Filters
        filtered_docs = self._apply_filters(raw_docs, filters)
        
        search_status = "تطابق ممتاز: وجدنا عقارات مطابقة لطلب العميل بدقة."
        
        # Determine if we need to use fallback
        if len(filtered_docs) == 0:
            logger.warning("No properties matched strict filters. Showing closest vector matches/location defaults.")
            
            # Determine reason for mismatch
            requested_loc = filters.get('location')
            if requested_loc and not any(requested_loc.lower() in d.metadata.get('location', '').lower() for d in raw_docs):
                 search_status = f"تنبيه: العميل طلب مكان ({requested_loc}) غير متوفر حالياً. اعرض البدائل التالية في أماكن أخرى ووضح ذلك بلطف."
            else:
                 search_status = "تنبيه: لا يوجد توفر بنفس المواصفات المطلوبة تماماً (سعر/غرف/نوع). اعرض البدائل الأقرب التالية."
                 
            # Fallback 1: Just Location
            filtered_docs = self._filter_by_location(query, raw_docs)
            if not filtered_docs:
                # Fallback 2: Just semantic vectors
                filtered_docs = raw_docs[:5]
            else:
                filtered_docs = filtered_docs[:5]
                
        elif len(filtered_docs) < 5:
             # We found SOME exact matches, but let's pad it with semantic matches just in case
             search_status = "تطابق جزئي: وجدنا عدد قليل من العقارات المطابقة. سيتم عرضها مع بعض البدائل الأخرى."
             existing_urls = {d.metadata.get('url') for d in filtered_docs}
             for doc in raw_docs:
                 if len(filtered_docs) >= 5: break
                 if doc.metadata.get('url') not in existing_urls:
                     filtered_docs.append(doc)
                     existing_urls.add(doc.metadata.get('url'))
        else:
            filtered_docs = filtered_docs[:5]

        final_docs = filtered_docs

        # 4. Construct Prompt
        template = """
        انت "AqarAI"، مستشار عقاري ذكي وصريح جداً وعملي.
        مهمتك الرد المباشر السريع بدون إطالة أو أسئلة غير ضرورية.

        [حالة البحث الفعلي في قاعدة البيانات]: 
        {search_status}

        [العقارات المتاحة لتعرضها على العميل]:
        {context}

        رسالة العميل: 
        {question}

        تعليمات الرد (أهم جزء في النظام - التزم بها حرفياً):
        1. **قراءة حالة البحث بدقة**: يجب أن تبني ردك الافتتاحي بشكل حرفي على [حالة البحث الفعلي].
           - لو الحالة "تطابق ممتاز" أو "تطابق جزئي": اعرض العقارات بفخر وثقة فوراً (مثال: "بناءً على طلبك، دي أفضل عقارات متاحة حالياً:").
           - لو الحالة "تنبيه" تفيد بعدم وجود العقار: اعتذر بلطف شديد واذكر السبب الموجود في التنبيه، ثم اعرض البدائل.
        2. **لا تسأل أبداً**: ممنوع توجيه أي أسئلة للعميل (مثل "محتاج كام غرفة؟" أو "ميزانيتك كام؟"). فقط اعرض ما لديك واختم كلامك.
        3. **أسلوب الرد**: عامية مصرية شيك واثقة، ومباشرة. لخص العقارات بشكل جذاب وسريع ولا تذكر أرقام العقارات التسلسلية.
        4. **إظهار العقارات**:
           - **يجب** وضع كلمة `[SHOW_CARDS]` في سطر مستقل في نهاية ردك تماماً لكي يتمكن النظام من عرض صور العقارات للعميل. لا تنسى هذه الكلمة أبداً.
        
        ردك:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {
                "context": lambda x: self._format_docs_to_string(final_docs), 
                "question": RunnablePassthrough(),
                "search_status": lambda x: search_status
            }
            | prompt
            | self.llm
        )

        # 5. Generate & Handle Errors
        try:
            response = chain.invoke(query)
            # Handle different return types from LangChain integrations
            content = response.content if hasattr(response, 'content') else str(response)

            # Clean output
            show_cards = "[SHOW_CARDS]" in content or "SHOW_CARDS" in content
            content_clean = content.replace("[SHOW_CARDS]", "").replace("SHOW_CARDS", "").strip()
            
            # If the LLM forgot to add the tag, but we have strict exact filters mapped to final_docs, logic can enforce it:
            if final_docs and ("شقة" in query or "فيلا" in query or "عقار" in query or "تجمع" in query or "زايد" in query):
                show_cards = True
                
            # ONLY return docs if the LLM decided to show them
            final_docs_to_return = final_docs if show_cards else []
                
            return content_clean, final_docs_to_return
        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            if "429" in str(e) or "rate_limit_exceeded" in str(e).lower():
                logger.warning("Groq Rate Limit Exceeded. Attempting to fallback to HuggingFace...")
                try:
                    fallback_llm = self._init_hf_fallback()
                    fallback_chain = (
                        {
                            "context": lambda x: self._format_docs_to_string(final_docs), 
                            "question": RunnablePassthrough(),
                            "search_status": lambda x: search_status
                        }
                        | prompt
                        | fallback_llm
                    )
                    # For HuggingFaceEndpoint, invoke usually returns a string directly
                    response = fallback_chain.invoke(query)
                    content = response if isinstance(response, str) else (response.content if hasattr(response, 'content') else str(response))
                    show_cards = "[SHOW_CARDS]" in content or "SHOW_CARDS" in content
                    content_clean = content.replace("[SHOW_CARDS]", "").replace("SHOW_CARDS", "").strip()
                    if final_docs and ("شقة" in query or "فيلا" in query or "عقار" in query or "تجمع" in query or "زايد" in query):
                        show_cards = True
                    final_docs_to_return = final_docs if show_cards else []
                    return content_clean, final_docs_to_return
                except Exception as fallback_e:
                    logger.error(f"Fallback LLM Error: {fallback_e}")
                    return "معلش السيرفر عليه ضغط حالياً. ممكن تحاول كمان شوية؟", []
            return "معلش حصل مشكلة بسيطة في السيستم. ممكن تكرر طلبك؟", []

    def search_properties(self, query: str) -> Tuple[Dict, List[Document]]:
        """
        API Endpoint method for retrieving properties directly (without generating a chat response).
        Extracts filters from query and returns raw matching documents.
        """
        if not self.vectorstore:
            return {}, []
            
        # 1. Extract Filters
        filters = self._extract_filters(query)
        logger.info(f"Direct Search Filters: {filters}")
        
        # 2. Retrieve
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 100})
        raw_docs = retriever.invoke(query)
        
        # 3. Apply Filters
        filtered_docs = self._apply_filters(raw_docs, filters)
        
        if not filtered_docs:
            logger.warning("Search API: No exact matches. Showing closest vectors.")
            # Basic fallback
            loc_fallback = filters.get('location', '')
            if loc_fallback:
                filtered_docs = [d for d in raw_docs if loc_fallback.lower() in d.metadata.get('location', '').lower()]
            if not filtered_docs:
                filtered_docs = raw_docs[:10]
            else:
                filtered_docs = filtered_docs[:10]
        else:
            filtered_docs = filtered_docs[:10]
            
        return filters, filtered_docs

    def get_similar_properties(self, target_text: str, k: int = 5) -> List[Document]:
        """
        Recommendation Engine: Finds properties similar to the provided description/text.
        """
        if not self.vectorstore:
            return []
            
        logger.info("Recommendation Engine: Finding similar properties.")
        # Perform pure semantic search based on the provided text
        return self.vectorstore.similarity_search(target_text, k=k)

    def _extract_filters(self, query: str) -> Dict:
        """
        Extracts filters using LLM (Groq) if available, otherwise falls back to Regex.
        """
        # Try LLM Extraction first if Groq is active
        if self.groq_api_key and ChatGroq and isinstance(self.llm, ChatGroq):
            try:
                return self._extract_filters_llm(query)
            except Exception as e:
                logger.warning(f"LLM Filter Extraction failed: {e}. Falling back to Regex.")

        return self._extract_filters_regex(query)

    def _extract_filters_llm(self, query: str) -> Dict:
        """Uses LLM to parse natural language into structured filters."""
        parser = JsonOutputParser()
        
        # Convert set to sorted list for consistent prompts
        locations_str = ", ".join(sorted(list(self.available_locations))) if self.available_locations else "No specific locations indexed."

        template = """
        Extract search filters from the user query into a JSON object.
        
        Available Locations in Database:
        [{locations}]
        
        Keys:
        - location: Match the user's request (Arabic or English) to the CLOSEST EXACT NAME from the "Available Locations" list above. If no match found, output null.
        - min_price (number)
        - max_price (number)
        - min_bedrooms (number)
        - max_bedrooms (number)
        - min_bathrooms (number)
        - property_type (string)

        If information is missing, do NOT include the key.
        
        User Query: {query}
        
        JSON Output:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | parser
        
        filters = chain.invoke({"query": query, "locations": locations_str})
        logger.info(f"LLM Extracted Filters: {filters}")
        return filters

    def _extract_filters_regex(self, query: str) -> Dict:
        """Extracts structured filters from query text using Regex (Legacy/Fallback)."""
        filters = {}
        
        # Bedrooms (e.g., "3 bedrooms", "3 rooms", "3 غرف", "غرفتين")
        # Handle "غرفتين" -> 2
        if "غرفتين" in query or "2 bedrooms" in query or "2 rooms" in query:
            filters['min_bedrooms'] = 2
            filters['max_bedrooms'] = 2 
        
        bed_match = re.search(r'(\d+)\s*(?:ghorfa|oda|bedrooms?|rooms?|beds?|غرف|نوم)', query.lower())
        if bed_match:
            val = int(bed_match.group(1))
            filters['min_bedrooms'] = val
            filters['max_bedrooms'] = val + 1

        # Budget / Price
        million_match = re.search(r'(\d+)\s*(?:m|million|millions|مليون)', query.lower())
        if million_match:
            amount = float(million_match.group(1)) * 1_000_000
            filters['max_price'] = amount
        else:
            num_match = re.search(r'(\d{6,})', query.replace(',', ''))
            if num_match:
                filters['max_price'] = float(num_match.group(1))

        # Location
        filters['location'] = self._get_location_from_query(query)
        
        return filters

    def _apply_filters(self, docs: List[Document], filters: Dict) -> List[Document]:
        """Filters documents based on extracted constraints."""
        filtered = docs
        
        def safe_float(val):
            try: return float(val)
            except: return 0.0

        # 1. Location Filter
        if filters.get('location'):
            target_loc = filters['location'].lower()
            filtered = [d for d in filtered if target_loc in d.metadata.get('location', '').lower()]

        # 2. Bedroom Filter
        if 'min_bedrooms' in filters:
            min_b = safe_float(filters['min_bedrooms'])
            filtered = [d for d in filtered if safe_float(d.metadata.get('bedrooms', 0)) >= min_b]
            
            if 'max_bedrooms' in filters:
                 max_b = safe_float(filters['max_bedrooms'])
                 filtered = [d for d in filtered if safe_float(d.metadata.get('bedrooms', 0)) <= max_b]
        
        # 3. Price Filter (Max Budget)
        if 'max_price' in filters:
            max_p = safe_float(filters['max_price'])
            # Filter out things way above budget (allow 10% wiggle room?)
            filtered = [d for d in filtered if safe_float(d.metadata.get('price', 0)) <= max_p * 1.1]

        # 4. Property Type
        if filters.get('property_type'):
            ptype = filters['property_type'].lower()
            # fuzzy match?
            filtered = [d for d in filtered if ptype in d.metadata.get('type', '').lower() or ptype in d.metadata.get('title', '').lower()]

        return filtered

    def _get_location_from_query(self, query: str) -> Optional[str]:
        """Helper to extract location string."""
        location_map = {
            "تجمع": "New Cairo", "خامس": "New Cairo", "new cairo": "New Cairo",
            "زايد": "Sheikh Zayed", "اكتوبر": "October",
            "سخنة": "Ain Sokhna", "ساحل": "North Coast",
            "عاصمة": "New Capital", "مستقبل": "Mostakbal",
            "شروق": "Shorouk", "رحاب": "Rehab", "مدينتي": "Madinaty",
            "معادي": "Maadi", "نصر": "Nasr City", "اسكندرية": "Alexandria",
            "منصورة": "Mansoura", "عبور": "Obour", "دمياط": "Damietta",
            "بورسعيد": "Port Said", "اسماعيلية": "Ismailia", "سويس": "Suez",
            "غردقة": "Hurghada", "شرم": "Sharm El Sheikh", "جونا": "El Gouna"
        }
        for Arabic_key, English_val in location_map.items():
            if Arabic_key in query:
                return English_val
        return None

    def _filter_by_location(self, query: str, docs: List[Document]) -> List[Document]:
        """Legacy wrapper for backward compatibility or fallback."""
        loc = self._get_location_from_query(query)
        if not loc: return docs
        return [d for d in docs if loc.lower() in d.metadata.get('location', '').lower()]

    def _format_docs_to_string(self, docs: List[Document]) -> str:
        """Helper to format docs for the prompt."""
        return "\n\n".join([f"عقار {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
