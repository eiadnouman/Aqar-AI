import os
import csv
import re
import logging
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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

    def __init__(self, index_path: str = "faiss_index_cloud"):
        """
        Initialize the RAG engine with embeddings and LLM.
        
        Args:
            index_path (str): Path to persist/load the FAISS index.
        """
        self.index_path = index_path
        self.vectorstore = None
        
        # Load API Keys
        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        if not self.hf_token:
            logger.error("HUGGINGFACEHUB_API_TOKEN is missing in .env")
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is required.")

        # 1. Setup Embeddings (Local Calculation - Faster & Free)
        logger.info("Initializing Embeddings Engine (Local)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # 2. Setup LLM (Groq with Fallback)
        self.llm = self._initialize_llm()
        
        # 3. Load Vector Store
        self._load_index()

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
        """Fallback to Google Flan-T5 via HuggingFace API."""
        repo_id = "google/flan-t5-large"
        logger.info(f"Initializing Fallback Model: {repo_id}")
        return HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.1,
            huggingfacehub_api_token=self.hf_token
        )

    def _load_index(self):
        """Loads the FAISS index from disk if it exists."""
        if os.path.exists(self.index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector Database loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load vector index: {e}")

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
                
                for row in reader:
                    content = self._format_doc_content(row)
                    meta = self._extract_metadata(row)
                    
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
            f"Type: {row.get('type', 'Unknown')}\n"
            f"Location: {row.get('location', 'Unknown')}\n"
            f"Price: {row.get('price', '0')}\n"
            f"Size: {row.get('size', '0')} sqm\n"
            f"Bedrooms: {row.get('bedrooms', '0')} | Bathrooms: {row.get('bathrooms', '0')}\n"
            f"Description: {row.get('description', '')}"
        )

    def _extract_metadata(self, row: Dict) -> Dict:
        """Extracts and cleans metadata for the document."""
        def clean_num(value):
            try:
                return float(re.sub(r'[^\d.]', '', str(value))) if value else 0
            except ValueError:
                return 0

        return {
            "price": clean_num(row.get('price')),
            "size": clean_num(row.get('size')),
            "location": row.get('location', ''),
            "type": row.get('type', ''),
            "url": row.get('url', '#')
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

    def generate_recommendation(self, query: str) -> 'Tuple[str, List[Document]]':
        """
        Generates a persona-driven recommendation based on the query.
        """
        if not self.vectorstore:
            return "النظام بيجهز الداتا... ثواني وراجعلك!", []

        # 1. Retrieve & Filter
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 50})
        raw_docs = retriever.invoke(query)
        filtered_docs = self._filter_by_location(query, raw_docs)
        final_docs = filtered_docs[:5]

        # 2. Construct Prompt
        template = """
        انت "Aqar"، مستشار عقاري مصري محترف وذكي.
        
        البيانات المتاحة (العقارات اللي لقيتها):
        {context}

        كلام العميل: 
        {question}

        قواعد صارمة جداً للرد (لا تخالفها أبداً):
        1. **لو العميل بيسلم بس (أهلاً، السلام عليكم):**
           - رد بجملة واحدة ترحيبية ودودة باللهجة المصرية. 
           - **ممنوع** تعمل أي قوائم (Lists) أو ترقيم (1، 2، 3) في الترحيب. ده ممنوع تماماً.
           - مثال للرد الصح: "أهلاً بيك يا فندم، منورنا! قولي بتدور على إيه النهاردة؟"

        2. **لو العميل بيسأل عن عقار والداتا فيها نتائج ({context} مش فاضية):**
           - اتكلم باختصار عن العقارات اللي لقيتها كأنك بتدردش (مثلاً: "لقيتلك شقة ممتازة في التجمع...").
           - **ممنوع** تسرد التفاصيل الفنية (مساحة، سعر، غرف) في الرد النصي لأنها هتظهر في الكروت.
           - لازم تنهي ردك بكلمة `[SHOW_CARDS]` في سطر لوحدها في الآخر عشان الكروت تظهر.
           - لو نسيت تكتب `[SHOW_CARDS]` الكروت مش هتظهر والعميل هيزعل.

        3. **لو مفيش عقارات مناسبة:**
           - اعتذر بشياكة واقترح بديل.

        الرد بتاعك:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": lambda x: self._format_docs_to_string(final_docs), "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )

        # 3. Generate & Handle Errors
        try:
            response = chain.invoke(query)
            # Handle different return types from LangChain integrations
            content = response.content if hasattr(response, 'content') else str(response)
            return content, final_docs
        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            if not final_docs:
                return "للأسف مش لاقي عقارات بالظبط زي ما طلبت دلوقتي. ممكن نغير شروط البحث؟", []
            
            return f"واجهت مشكلة تقنية بسيطة في التلخيص، لكن بناءً على بحثي، دي أنسب العقارات ليك:\n\n{self._format_docs_to_string(final_docs)}", final_docs

    def _filter_by_location(self, query: str, docs: List[Document]) -> List[Document]:
        """Filters documents based on Arabic location keywords in the query."""
        # Mapping: Arabic Keyword -> English Dataset Value
        location_map = {
            "تجمع": "New Cairo", "خامس": "New Cairo", "new cairo": "New Cairo",
            "زايد": "Sheikh Zayed", "اكتوبر": "6 October",
            "سخنة": "Ain Sokhna", "ساحل": "North Coast",
            "عاصمة": "New Capital", "مستقبل": "Mostakbal City",
            "شروق": "Shorouk", "رحاب": "Rehab", "مدينتي": "Madinaty",
            "معادي": "Maadi", "نصر": "Nasr City"
        }

        target_english = None
        for arabic_key, english_val in location_map.items():
            if arabic_key in query:
                target_english = english_val
                break
        
        if not target_english:
            return docs

        filtered = [d for d in docs if target_english.lower() in d.metadata.get('location', '').lower()]
        return filtered if filtered else []

    def _format_docs_to_string(self, docs: List[Document]) -> str:
        """Helper to format docs for the prompt."""
        return "\n\n".join([f"عقار {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
