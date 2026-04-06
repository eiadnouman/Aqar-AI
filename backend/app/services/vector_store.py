import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np
from app.core.config import settings
from app.core.logging import logger


class VectorStoreManager:
    """
    Manages the lifecycle and queries of the local FAISS database.
    Abstracts embedding models and raw dataset interactions.
    """
    def __init__(self):
        self.embeddings = self._initialize_embeddings()
        self.vectorstore: Optional[FAISS] = None
        self.bm25_model: Optional[BM25Okapi] = None
        self.all_docs_list: List[Document] = []
        self.available_locations = set()
        self.available_services: Set[str] = set()
        self.property_lookup: Dict[str, Dict[str, Any]] = {}
        
        self._load_property_catalog()
        
        self._load_index()

    def _initialize_embeddings(self) -> Optional[HuggingFaceEmbeddings]:
        """
        Initializes embeddings with a project-local HuggingFace cache path.
        This avoids failures when the global cache path is not a directory or
        is not writable in some runtimes.
        """
        try:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            hf_cache_dir = os.path.join(project_root, ".cache", "huggingface")
            transformers_cache_dir = os.path.join(hf_cache_dir, "transformers")

            os.makedirs(transformers_cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = hf_cache_dir
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = transformers_cache_dir

            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                cache_folder=hf_cache_dir,
                model_kwargs={"device": "cpu"},
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
            return None

    def _load_index(self):
        """Loads the FAISS index from the configured disk path."""
        if not self.embeddings:
            logger.error("Embeddings are unavailable; skipping FAISS index loading.")
            return

        if os.path.exists(settings.faiss_index_path):
            try:
                # Security Override: We inherently trust our own pre-built FAISS index
                self.vectorstore = FAISS.load_local(
                    settings.faiss_index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector Database loaded successfully.")
                self._extract_dynamic_locations()
                self._initialize_bm25()
            except Exception as e:
                logger.error(f"Failed to load vector index: {e}")
        else:
            logger.error(f"FAISS Index Path NOT FOUND at: {os.path.abspath(settings.faiss_index_path)}")

    def _load_property_catalog(self):
        """
        Loads structured rows from properties.csv to enrich retrieved documents with:
        - coordinates (lat/lon)
        - normalized services around each property
        """
        try:
            project_root = Path(__file__).resolve().parents[3]
            csv_path = project_root / "data" / "properties.csv"
            if not csv_path.exists():
                logger.warning(f"properties.csv not found at: {csv_path}")
                return

            with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for raw_row in reader:
                    row = {self._clean_key(k): (v or "").strip() for k, v in raw_row.items()}
                    url = row.get("url", "")
                    if not url:
                        continue

                    services = self._extract_services(row.get("description", ""))
                    self.available_services.update(services)

                    self.property_lookup[url] = {
                        "title": row.get("title"),
                        "location": row.get("location_full_name"),
                        "type": row.get("property_type"),
                        "listing_intent": self._infer_listing_intent(url),
                        "lat": self._safe_float(row.get("lat")),
                        "lon": self._safe_float(row.get("lon")),
                        "price": self._safe_float(row.get("price_value")),
                        "bedrooms": self._safe_float(row.get("bedrooms")),
                        "bathrooms": self._safe_float(row.get("bathrooms")),
                        "size": self._safe_float(row.get("size_value")),
                        "nearby_services": services,
                    }

            logger.info(
                f"Loaded property catalog enrichment for {len(self.property_lookup)} listings "
                f"and discovered {len(self.available_services)} service tags."
            )
        except Exception as e:
            logger.warning(f"Failed to load property catalog enrichment: {e}")

    def _extract_dynamic_locations(self):
        """Scans the loaded DB to cache all available canonical locations."""
        try:
            if hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
                all_docs = self.vectorstore.docstore._dict.values()
                for doc in all_docs:
                    loc = doc.metadata.get('location')
                    if loc:
                        self.available_locations.add(loc)
                # Merge raw catalog locations as well (useful when metadata is sparse)
                for item in self.property_lookup.values():
                    loc = item.get("location")
                    if loc:
                        self.available_locations.add(loc)
                logger.info(f"Extracted {len(self.available_locations)} dynamic database locations.")
        except Exception as e:
            logger.warning(f"Could not extract dynamic locations from DB: {e}")

    def _initialize_bm25(self):
        """Initializes the Sparse BM25 Keyword Search Index from the FAISS docstore."""
        try:
            if hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
                self.all_docs_list = list(self.vectorstore.docstore._dict.values())
                
                # Tokenize (simple whitespace tokenization for BM25)
                tokenized_corpus = [doc.page_content.lower().split() for doc in self.all_docs_list]
                self.bm25_model = BM25Okapi(tokenized_corpus)
                logger.info("BM25 Keyword Index initialized successfully.")
        except Exception as e:
            logger.error(f"Could not initialize BM25: {e}")

    def hybrid_search(self, query: str, k: int = 100) -> List[Document]:
        """
        Retrieves top K documents using Semantic (FAISS) + Keyword (BM25) search.
        It unifies them using Reciprocal Rank Fusion (RRF) for extreme accuracy.
        """
        if not self.vectorstore or not self.bm25_model:
            return []

        # 1. Semantic FAISS retrieval
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k * 2}) # Pull more for safe intersection
        faiss_docs = faiss_retriever.invoke(query)
        
        # 2. Keyword BM25 retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_model.get_scores(tokenized_query)
        # Get top K indices for BM25
        top_n_idx = np.argsort(bm25_scores)[::-1][:k * 2]
        bm25_docs = [self.all_docs_list[i] for i in top_n_idx if bm25_scores[i] > 0]
        
        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = 1 / (rank + k_const)
        k_const = 60
        rrf_scores = {}

        # Rank FAISS
        for rank, doc in enumerate(faiss_docs):
            url = doc.metadata.get('url', str(id(doc)))
            rrf_scores[url] = rrf_scores.get(url, 0.0) + (1.0 / (rank + k_const))

        # Rank BM25
        for rank, doc in enumerate(bm25_docs):
            url = doc.metadata.get('url', str(id(doc)))
            rrf_scores[url] = rrf_scores.get(url, 0.0) + (1.0 / (rank + k_const))

        # Re-sort all discovered docs by their merged RRF Score
        all_intersection_docs = {d.metadata.get('url', str(id(d))): d for d in faiss_docs + bm25_docs}
        
        # Sort aggressively by RRF DESC
        sorted_urls = sorted(rrf_scores.keys(), key=lambda url: rrf_scores[url], reverse=True)
        final_docs = [all_intersection_docs[url] for url in sorted_urls]

        return self._enrich_docs(final_docs[:k])

    def retrieve(self, query: str, k: int = 100) -> List[Document]:
        """Legacy pass-through (Re-routed to Hybrid internally to prevent codebase rewrites)."""
        return self.hybrid_search(query, k=k)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Direct vector similarity search."""
        if not self.vectorstore:
             return []
        docs = self.vectorstore.similarity_search(query, k=k)
        return self._enrich_docs(docs)

    def _enrich_docs(self, docs: List[Document]) -> List[Document]:
        enriched: List[Document] = []
        for doc in docs:
            meta = doc.metadata if isinstance(doc.metadata, dict) else {}
            url = str(meta.get("url", "")).strip()
            row = self.property_lookup.get(url)

            if row:
                if not meta.get("title") and row.get("title"):
                    meta["title"] = row["title"]
                if not meta.get("location") and row.get("location"):
                    meta["location"] = row["location"]
                if not meta.get("type") and row.get("type"):
                    meta["type"] = row["type"]
                if not meta.get("listing_intent") and row.get("listing_intent"):
                    meta["listing_intent"] = row["listing_intent"]

                lat = row.get("lat", 0.0)
                lon = row.get("lon", 0.0)
                if lat:
                    meta["lat"] = lat
                    meta["latitude"] = lat
                if lon:
                    meta["lon"] = lon
                    meta["longitude"] = lon

                if not meta.get("price") and row.get("price"):
                    meta["price"] = row["price"]
                if not meta.get("bedrooms") and row.get("bedrooms"):
                    meta["bedrooms"] = row["bedrooms"]
                if not meta.get("bathrooms") and row.get("bathrooms"):
                    meta["bathrooms"] = row["bathrooms"]
                if not meta.get("size") and row.get("size"):
                    meta["size"] = row["size"]

                if not meta.get("nearby_services"):
                    meta["nearby_services"] = row.get("nearby_services", [])
            else:
                # Fallback extraction when URL is missing from catalog.
                if not meta.get("nearby_services"):
                    meta["nearby_services"] = self._extract_services(doc.page_content)

            doc.metadata = meta
            enriched.append(doc)
        return enriched

    @staticmethod
    def _infer_listing_intent(url: str) -> str:
        token = (url or "").lower()
        if "/rent/" in token:
            return "rent"
        if "/buy/" in token:
            return "buy"
        return "unknown"

    @staticmethod
    def _clean_key(value: str) -> str:
        return (value or "").replace("\ufeff", "").strip()

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _extract_services(self, text: str) -> List[str]:
        blob = (text or "").lower()
        service_map = {
            "security": ["security", "secure", "gated", "حراسة", "امن", "أمن", "كاميرات"],
            "swimming_pool": ["swimming pool", "pools", "pool", "حمام سباحة", "حمامات سباحة"],
            "green_spaces": ["green space", "landscaped", "garden", "park", "حدائق", "مساحات خضراء", "لاندسكيب"],
            "commercial_area": ["commercial", "mall", "shopping", "shops", "تجاري", "مول", "تسوق"],
            "club_house": ["club house", "clubhouse", "club", "نادي", "كلوب هاوس"],
            "schools": ["school", "schools", "جامعة", "university", "مدرسة", "مدارس"],
            "hospitals": ["hospital", "clinic", "medical", "مستشفى", "عيادة", "طبي"],
            "transport": ["metro", "transport", "bus", "مواصلات", "مترو", "محور", "طريق"],
        }
        found: List[str] = []
        for canonical, keywords in service_map.items():
            if any(keyword in blob for keyword in keywords):
                found.append(canonical)
        return found
