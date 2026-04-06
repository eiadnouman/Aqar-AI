import os
from typing import List, Optional
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

    def _extract_dynamic_locations(self):
        """Scans the loaded DB to cache all available canonical locations."""
        try:
            if hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
                all_docs = self.vectorstore.docstore._dict.values()
                for doc in all_docs:
                    loc = doc.metadata.get('location')
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

        return final_docs[:k]

    def retrieve(self, query: str, k: int = 100) -> List[Document]:
        """Legacy pass-through (Re-routed to Hybrid internally to prevent codebase rewrites)."""
        return self.hybrid_search(query, k=k)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Direct vector similarity search."""
        if not self.vectorstore:
             return []
        return self.vectorstore.similarity_search(query, k=k)
