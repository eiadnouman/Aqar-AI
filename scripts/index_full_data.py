import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from rag_engine import RealEstateRAG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def index_full_data():
    print("--- Starting Full Data Indexing ---")
    
    # Initialize RAG with the PRODUCTION index path (default)
    # The default in rag_engine.py is "data/faiss_index_cloud"
    rag = RealEstateRAG() 
    
    csv_path = "data/properties.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Reading from: {csv_path}")
    print(f"Target Index: {rag.index_path}")
    
    # Run Indexing
    rag.load_and_index_data(csv_path, batch_size=500)
    
    print("\n[SUCCESS] Indexing Complete!")
    print(f"Total Vectors: {rag.vectorstore.index.ntotal}")

if __name__ == "__main__":
    index_full_data()
