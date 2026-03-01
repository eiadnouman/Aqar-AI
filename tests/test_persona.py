import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from rag_engine import RealEstateRAG

# Disable heavy logging for clean output
logging.getLogger().setLevel(logging.ERROR)

def test_persona_response():
    print("--- Testing Persona Response ---")
    rag = RealEstateRAG()
    
    query = "عندك شقق في اماكن ايه"
    print(f"\nUser Query: {query}")
    
    response, docs = rag.generate_recommendation(query)
    
    print("\n[AI Response Start]")
    print(response)
    print("[AI Response End]")

if __name__ == "__main__":
    test_persona_response()
