import sys
import os
import logging
from langchain_core.documents import Document

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from rag_engine import RealEstateRAG

logging.basicConfig(level=logging.INFO)

def test_filtering():
    print("--- Testing Search Filtering ---")
    rag = RealEstateRAG()

    # Case 1: Bedrooms
    # We need to ensure we have data indexed. Assuming index_full_data.py ran via user or we use existing index.
    # Let's mock a vectorstore result if possible, or just run against real index.
    # Since we can't easily mock here without extensive setup, we will test the logic methods directly
    # and then integration test.
    
    print("\n[UnitTest] Testing Extraction Logic...")
    
    q1 = "عايز شقة غرفتين في التجمع"
    f1 = rag._extract_filters(q1)
    print(f"Query: {q1} -> Filters: {f1}")
    assert f1['min_bedrooms'] == 2
    assert f1['location'] == 'New Cairo'
    
    q2 = "villa under 5 million"
    f2 = rag._extract_filters(q2)
    print(f"Query: {q2} -> Filters: {f2}")
    assert f2['max_price'] == 5000000.0
    
    q3 = "شقة 3 غرف"
    f3 = rag._extract_filters(q3)
    print(f"Query: {q3} -> Filters: {f3}")
    assert f3['min_bedrooms'] == 3

    print("\n[UnitTest] Testing Apply Filters...")
    docs = [
        Document(page_content="A", metadata={'bedrooms': 2, 'price': 1000000, 'location': 'New Cairo'}),
        Document(page_content="B", metadata={'bedrooms': 3, 'price': 2000000, 'location': 'New Cairo'}),
        Document(page_content="C", metadata={'bedrooms': 2, 'price': 6000000, 'location': 'Zayed'}),
    ]
    
    # Filter for 2 bedrooms New Cairo
    res1 = rag._apply_filters(docs, {'min_bedrooms': 2, 'max_bedrooms': 2, 'location': 'New Cairo'})
    print(f"Docs: 3 -> Filtered (2 beds, New Cairo): {len(res1)}")
    assert len(res1) == 1
    assert res1[0].page_content == "A"
    
    # Filter for max price 5M
    res2 = rag._apply_filters(docs, {'max_price': 5000000.0})
    print(f"Docs: 3 -> Filtered (Max 5M): {len(res2)}")
    assert len(res2) == 2 # A and B

    print("\n[SUCCESS] Logic verification passed.")

if __name__ == "__main__":
    test_filtering()
