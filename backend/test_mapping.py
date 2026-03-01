import os
import logging
from app.core.rag import RealEstateRAG

# Verify we can extract filters without full vector store loading for this test
# We just want to test the LLM logic

# Mock the class slightly or just instantiate and inject data
rag = RealEstateRAG(index_path="dummy_path") 
# Force inject the new location
rag.available_locations = {"Portsaid", "New Cairo", "Mansoura"}

query = "عايز شقة في بورسعيد"
print(f"Testing Query: {query}")
print(f"Available DB Locations: {rag.available_locations}")

try:
    filters = rag._extract_filters(query)
    print("\n✅ Extracted Filters:")
    print(filters)
    
    if filters.get("location") == "Portsaid":
        print("\n🎉 SUCCESS: Mapped 'بورسعيد' -> 'Portsaid'")
    else:
        print(f"\n❌ FAILED: Expected 'Portsaid', got '{filters.get('location')}'")
except Exception as e:
    print(f"\n❌ Error: {e}")
