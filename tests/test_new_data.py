import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from rag_engine import RealEstateRAG

def test_new_data_loading():
    print("--- Testing Data Loading ---")
    rag = RealEstateRAG(index_path="data/test_faiss_index")
    
    # Use a small subset or just load the main file? 
    # Let's try loading the main file but maybe we should mock the reader to stop after 5 rows to save time?
    # Actually, the file is small (2900 lines), so it might take a minute.
    # Let's run it fully to ensure no errors in the whole file.
    
    csv_path = "data/properties.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # To save time for this test, we can just load it. 
    # But wait, load_and_index_data processes the WHOLE file.
    # Let's modify the function locally or just run it?
    # Let's just run it. It's 3000 rows, FAISS with local embeddings might take a bit but should be ok.
    # Actually, 3000 rows with full embeddings might be slow (minutes). 
    # Let's check if we can limit it.
    
    # Hack: I'll manually modify the load_and_index_data in the loop to break after 10 items 
    # inside the rag engine OR I can just create a dummy csv with 10 lines.
    
    dummy_csv = "data/dummy_properties.csv"
    with open(csv_path, 'r', encoding='utf-8') as f:
        head = [next(f) for _ in range(11)]
    
    with open(dummy_csv, 'w', encoding='utf-8') as f:
        f.writelines(head)
        
    print(f"Created dummy CSV with 10 rows at {dummy_csv}")
    
    rag.load_and_index_data(dummy_csv)
    
    # Check Verification
    docs = rag.search("New Cairo", k=1)
    if docs:
        print("\n[SUCCESS] Document Retrieved:")
        print(f"Content: {docs[0].page_content[:100]}...")
        print(f"Metadata: {docs[0].metadata}")
        
        # Verify Image Path
        expected_image_start = "images/images/property_"
        if docs[0].metadata['image'].startswith(expected_image_start):
             print(f"\n[SUCCESS] Image Path format correct: {docs[0].metadata['image']}")
        else:
             print(f"\n[FAIL] Image Path format incorrect: {docs[0].metadata['image']}")
    else:
        print("\n[FAIL] No documents found.")

    # Cleanup
    if os.path.exists(dummy_csv):
        os.remove(dummy_csv)

def test_persona():
    print("\n\n--- Testing Persona ---")
    # Load the real index (or the one we just made)
    rag = RealEstateRAG(index_path="data/test_faiss_index")
    
    # We might need to index data again if the previous step failed or if we want to ensure data exists.
    # Ideally we use the index created in step 1.
    
    query = "عايز شقة في التجمع"
    print(f"Query: {query}")
    response, docs = rag.generate_recommendation(query)
    
    print("\nResponse:")
    print(response)
    
    if "[SHOW_CARDS]" in response:
        print("\n[SUCCESS] Response contains [SHOW_CARDS]")
    else:
        print("\n[FAIL] Response missing [SHOW_CARDS]")

if __name__ == "__main__":
    test_new_data_loading()
    test_persona()
