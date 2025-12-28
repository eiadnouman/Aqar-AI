import os
from rag_engine import RealEstateRAG
from dotenv import load_dotenv

def main():
    load_dotenv()
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("Error: Missing HUGGINGFACEHUB_API_TOKEN in .env")
        return

    print("--- Real Estate AI Setup ---")
    rag = RealEstateRAG()
    
    csv_file = "egypt_real_estate_listings.csv"
    if os.path.exists(csv_file):
        try:
            rag.load_and_index_data(csv_file)
            print("\nSuccess! Run 'run.bat' to start the app.")
        except Exception as e:
            print(f"\nError: {e}")
    else:
        print(f"File {csv_file} not found.")

if __name__ == "__main__":
    main()
