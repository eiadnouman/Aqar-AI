import csv
import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def build_index():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hf_cache_dir = os.path.join(project_root, ".cache", "huggingface")
    transformers_cache_dir = os.path.join(hf_cache_dir, "transformers")
    os.makedirs(transformers_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache_dir

    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        cache_folder=hf_cache_dir,
        model_kwargs={"device": "cpu"},
    )

    csv_path = os.path.join(project_root, "data", "properties.csv")
    index_path = os.path.join(project_root, "data", "faiss_index_cloud")

    print(f"Reading properties from {csv_path}...")
    docs = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            url = row.get("url", "").strip()
            if not url:
                continue
            
            title = row.get("title", "").strip()
            desc = row.get("description", "").strip()
            loc = row.get("location_full_name", "").strip()
            ptype = row.get("property_type", "").strip()
            try:
                price = float(row.get("price_value", 0))
            except:
                price = 0.0

            content = f"passage: Title: {title}\nLocation: {loc}\nType: {ptype}\nDescription: {desc}"
            docs.append(Document(page_content=content, metadata={
                "url": url,
                "location": loc,
                "type": ptype,
                "price": price
            }))

    print(f"Loaded {len(docs)} documents. Building FAISS index...")
    if os.path.exists(index_path):
        print(f"Removing old index at {index_path}...")
        shutil.rmtree(index_path)

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_path)
    print("Successfully built and saved new FAISS index.")

if __name__ == "__main__":
    build_index()
