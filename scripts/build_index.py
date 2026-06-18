import csv
import json
import os
import shutil
import sys
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Add backend to path so we can import our custom embeddings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from app.services.vector_store import HFInferenceEmbeddings


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value):
    try:
        return int(float(value))
    except Exception:
        return 0


def first_image(value):
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list) and parsed:
                    return str(parsed[0])
            except Exception:
                return stripped
        return stripped
    return ""


def infer_property_kind(*parts):
    text = " ".join(str(part or "").lower() for part in parts)
    if any(token in text for token in ["villa", "فيلا"]):
        return "Villa"
    if any(token in text for token in ["duplex", "دوبلكس"]):
        return "Duplex"
    if any(token in text for token in ["studio", "استوديو"]):
        return "Studio"
    return "Apartment"


def infer_listing_intent(value):
    token = str(value or "").lower()
    if token in {"for_rent", "rent", "rental"}:
        return "rent"
    if token in {"for_sale", "sale", "buy"}:
        return "buy"
    return "unknown"


def docs_from_csv(csv_path):
    print(f"Reading properties from {csv_path}...")
    docs = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("url", "").strip()
            if not url:
                continue

            title = row.get("title", "").strip()
            desc = row.get("description", "").strip()
            loc = row.get("location_full_name", "").strip()
            ptype = row.get("property_type", "").strip()
            price = safe_float(row.get("price_value", 0))
            property_id = safe_int(row.get("property_id") or row.get("id") or row.get("source_index"))

            content = f"passage: Title: {title}\nLocation: {loc}\nType: {ptype}\nDescription: {desc}"
            metadata = {
                "url": url,
                "location": loc,
                "type": ptype,
                "price": price,
            }
            if property_id:
                metadata["property_id"] = property_id
                metadata["id"] = property_id
            docs.append(Document(page_content=content, metadata=metadata))
    return docs


def docs_from_external_api():
    base_url = (os.getenv("EXTERNAL_API_BASE_URL") or "").rstrip("/")
    api_key = (os.getenv("INTERNAL_API_KEY") or "").strip()
    if not base_url or not api_key:
        raise RuntimeError(
            "data/properties.csv is missing and EXTERNAL_API_BASE_URL/INTERNAL_API_KEY are not configured."
        )

    print(f"Syncing active properties from {base_url}/internal/ai-sync...")
    response = requests.get(
        f"{base_url}/internal/ai-sync",
        headers={"x-api-key": api_key},
        timeout=int(os.getenv("EXTERNAL_API_TIMEOUT_SEC", "30")),
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise RuntimeError("External API sync returned a non-list payload.")

    docs = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        property_id = safe_int(row.get("property_id") or row.get("id"))
        if not property_id:
            continue

        title = str(row.get("property_name") or row.get("title") or f"Property {property_id}").strip()
        desc = str(row.get("property_desc") or row.get("description") or "").strip()
        loc = str(row.get("location") or row.get("location_full_name") or "").strip()
        ptype = infer_property_kind(title, desc)
        listing_intent = infer_listing_intent(row.get("property_type"))
        url = str(row.get("url") or f"{base_url}/property/{property_id}").strip()

        content = f"passage: Title: {title}\nLocation: {loc}\nType: {ptype}\nDescription: {desc}"
        metadata = {
            "id": property_id,
            "property_id": property_id,
            "url": url,
            "title": title,
            "location": loc,
            "type": ptype,
            "listing_intent": listing_intent,
            "price": safe_float(row.get("price_value") or row.get("price_per_day")),
            "bedrooms": safe_float(row.get("bedrooms_no") or row.get("bedrooms")),
            "bathrooms": safe_float(row.get("bathrooms_no") or row.get("bathrooms")),
            "size": safe_float(row.get("size") or row.get("size_value")),
            "lat": safe_float(row.get("latitude") or row.get("lat")),
            "lon": safe_float(row.get("longitude") or row.get("lon")),
            "image": first_image(row.get("images")),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def build_index():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, ".env"))

    print("Initializing embeddings...")
    api_token = (os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip() or None
    try:
        embeddings = HFInferenceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            api_token=api_token,
        )
        embeddings.embed_query("test")
        print("Using HF Inference API embeddings (cloud mode).")
    except Exception as e:
        print(f"HF Inference API unavailable ({e}); falling back to local model.")
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_cache_dir = os.path.join(project_root, ".cache", "huggingface")
        os.makedirs(hf_cache_dir, exist_ok=True)
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            cache_folder=hf_cache_dir,
            model_kwargs={"device": "cpu"},
        )
        print("Using local SentenceTransformer embeddings (dev mode).")

    csv_path = os.path.join(project_root, "data", "properties.csv")
    index_path = os.path.join(project_root, "data", "faiss_index_cloud")

    if os.path.exists(csv_path):
        docs = docs_from_csv(csv_path)
    else:
        print(f"{csv_path} not found; falling back to external API sync.")
        docs = docs_from_external_api()

    if not docs:
        raise RuntimeError("No property documents were loaded; aborting FAISS build.")

    print(f"Loaded {len(docs)} documents. Building FAISS index...")
    if os.path.exists(index_path):
        print(f"Removing old index at {index_path}...")
        shutil.rmtree(index_path)

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_path)
    print("Successfully built and saved new FAISS index.")

if __name__ == "__main__":
    build_index()
