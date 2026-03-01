import streamlit as st
import os
import requests
import logging
from dotenv import load_dotenv

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Aqar AI | Smart Real Estate Broker", 
    page_icon="🏙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Validations ---
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    st.error("🚨 Critical Error: `HUGGINGFACEHUB_API_TOKEN` is missing from .env file.")
    st.stop()

# --- Custom CSS (Premium Dark/Glass Theme) ---
ST_STYLE = """
<style>
    /* Global Theme */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Header Gradient */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }

    /* Property Card - Glassmorphism */
    .property-card-container {
        border-radius: 15px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .property-card-container:hover {
        transform: translateY(-5px);
        border-color: #00d2ff;
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.2);
    }

    /* Card Text Elements */
    .price-tag {
        color: #00d2ff;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .location-text {
        color: #cccccc;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 5px;
        margin-bottom: 8px;
    }
    .spec-row {
        display: flex;
        gap: 10px;
        font-size: 0.85rem;
        color: #a0a0a0;
        margin-bottom: 10px;
    }
    .spec-item {
        background: rgba(255,255,255,0.1);
        padding: 4px 8px;
        border-radius: 6px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 25px;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }
    
    /* View Details Link */
    .view-link {
        display: inline-block;
        width: 100%;
        text-align: center;
        margin-top: 10px;
        padding: 8px 0;
        background: rgba(0, 210, 255, 0.1);
        color: #00d2ff !important;
        text-decoration: none;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid rgba(0, 210, 255, 0.3);
        transition: all 0.2s;
    }
    .view-link:hover {
        background: rgba(0, 210, 255, 0.2);
    }
</style>
"""
st.markdown(ST_STYLE, unsafe_allow_html=True)

# --- API Connection Helper ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/v1")

def chat_with_api(message: str, session_id: str = "default_session"):
    try:
        response = requests.post(f"{API_BASE_URL}/chat", json={"message": message, "session_id": session_id})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error: {e}")
        return {"answer": "معلش السيرفر مش شغال دلوقتي، تأكد إن الباك-إند شغال.", "properties": []}

# --- GUI Helper: Render Card ---
def render_property_card(doc):
    # Resolve Image Path
    # Meta has 'images/images/property_X/1.jpg'. We append 'data/' because the app runs from root.
    image_rel_path = doc.get('image_url', doc.get('image', ''))
    full_image_path = os.path.join("data", image_rel_path)
    
    # Fallback if image doesn't exist
    if not os.path.exists(full_image_path):
        # Use a placeholder (online transparent placeholder or local default)
        display_image = "https://via.placeholder.com/400x300.png?text=No+Image"
    else:
        display_image = full_image_path

    # Layout: Image Top, Info Bottom
    with st.container():
        st.markdown('<div class="property-card-container">', unsafe_allow_html=True)
        
        # Display Image using Streamlit native widget for best responsiveness
        st.image(display_image, use_container_width=True)
        
        # Info Block
        st.markdown(f"""
            <div style="padding: 15px;">
                <div class="price-tag">{doc.get('price', 0):,.0f} EGP</div>
                <div class="location-tag">📍 {doc.get('location', 'Cairo')}</div>
                <div class="spec-row">
                    <span class="spec-item">🛏️ {int(doc.get('bedrooms', 0))} Beds</span>
                    <span class="spec-item">🚿 {int(doc.get('bathrooms', 0))} Baths</span>
                    <span class="spec-item">📐 {int(doc.get('size', 0))} sqm</span>
                </div>
                <div style="font-size:0.85rem; opacity: 0.7; height:45px; overflow:hidden; margin-bottom:10px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">
                    {doc.get('description', '')}
                </div>
                <a href="{doc.get('url', '#')}" target="_blank" class="view-link">🔗 View Details</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Main App Logic ---
def main():
    # 1. Sidebar
    with st.sidebar:
        st.title("🏙️ Aqar AI")
        st.caption("Smart Real Estate Assistant")
        st.markdown("---")
        
        # Engine Stats
        try:
            health = requests.get(f"http://127.0.0.1:8000/health", timeout=2)
            if health.status_code == 200:
                st.metric("Engine Status", "Online", delta="Connected")
            else:
                st.error("Engine Disconnected")
        except:
            st.error("Engine Offline")
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.info(
            "Try asking:\n"
            "- 'شقة في التجمع بـ 5 مليون'\n"
            "- 'فيلا في الساحل قريبة من البحر'\n"
            "- 'ارخص شقة في زايد'"
        )
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # 2. Hero Section / Chat Header
    st.markdown("# 👋 أهلاً بيك في Aqar AI")
    st.markdown("### مستشارك العقاري الذكي... سألني عن أي عقار في مصر 🇪🇬")
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Initial greeting from Persona
        st.session_state.messages.append({
            "role": "assistant",
            "content": "يا مرحب! أنا Aqar ❤️\nجاهز أساعدك تلاقي بيت أحلامك أو أفضل فرصة استثمار.\n\nبتدور في منطقة معينة في بالك؟"
        })

    # 3. Display Chat Flow
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If this message has attached property cards, render them
            if "properties" in msg and msg["properties"]:
                st.markdown("---")
                # Grid Layout
                cols = st.columns(3)
                for i, doc in enumerate(msg["properties"]):
                    with cols[i % 3]:
                        render_property_card(doc)

    # 4. Handle User Input
    if prompt := st.chat_input("اكتب طلبك هنا (مثال: شقة في التجمع 3 غرف)..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🔮"):
                # Hit API
                api_response = chat_with_api(prompt, session_id="user_streamlit")
                
                response_text = api_response.get("answer", "")
                related_docs = api_response.get("properties", [])
                
                # B. Check for [SHOW_CARDS] Intent
                show_cards = False
                if "[SHOW_CARDS]" in response_text:
                    show_cards = True
                    response_text = response_text.replace("[SHOW_CARDS]", "").strip()
                
                st.markdown(response_text)
                
                # C. Conditionally Show Cards
                prop_data = None
                if show_cards and related_docs:
                    st.markdown("---")
                    # Grid Layout for current response
                    cols = st.columns(3)
                    for i, doc in enumerate(related_docs):
                        with cols[i % 3]:
                            render_property_card(doc)
                    prop_data = related_docs
                    
                    # Save context
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "properties": prop_data 
                    })

if __name__ == "__main__":
    main()
