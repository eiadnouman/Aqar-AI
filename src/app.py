import streamlit as st
import os
import logging
from rag_engine import RealEstateRAG
from dotenv import load_dotenv

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Aqar AI", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Validations ---
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    st.error("🚨 Critical Error: `HUGGINGFACEHUB_API_TOKEN` is missing from .env file.")
    st.stop()

# --- Custom CSS (Clean & Professional) ---
# --- Custom CSS (Theme-Aware) ---
ST_STYLE = """
<style>
    /* Card Container */
    .property-card {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid var(--text-color-20);
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .property-card:hover { transform: translateY(-2px); border-color: var(--primary-color); }
    
    .price-tag { color: #059669; font-weight: 700; font-size: 1.1rem; }
    .location-tag { opacity: 0.8; font-size: 0.9rem; margin-bottom: 5px; }
    
    .spec-badge { 
        background: var(--background-color); 
        border: 1px solid var(--text-color-20);
        padding: 2px 8px; border-radius: 6px; 
        font-size: 0.8rem; font-weight: 600;
        display: inline-block; margin-right: 5px;
    }
    
    .view-btn {
        display: block; width: 100%; padding: 8px;
        background: var(--primary-color); color: white !important;
        text-align: center; border-radius: 6px;
        text-decoration: none; font-size: 0.9rem;
        margin-top: 10px;
    }
    .view-btn:hover { opacity: 0.9; }
</style>
"""
st.markdown(ST_STYLE, unsafe_allow_html=True)

# --- Initialization ---
@st.cache_resource
def get_engine():
    try:
        return RealEstateRAG()
    except Exception as e:
        logger.error(f"Failed to init RAG Engine: {e}")
        return None

# --- Main App Logic ---
def main():
    # 1. Sidebar
    with st.sidebar:
        st.title("🏠 Aqar AI")
        st.caption("مستشارك العقاري الذكي")
        st.markdown("---")
        
        # Engine Stats
        rag = get_engine()
        if rag and rag.vectorstore:
            count = rag.vectorstore.index.ntotal if hasattr(rag.vectorstore, 'index') else 0
            st.metric("Properties Indexed", f"{count:,}")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info(
            "- Specify location (e.g., 'Zayed', 'New Cairo').\n"
            "- Mention budget (e.g., 'Under 5M').\n"
            "- Ask for details (e.g., '3 Bedrooms')."
        )
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # 2. Chat Interface Setup
    st.title("أهلاً بيك في Aqar 🇪🇬")
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Initial greeting
        st.session_state.messages.append({
            "role": "assistant",
            "content": "أهلاً يا فندم! 👋\nأنا Aqar، معاك عشان نلاقي أنسب بيت ليك.\nتحب ندور فين النهاردة؟ ولا عندك مواصفات معينة في بالك؟"
        })

    # 3. Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If this message has attached property cards, render them
            if "properties" in msg:
                display_properties(msg["properties"])

    # 4. Handle User Input
    if prompt := st.chat_input("Describe your dream home..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            if not rag:
                st.error("Engine failed to load. Check logs.")
            else:
                with st.spinner("Thinking..."):
                    # A. Get Response & Docs
                    response_text, related_docs = rag.generate_recommendation(prompt)
                    
                    # B. Check for [SHOW_CARDS] Intent
                    show_cards = False
                    if "[SHOW_CARDS]" in response_text:
                        show_cards = True
                        response_text = response_text.replace("[SHOW_CARDS]", "").strip()
                    
                    st.markdown(response_text)
                    
                    # C. Conditionally Show Cards (Using the SAME docs used for context)
                    prop_data = None
                    if show_cards and related_docs:
                        display_properties(related_docs)
                        prop_data = related_docs
                    
                    # Save context
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "properties": prop_data 
                    })

def display_properties(docs):
    """Renders document list as visual cards."""
    if not docs: return
    
    st.markdown("##### 🔍 Found Listings")
    cols = st.columns(len(docs) if len(docs) < 4 else 4)
    
    for i, doc in enumerate(docs):
        # Handle grid wrapping if needed, here strictly top K
        full_col = cols[i] if i < len(cols) else cols[0] # Fallback
        
        meta = doc.metadata
        with full_col:
            st.markdown(f"""
            <div class="property-card">
                <div class="price-tag">{meta.get('price', 0):,.0f} EGP</div>
                <div class="location-tag">📍 {meta.get('location', 'Cairo')}</div>
                <div style="margin-bottom:8px;">
                     <span class="spec-badge">🛏️ {meta.get('type')}</span>
                </div>
                <div style="font-size:0.85rem; opacity: 0.7; height:45px; overflow:hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">
                    {doc.page_content.split('Description: ')[-1][:100]}
                </div>
                <a href="{meta.get('url', '#')}" target="_blank" class="view-btn">View Details</a>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
