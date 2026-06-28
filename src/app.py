import streamlit as st
import os
import requests
import logging
import uuid
import html
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
if not any(
    [
        os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        os.getenv("GROQ_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
    ]
):
    st.error(
        "🚨 Critical Error: add at least one model key in `.env` "
        "(`GROQ_API_KEY` or `OPENAI_API_KEY` or `HUGGINGFACEHUB_API_TOKEN`)."
    )
    st.stop()

# --- Custom CSS (Premium Dark/Glass Theme) ---
ST_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Tajawal:wght@400;500;700&display=swap');

    /* Global Theme */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
        direction: rtl;
        font-family: 'Cairo', 'Tajawal', sans-serif;
    }

    html, body, [class*="css"] {
        font-family: 'Cairo', 'Tajawal', sans-serif !important;
    }
    
    /* Input Fields RTL */
    .stTextInput input, .stTextArea textarea, .stChatInput input, .stChatInput textarea {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: plaintext;
    }

    /* Chat Messages RTL align */
    .stChatMessage, .stChatMessageContent, .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        direction: rtl;
        text-align: right;
        unicode-bidi: plaintext;
        line-height: 1.9;
        letter-spacing: 0;
    }

    .stMarkdown p, .stMarkdown li {
        font-size: 1.02rem;
    }

    .stMarkdown ul, .stMarkdown ol {
        padding-right: 1.2rem;
        padding-left: 0;
    }
    
    /* Header Gradient */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Cairo', 'Tajawal', sans-serif;
        font-weight: 800;
        text-align: right;
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
        direction: rtl;
        text-align: right;
    }
    .property-card-container:hover {
        transform: translateY(-5px);
        border-color: #00d2ff;
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.2);
    }

    /* Card Text Elements */
    .price-tag {
        color: #00d2ff;
        font-size: 1.15rem;
        font-weight: 800;
        margin-top: 5px;
        line-height: 1.5;
    }
    .location-tag {
        color: #cccccc;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 6px;
        margin: 6px 0 10px 0;
        line-height: 1.8;
        unicode-bidi: plaintext;
    }
    .spec-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        font-size: 0.88rem;
        color: #d5d9e0;
        margin-bottom: 12px;
    }
    .spec-item {
        background: rgba(255,255,255,0.12);
        padding: 4px 8px;
        border-radius: 6px;
        text-align: center;
        white-space: nowrap;
    }
    .property-desc {
        font-size: 0.92rem;
        opacity: 0.84;
        line-height: 1.8;
        min-height: 52px;
        margin-bottom: 10px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        unicode-bidi: plaintext;
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
        font-size: 0.92rem;
        line-height: 1.7;
    }
    .view-link:hover {
        background: rgba(0, 210, 255, 0.2);
    }
</style>
"""
st.markdown(ST_STYLE, unsafe_allow_html=True)

# --- API Connection Helper ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")
API_ROOT_URL = API_BASE_URL.rsplit("/api/v1", 1)[0]

def chat_with_api(message: str, session_id: str = "default_session"):
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": message, "session_id": session_id},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("API timeout while waiting for /chat response.")
        return {
            "answer": "الطلب أخد وقت أطول من المتوقع. جرّب تاني أو قلّل عدد الشروط في الرسالة.",
            "properties": [],
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error: {e}")
        return {"answer": "معلش السيرفر مش شغال دلوقتي، تأكد إن الباك-إند شغال.", "properties": []}


def stream_chat_with_api(message: str, session_id: str = "default_session"):
    """Calls the /chat/stream SSE endpoint. Returns a generator of text tokens and
    a container that will hold the properties once the stream finishes."""
    import json as _json

    result = {"properties": [], "_failed": False}

    def token_generator():
        try:
            resp = requests.post(
                f"{API_BASE_URL}/chat/stream",
                json={"message": message, "session_id": session_id},
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()

            event_type = None
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    event_type = None
                    continue
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                    continue
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    try:
                        data = _json.loads(payload)
                    except _json.JSONDecodeError:
                        data = {}

                    if event_type == "token":
                        text = data.get("text", "")
                        # Strip any SHOW_CARDS tags before displaying
                        text = text.replace("[SHOW_CARDS]", "").replace("SHOW_CARDS", "")
                        if text:
                            yield text
                    elif event_type == "properties":
                        result["properties"] = data.get("properties", [])
                    # event_type == "done" -> stream finished
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            result["_failed"] = True

    return token_generator, result


def track_property_interest(session_id: str, property_id: int):
    try:
        response = requests.post(
            f"{API_BASE_URL}/interactions/property-click",
            json={
                "session_id": session_id,
                "property_id": property_id,
                "event_type": "save_interest",
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Interaction API Error: {e}")
        return {"saved": False, "property_ids": []}


def get_session_recommendations(session_id: str, limit: int = 5):
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend/session",
            json={"session_id": session_id, "limit": limit},
            timeout=20,
        )
        response.raise_for_status()
        return response.json().get("properties", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Session recommendation API Error: {e}")
        return []


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _resolve_card_image(doc):
    placeholder = "https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80"
    raw_value = doc.get("image_url") or doc.get("image") or ""
    candidate = str(raw_value).strip()
    if not candidate:
        return placeholder

    if candidate.startswith(("http://", "https://")):
        return candidate

    normalized = candidate.lstrip("/\\")
    local_candidates = [normalized]
    if not normalized.startswith("data/"):
        local_candidates.append(os.path.join("data", normalized))

    for path in local_candidates:
        if os.path.isfile(path):
            return path

    return placeholder

# --- GUI Helper: Render Card ---
def render_property_card(doc, key_suffix="card"):
    display_image = _resolve_card_image(doc)
    property_id = doc.get("id")
    safe_location = html.escape(str(doc.get("location", "Cairo")))
    safe_description = html.escape(str(doc.get("description", "")))
    safe_url = html.escape(str(doc.get("url", "#")), quote=True)

    # Layout: Image Top, Info Bottom
    with st.container():
        st.markdown('<div class="property-card-container">', unsafe_allow_html=True)
        
        # Display Image using Streamlit native widget for best responsiveness
        st.image(display_image, use_container_width=True)
        
        # Info Block
        lat = _safe_float(doc.get("latitude"))
        lon = _safe_float(doc.get("longitude"))
        map_link = ""
        if lat is not None and lon is not None:
            map_link = f'<a href="https://www.google.com/maps?q={lat},{lon}" target="_blank" class="view-link" style="margin-top:8px;">🗺️ عرض على الخريطة</a>'

        st.markdown(f"""
            <div style="padding: 15px;">
                <div class="price-tag">{doc.get('price', 0):,.0f} جنيه</div>
                <div class="location-tag">📍 {safe_location}</div>
                <div class="spec-row">
                    <span class="spec-item">🛏️ {int(doc.get('bedrooms', 0))} غرف</span>
                    <span class="spec-item">🚿 {int(doc.get('bathrooms', 0))} حمام</span>
                    <span class="spec-item">📐 {int(doc.get('size', 0))} م²</span>
                </div>
                <div class="property-desc">
                    {safe_description}
                </div>
                <a href="{safe_url}" target="_blank" class="view-link">🔗 عرض التفاصيل</a>
                {map_link}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if property_id:
            interest_key = f"interest_{st.session_state.session_id}_{property_id}_{key_suffix}"
            if st.button("⭐ مهتم", key=interest_key, use_container_width=True):
                tracked = track_property_interest(st.session_state.session_id, int(property_id))
                st.session_state.interested_property_ids = tracked.get("property_ids", [])
                recs = get_session_recommendations(st.session_state.session_id)
                if recs:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "حفظت اهتمامك. دي ترشيحات قريبة من اختياراتك الأخيرة:",
                            "properties": recs,
                        }
                    )
                else:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "حفظت اهتمامك. اختار شقة كمان علشان أحسن الترشيحات.",
                        }
                    )
                st.rerun()

        if lat is not None and lon is not None:
            static_map = (
                "https://staticmap.openstreetmap.de/staticmap.php"
                f"?center={lat},{lon}&zoom=14&size=600x260&markers={lat},{lon},red-pushpin"
            )
            st.image(static_map, use_container_width=True)

# --- Main App Logic ---
def main():
    # 1. Sidebar
    with st.sidebar:
        st.title("🏙️ Aqar AI")
        st.caption("المساعد العقاري الذكي")
        st.markdown("---")
        
        # Engine Stats
        try:
            health = requests.get(f"{API_ROOT_URL}/health", timeout=2)
            if health.status_code == 200:
                st.metric("حالة النظام", "شغّال", delta="متصل")
            else:
                st.error("الخادم غير متصل")
        except:
            st.error("الخادم متوقف")
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.info(
            "جرّب تسأل:\n"
            "- شقة في التجمع بـ 5 مليون\n"
            "- فيلا في الساحل قريبة من البحر\n"
            "- أرخص شقة في زايد"
        )
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.interested_property_ids = []
            st.session_state.session_id = str(uuid.uuid4())
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
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "interested_property_ids" not in st.session_state:
        st.session_state.interested_property_ids = []

    # 3. Display Chat Flow
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If this message has attached property cards, render them
            if "properties" in msg and msg["properties"]:
                st.markdown("---")
                # Grid Layout
                cols = st.columns(3)
                for i, doc in enumerate(msg["properties"]):
                    with cols[i % 3]:
                        render_property_card(doc, key_suffix=f"history_{msg_idx}_{i}")

    # 4. Handle User Input
    if prompt := st.chat_input("اكتب طلبك هنا (مثال: شقة في التجمع 3 غرف)..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response with Streaming
        with st.chat_message("assistant"):
            token_gen, stream_result = stream_chat_with_api(prompt, session_id=st.session_state.session_id)
            response_text = st.write_stream(token_gen())

            # Fallback to non-streaming if SSE failed
            if stream_result.get("_failed"):
                api_response = chat_with_api(prompt, session_id=st.session_state.session_id)
                response_text = api_response.get("answer", "")
                related_docs = api_response.get("properties", [])
                st.markdown(response_text)
            else:
                if not isinstance(response_text, str):
                    response_text = str(response_text or "")
                related_docs = stream_result.get("properties", [])

            # Clean up any residual SHOW_CARDS tags
            response_text = response_text.replace("[SHOW_CARDS]", "").replace("SHOW_CARDS", "").strip()

            # Show property cards if returned
            prop_data = None
            if related_docs:
                st.markdown("---")
                cols = st.columns(3)
                for i, doc in enumerate(related_docs):
                    with cols[i % 3]:
                        render_property_card(doc, key_suffix=f"current_{len(st.session_state.messages)}_{i}")
                prop_data = related_docs

            # Save context
            payload = {"role": "assistant", "content": response_text}
            if prop_data:
                payload["properties"] = prop_data
            st.session_state.messages.append(payload)

if __name__ == "__main__":
    main()
