![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)

# 🏙️ Smart Egyptian Real Estate Advisor

A sophisticated RAG (Retrieval-Augmented Generation) application designed to act as an intelligent, human-like real estate consultant for the Egyptian market.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![Groq](https://img.shields.io/badge/Groq-Llama3-orange)

## 🌟 Features

-   **Human-like Personality**: Speaks friendly, professional Egyptian Arabic (عامية راقية).
-   **Intelligent Search**: Uses Semantic Search to understand "intent" not just keywords (e.g., "شقة لقطة في التجمع" works!).
-   **Dual-Engine Power**:
    -   **Primary**: Groq (Llama 3 70B) for lightning-fast, high-IQ responses.
    -   **Fallback**: Mixtral-8x7B (via HuggingFace) ensures the app never stops working.
-   **Modern UI**: Built with Streamlit for a clean, responsive experience.

## 🚀 Getting Started

### Prerequisites

-   Python 3.9 or higher.
-   A [HuggingFace API Token](https://huggingface.co/settings/tokens) (Required for logic).
-   A [Groq API Key](https://console.groq.com/keys) (Recommended for speed/quality).

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/eiadnouman/egypt-real-estate-ai.git
    cd egypt-real-estate-ai
    ```

2.  **Create Environment** (Optional but recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration**:
    Create a `.env` file in the root directory:
    ```env
    HUGGINGFACEHUB_API_TOKEN=hf_xxxx...
    GROQ_API_KEY=gsk_xxxx...
    ```

5.  **Initialize Data**:
    Run the setup script to create the vector database:
    ```bash
    python setup.py
    ```

### ▶️ Usage

Run the web application:
```bash
streamlit run app.py
```
*Or simply double-click `run.bat` on Windows.*

## 🛠️ Tech Stack

-   **Framework**: Streamlit
-   **Orchestration**: LangChain
-   **LLMs**: Llama 3 (Groq), Mixtral (HuggingFace)
-   **Embeddings**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
-   **Vector DB**: FAISS

## 📝 License

This project is open source and available for personal and educational use.
