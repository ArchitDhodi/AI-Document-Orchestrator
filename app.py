import json
import io
import os
import requests
import streamlit as st
from pathlib import Path

# --- Compatibility Imports ---
try:
    import tomllib  # Python 3.11+ standard library
except ImportError:
    try:
        import tomli as tomllib  # Python < 3.11 fallback
    except ImportError:
        tomllib = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# --- Page Config ---
st.set_page_config(page_title="AI Document Orchestrator", layout="wide")


# --- Helpers ---

def get_secret(name: str):
    if hasattr(st, "secrets"):
        if name in st.secrets:
            return st.secrets[name]
        for key in st.secrets:
            if key.lstrip('\ufeff') == name:
                return st.secrets[key]
    return os.environ.get(name)


def load_client():
    if genai is None:
        return None
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai


def extract_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    mime = (uploaded_file.type or "").lower()
    
    if "text" in mime or uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    if "pdf" in mime or uploaded_file.name.lower().endswith(".pdf"):
        if pdfplumber is None:
            return "(pdfplumber not installed; cannot parse PDF)"
        try:
            with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n".join(pages)
        except Exception as e:
            return f"(Error reading PDF: {e})"
    return ""


def build_schema():
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "A detailed 2-3 sentence summary including Vendor, Key Dates, and Main Items."},
            "risk_level": {"type": "string", "description": "High/Medium/Low"},
            "currency": {"type": "string", "description": "Currency symbol or code (e.g. $, €, ₹)"},
            "amount": {"type": "number", "description": "Numeric amount in original currency"},
            "amount_in_usd": {"type": "number", "description": "The amount converted to USD (approximate)"},
            "insights": {
                "type": "array",
                "description": "Top 4 key insights only (e.g. Vendor, Invoice No, Due Date, PO No)",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "value": {"type": "string"}
                    },
                    "required": ["field", "value"]
                }
            }
        },
        "required": ["summary", "insights", "risk_level", "amount", "currency", "amount_in_usd"]
    }


def demo_extraction(question: str):
    return {
        "summary": "(DEMO) API Key missing. Showing placeholder.",
        "risk_level": "High",
        "currency": "$",
        "amount": 12500.0,
        "amount_in_usd": 12500.0,
        "insights": [
            {"field": "question", "value": question or "Example"},
            {"field": "demo_note", "value": "Configure GEMINI_API_KEY to see real data"}
        ]
    }


def call_gemini(_client, text: str, question: str):
    if genai is None:
        st.error("google-generativeai library is not installed.")
        return demo_extraction(question)
        
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        st.warning("GEMINI_API_KEY not found in secrets.")
        return demo_extraction(question)

    if not text.strip():
        st.error("No text extracted from the document.")
        return None

    prompt = (
        "You are an expert document analyst. Extract a comprehensive structured summary. "
        "1. **Summary**: Write a detailed summary identifying Vendor, Purpose, and Dates. "
        "2. **Insights**: Extract exactly the **Top 4** most critical fields (e.g., Invoice No, Vendor, Due Date, PO #). "
        "3. **Currency**: Identify the currency symbol. "
        "4. **Amount**: Extract the total numeric amount. "
        "5. **USD Conversion**: If not in USD, convert to USD approx. "
        "6. **Risk**: Assess Risk Level (High/Medium/Low). "
        "Return strictly valid JSON matching the provided schema."
    )
    
    model_name = "gemini-2.0-flash"

    try:
        model = genai.GenerativeModel(model_name)
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=build_schema(),
            temperature=0,
        )

        response = model.generate_content(
            [prompt, f"Question: {question}", f"Document Text:\n{text}"],
            generation_config=generation_config,
        )
        
        if response.text:
            return json.loads(response.text)
            
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
        if "404" in str(e):
            st.error(f"Model '{model_name}' not found. Check your API Key tier.")
            
    return demo_extraction(question)


def send_to_n8n(n8n_url: str, text: str, extracted_json: dict, question: str, recipient: str):
    if not n8n_url:
        st.info("Demo mode: N8N_WEBHOOK_URL missing.")
        return {"status": "Simulated Sent"}

    payload = {
        "text": text,
        "extracted_json": extracted_json,
        "question": question,
        "recipient_email": recipient,
    }
    
    try:
        resp = requests.post(n8n_url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"n8n call failed: {exc}")
        return None


# --- UI ---

st.markdown(
    """
    <style>
        .stApp { background-color: #f3e7d9; }
        
        /* Headers & Standard Text Orange */
        h1, h2, h3, label, .stMarkdown p, .stSubheader { color: #e6682d !important; }
        
        /* 1) FORCE UPLOADED FILE NAME ORANGE (Detailed Selectors) */
        
        /* Target the small text (file size/name) */
        [data-testid="stFileUploader"] section div[data-testid="stFileUploaderDropzoneInstructions"] > div > small,
        [data-testid="stFileUploader"] div > div > small, 
        .uploadedFileName {
            color: #e6682d !important;
        }
        
        /* Target the actual filename display after upload */
        [data-testid="stFileUploader"] div[data-testid="stFileUploaderFileName"] {
             color: #e6682d !important;
        }

        /* Spinner Text */
        .stSpinner > div > div {
            color: #e6682d !important;
            font-weight: 500;
        }

        /* 2) Status Box Styling (White Card + Left Border) */
        .status-box {
            padding: 12px 15px;
            border-radius: 8px;
            margin: 10px 0;
            background-color: #ffffff;
            border-left: 5px solid;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            color: #333 !important;
        }
        .status-success { border-left-color: #28a745; }
        .status-warning { border-left-color: #ffc107; }
        .status-info    { border-left-color: #17a2b8; }
        
        /* Remove Gap */
        div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
        
        /* 3) Question Card (Black Text Override) */
        .question-card {
            background: #fff7ef; 
            border: 1px solid #f0e0d2;
            padding: 12px; 
            border-radius: 10px; 
            margin-bottom: 20px;
        }
        .question-card, .question-card strong, .question-card div {
            color: #000000 !important;
        }
        
        [data-testid="stFileUploader"] svg { color: #e6682d !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 4) Layout 0.9 : 1.1
left, right = st.columns([0.9, 1.1])

# --- LEFT COLUMN ---
with left:
    st.title("AI Document Orchestrator")
    st.markdown("Upload a PDF/TXT, extract structured data with Gemini, then trigger an n8n alert.")

    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    
    question = "Extract key invoice details including dates, amounts, and vendor info."
    
    # Target Box with Black Text
    st.markdown(
        """
        <div class="question-card">
            <strong>Target Extraction (Top 4):</strong><br>
            &bull; Vendor Name<br>
            &bull; Invoice Number<br>
            &bull; Due Date<br>
            &bull; Total Amount (USD)
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "extracted_json" not in st.session_state:
        st.session_state.extracted_json = None
    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = ""
    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    if uploaded:
        current_text = extract_text(uploaded)
        if st.session_state.last_file != uploaded.name:
            st.session_state.extracted_text = current_text
            st.session_state.last_file = uploaded.name
            
            client = load_client()
            with st.spinner("Analyzing document with Gemini..."):
                st.session_state.extracted_json = call_gemini(client, current_text, question)
        
        # Extraction Complete
        if st.session_state.extracted_json:
            st.markdown(
                '<div class="status-box status-success">✅ Extraction Complete</div>', 
                unsafe_allow_html=True
            )

    if uploaded and st.button("Re-run Analysis", type="primary"):
        client = load_client()
        with st.spinner("Re-analyzing..."):
            st.session_state.extracted_json = call_gemini(
                client, st.session_state.extracted_text, question
            )

    # Status & Alerts
    if st.session_state.extracted_json:
        data = st.session_state.extracted_json
        
        risk = data.get("risk_level", "Low")
        amount = data.get("amount", 0)
        currency = data.get("currency", "$")
        amount_in_usd = data.get("amount_in_usd", 0)
        
        is_high_risk = risk == "High"
        is_high_value = isinstance(amount_in_usd, (int, float)) and amount_in_usd > 500
        
        st.write("") 
        
        if is_high_risk or is_high_value:
            # High Risk Alert
            st.subheader(f"⚠️ Action Required")
            st.markdown(f"**Reason:** {risk} Risk / Amount > $500")
            
            recipient = st.text_input("Recipient email")
            
            if st.button("Send Alert to n8n", disabled=not recipient):
                n8n_url = get_secret("N8N_WEBHOOK_URL")
                with st.spinner("Triggering workflow..."):
                    resp = send_to_n8n(
                        n8n_url, 
                        st.session_state.extracted_text, 
                        st.session_state.extracted_json, 
                        question, 
                        recipient
                    )
                if resp:
                    st.markdown(
                        '<div class="status-box status-success">🚀 Alert triggered successfully</div>', 
                        unsafe_allow_html=True
                    )
        else:
            # No Action Needed
            st.markdown(
                f"""
                <div class="status-box status-success">
                    <div style="font-weight: 600; font-size: 1.05em;">✅ No Action Needed</div>
                    <div style="font-size: 0.9em; color: #555; margin-top: 4px;">
                        Risk: {risk} &bull; Amount: {currency}{amount} (${amount_in_usd:.2f})
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.markdown(
                """
                <div class="status-box status-info">
                    ℹ️ Note: Email alerts are only triggered for High Risk or amounts over $500 USD.
                </div>
                """,
                unsafe_allow_html=True
            )

# --- RIGHT COLUMN ---
with right:
    st.subheader("Analysis Results")
    if st.session_state.extracted_json:
        st.json(st.session_state.extracted_json)
    else:
        st.markdown(
            """
            <div style="
                border: 2px dashed #f0e0d2; 
                padding: 40px; 
                border-radius: 10px; 
                text-align: center; 
                color: #e6682d;">
                Waiting for document...
            </div>
            """, 
            unsafe_allow_html=True
        )