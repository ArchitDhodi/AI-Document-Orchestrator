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
st.set_page_config(page_title="RiskRadar: AI Document Orchestrator", layout="wide")

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
            return "(pdfplumber not installed)"
        try:
            with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n".join(pages)
        except Exception as e:
            return f"(Error: {e})"
    return ""

# --- Schema 1: Standard Invoice Extraction ---
def build_standard_schema():
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Detailed summary with Vendor, Dates, and Total Amount (with currency)."},
            "risk_level": {"type": "string", "description": "High/Medium/Low"},
            "currency": {"type": "string", "description": "Currency symbol (e.g. $, €, ₹)"},
            "amount": {"type": "number", "description": "Numeric amount"},
            "amount_in_usd": {"type": "number", "description": "Approx USD value"},
            "insights": {
                "type": "array",
                "description": "Top 4 key insights (Vendor, Invoice No, Due Date, Total)",
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

# --- Schema 2: Specific Question Answer ---
def build_qa_schema():
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string", "description": "Direct answer to the user's question."},
            "evidence": {"type": "string", "description": "Quote from document supporting the answer."}
        },
        "required": ["question", "answer", "evidence"]
    }

def call_gemini_standard(_client, text):
    prompt = (
        "You are an expert document analyst. Perform a standard extraction.\n"
        "1. **Summary**: Detailed summary with Vendor, Purpose, and Dates. Include Currency Symbol in text.\n"
        "2. **Insights**: Extract Top 4 critical fields (Vendor, Invoice #, Due Date, Total).\n"
        "3. **Financials**: Extract Amount, Currency, and convert to USD approx.\n"
        "4. **Risk**: High/Medium/Low.\n"
        "Return strictly valid JSON."
    )
    return _generate(_client, text, prompt, build_standard_schema())

def call_gemini_qa(_client, text, question):
    prompt = (
        f"You are a helpful assistant. The user has asked: '{question}'\n"
        "Answer the question based ONLY on the document provided.\n"
        "Return strictly valid JSON."
    )
    return _generate(_client, text, prompt, build_qa_schema())

def _generate(_client, text, prompt, schema):
    if not _client: return None
    try:
        model = _client.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(
            [prompt, f"Document Text:\n{text}"],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0
            )
        )
        return json.loads(resp.text) if resp.text else None
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

def send_to_n8n(n8n_url, text, json_data, recipient):
    if not n8n_url: return {"status": "Simulated", "final_answer": "Demo Answer", "email_body": "Demo Body"}
    try:
        resp = requests.post(n8n_url, json={
            "text": text,
            "extracted_json": json_data,
            "recipient_email": recipient
        }, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"n8n Error: {e}")
        return None

# --- UI Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #f3e7d9; }
    h1, h2, h3, label, p, .stSubheader { color: #e6682d !important; }
    .status-box { padding: 12px; border-radius: 8px; margin: 10px 0; background: #fff; border-left: 5px solid; color: #333 !important; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .status-success { border-left-color: #28a745; }
    .qa-box { background: #fff; padding: 20px; border-radius: 10px; border-left: 5px solid #e6682d; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .uploadedFileName, [data-testid="stFileUploader"] small { color: #e6682d !important; }
    .stSpinner > div > div { color: #e6682d !important; }
    </style>
""", unsafe_allow_html=True)

# --- Layout ---
left, right = st.columns([0.9, 1.1])

with left:
    st.title("RiskRadar: AI Document Orchestrator")
    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    # State Management
    if "std_result" not in st.session_state: st.session_state.std_result = None
    if "qa_result" not in st.session_state: st.session_state.qa_result = None
    if "last_file" not in st.session_state: st.session_state.last_file = None
    if "extracted_text" not in st.session_state: st.session_state.extracted_text = ""

    # File Processing
    if uploaded:
        text = extract_text(uploaded)
        # Reset if new file
        if st.session_state.last_file != uploaded.name:
            st.session_state.extracted_text = text
            st.session_state.last_file = uploaded.name
            st.session_state.std_result = None
            st.session_state.qa_result = None
            # Auto-run standard analysis on upload
            client = load_client()
            with st.spinner("Analyzing document..."):
                st.session_state.std_result = call_gemini_standard(client, text)

    # MODE SELECTION (Radio Button)
    mode = st.radio("Choose Action:", ["Standard Analysis", "Ask Specific Question"])

    if mode == "Standard Analysis":
        st.markdown("""
            <div style="background:#fff7ef; border:1px solid #f0e0d2; padding:12px; border-radius:10px; color:black;">
                <strong>Standard Extraction:</strong><br>&bull; Summary & Risk<br>&bull; Top 4 Insights<br>&bull; Financials
            </div>
        """, unsafe_allow_html=True)

        # Logic for Alerts (Only in Standard Mode)
        if st.session_state.std_result:
            data = st.session_state.std_result
            st.markdown('<div class="status-box status-success">✅ Analysis Loaded</div>', unsafe_allow_html=True)
            
            # Risk Logic
            risk = data.get("risk_level", "Low")
            usd = data.get("amount_in_usd", 0)
            is_alert = risk == "High" or (isinstance(usd, (int, float)) and usd > 500)
            
            st.write("")
            if is_alert:
                st.subheader("⚠️ Action Required")
                st.markdown(f"**Reason:** {risk} Risk / Amount > $500")
                email = st.text_input("Recipient Email")
                if st.button("Send Alert Mail", disabled=not email):
                    with st.spinner("Sending to n8n..."):
                        resp = send_to_n8n(get_secret("N8N_WEBHOOK_URL"), st.session_state.extracted_text, data, email)
                    if resp:
                        st.markdown(f'<div class="status-box status-success">🚀 Status: {resp.get("status", "Sent")}</div>', unsafe_allow_html=True)
                        with st.expander("View Automation Details (JSON)", expanded=True):
                            st.json(resp) # Shows full JSON including email_body
            else:
                 st.markdown(f'<div class="status-box status-success">✅ No Action Needed<br><small>Risk: {risk} | USD: ${usd:.2f}</small></div>', unsafe_allow_html=True)

    elif mode == "Ask Specific Question":
        # Callback to run on Enter
        def run_qa():
            if st.session_state.user_q:
                client = load_client()
                with st.spinner("Thinking..."):
                    st.session_state.qa_result = call_gemini_qa(client, st.session_state.extracted_text, st.session_state.user_q)

        st.text_input("Type your question and press Enter:", key="user_q", on_change=run_qa)

with right:
    # Right column changes based on mode
    if mode == "Standard Analysis":
        st.subheader("Analysis Results")
        if st.session_state.std_result:
            st.json(st.session_state.std_result)
        else:
            st.info("Upload a file to see results.")
            
    elif mode == "Ask Specific Question":
        st.subheader("AI Answer")
        if st.session_state.qa_result:
            # Display QA nicely
            res = st.session_state.qa_result
            st.markdown(f"""
                <div class="qa-box">
                    <h4 style="color:#e6682d; margin-top:0;">Q: {res.get('question')}</h4>
                    <p style="font-size:1.1em; color:#333;"><strong>A:</strong> {res.get('answer')}</p>
                    <hr>
                    <p style="color:#666; font-size:0.9em;"><em>Source: "{res.get('evidence')}"</em></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Ask a question on the left to see the answer here.")
