import streamlit as st
from PIL import Image
import numpy as np
import random
import io
import base64
import requests

from sklearn.linear_model import LinearRegression
from openai import OpenAI
import plotly.graph_objects as go

# ----------------------------
# SETUP
# ----------------------------
# Read API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GEMINI_API_KEY  = st.secrets["GEMINI_API_KEY"]
GEMINI_URL      = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="AgroOptik Ireland", layout="wide")

# ----------------------------
# CACHED AI UTILS
# ----------------------------
@st.cache_data(show_spinner=False)
def call_agentic_ai(prompt_text: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are AgroOptik, an expert in Irish agri-diagnostics and seed classification."},
            {"role": "user",   "content": prompt_text}
        ],
        temperature=0.6
    )
    return resp.choices[0].message.content

@st.cache_data(show_spinner=False)
def call_gemini_ai(prompt_text: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    payload = {"contents":[{"parts":[{"text": prompt_text}]}]}
    r = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=15)
    r.raise_for_status()
    data = r.json()
    candidates = data.get("candidates", [])
    return candidates[0].get("content", "") if candidates else ""

@st.cache_data(show_spinner=False)
def run_econometric_model(purity: float, moisture: float):
    X = np.array([
        [91,12.5],[88,13.1],[94,11.0],[90,12.0],[95,10.8],
        [97,12.9],[89,11.2],[93,13.3],[96,10.9],[92,12.6]
    ])
    y = np.array([92,89,96,90,97,98,88,93,99,94])
    model = LinearRegression().fit(X, y)
    pred  = model.predict([[purity, moisture]])[0]
    return model, X, y, pred

@st.cache_data(show_spinner=False)
def get_ai_explanation(purity: float, moisture: float, pred: float) -> str:
    prompt = (
        f"Interpret seed purity {purity}% and moisture {moisture}% "
        f"yielding a germination rate of {pred:.1f}%. "
        "Give one concise agronomic insight."
    )
    return call_agentic_ai(prompt)

def get_iot_data():
    return {
        "Soil Moisture": f"{random.uniform(12,18):.1f}%",
        "Air Temp":      f"{random.uniform(14,22):.1f} °C",
        "Seedling pH":   f"{random.uniform(6.2,6.8):.2f}"
    }

# ----------------------------
# UI LAYOUT
# ----------------------------
st.title("AgroOptik Ireland | AI-Augmented Crop & Seed Insights")

with st.sidebar:
    st.header("📡 IoT Sensor Feed")
    for k, v in get_iot_data().items():
        st.metric(k, v)
    st.markdown("---")
    st.info("Simulated sensor data")

tabs = st.tabs(["Crop ID & Analysis", "Seed Quality", "Inspector Panel"])

# ----------- TAB 1: Crop ID & Analysis -----------
with tabs[0]:
    st.header("Crop Identification & Contextual Insights")
    img_file = st.file_uploader("Upload Crop Image", type=["jpg","png","jpeg"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_container_width=True)

        # Base64-encode the image
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # 1) Auto-detect crop via Gemini
        prompt_detect = (
            "Identify this crop from the base64-encoded image. "
            f"Image(base64)={img_b64}"
        )
        detected = call_gemini_ai(prompt_detect).strip()
        st.success(f"Detected Crop: **{detected}**")

        # 2) Fetch scientific details via Gemini
        prompt_details = (
            f"Provide scientific name, family, and key traits for '{detected}' in Ireland."
        )
        details = call_gemini_ai(prompt_details)
        st.markdown("**Crop Details (Gemini):**")
        st.write(details)

        # 3) Analyze phenotypic stress via Gemini
        prompt_stress = (
            "Analyze this base64-encoded image for phenotypic stress symptoms. "
            f"Image(base64)={img_b64}"
        )
        stress_info = call_gemini_ai(prompt_stress).strip()
        st.warning(f"Phenotypic Stress: {stress_info}")

        # 4) Agentic AI Q&A
        q_crop = st.text_input("Ask AgroOptik AI about this crop / stress:")
        if q_crop:
            ans = call_agentic_ai(f"Crop: {detected}. Stress: {stress_info}. {q_crop}")
            st.markdown("**AgroOptik AI Response:**")
            st.write(ans)

# ----------- TAB 2: Seed Quality -----------
with tabs[1]:
    st.header("Seed Purity & 3D OLS Econometric Model")
    purity   = st.slider("Seed Purity (%)", 85, 100, 92)
    moisture = st.slider("Moisture (%)", 10, 16, 12)

    model, X_train, y_train, prediction = run_econometric_model(purity, moisture)
    st.metric("Predicted Germination Rate (%)", f"{prediction:.1f}")

    # Hover text for training data
    hover_train = [
        f"Purity {x:.1f}%, Moisture {m:.1f}%, Germ {y:.1f}%"
        for x, m, y in zip(X_train[:,0], X_train[:,1], y_train)
    ]
    # AI insight for current input
    insight       = get_ai_explanation(purity, moisture, prediction)
    hover_current = (
        f"Purity {purity}%, Moisture {moisture}%, Pred {prediction:.1f}%\n"
        f"Insight: {insight}"
    )

    # Build interactive 3D Plotly scene
    P, M = np.meshgrid(np.linspace(85,100,20), np.linspace(10,16,20))
    Z    = model.predict(np.column_stack((P.ravel(), M.ravel()))).reshape(P.shape)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_train[:,0], y=X_train[:,1], z=y_train,
        mode='markers', marker=dict(size=4),
        name='Historical Data',
        hovertext=hover_train, hoverinfo='text'
    ))
    fig.add_trace(go.Surface(
        x=P, y=M, z=Z, opacity=0.5, showscale=False,
        name='OLS Surface'
    ))
    fig.add_trace(go.Scatter3d(
        x=[purity], y=[moisture], z=[prediction],
        mode='markers', marker=dict(size=8, symbol='diamond'),
        name='Current Input', hovertext=[hover_current],
        hoverinfo='text'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Purity (%)',
            yaxis_title='Moisture (%)',
            zaxis_title='Germination (%)',
            bgcolor='#212121',
            xaxis=dict(color='white'),
            yaxis=dict(color='white'),
            zaxis=dict(color='white')
        ),
        paper_bgcolor='#212121',
        font_color='white',
        width=900, height=600,
        title_text="3D OLS Regression with NLP Hover"
    )
    st.plotly_chart(fig, use_container_width=False)

    # Seed Q&A
    q_seed = st.text_input("Ask AgroOptik AI about seed metrics:")
    if q_seed:
        ans2 = call_agentic_ai(f"Purity={purity}%, Moisture={moisture}%. {q_seed}")
        st.markdown("**Seed AI Response:**")
        st.write(ans2)

# ----------- TAB 3: Inspector Panel -----------
with tabs[2]:
    st.header("Inspector Panel & AI Guidance")
    if st.checkbox("Inspector Login (Simulated)"):
        batch    = st.text_input("Batch ID")
        notes    = st.text_area("Field Notes")
        decision = st.radio("Decision", ["Approved", "Pending", "Rejected"])
        st.success("Entry saved.")
        q_ins = st.text_input("Ask Inspector AI for guidance:")
        if q_ins:
            ans3 = call_agentic_ai(f"Batch={batch}. Notes={notes}. {q_ins}")
            st.markdown("**Inspector AI Response:**")
            st.write(ans3)

st.caption("Prototype by Jit | AI, Econometrics, IoT integrated")
