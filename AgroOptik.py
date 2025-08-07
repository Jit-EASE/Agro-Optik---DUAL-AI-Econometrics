import streamlit as st
from PIL import Image
import numpy as np
import random
import io
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import requests
from openai import OpenAI
import base64

# ----------------------------
# CONFIGURATION
# ----------------------------
# Load API keys from Streamlit secrets
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Streamlit page config
st.set_page_config(page_title="AgroOptik – Ireland", layout="wide")

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
@st.cache_data
def get_iot_data():
    # Simulated IoT sensor readings
    return {
        "Soil Moisture": f"{random.uniform(12,18):.1f}%",
        "Air Temp": f"{random.uniform(14,22):.1f} °C",
        "Seedling pH": f"{random.uniform(6.2,6.8):.2f}"
    }

@st.cache_data
def run_econometric_model(purity, moisture):
    # OLS: Germination ~ Purity + Moisture
    X = np.array([
        [91,12.5],[88,13.1],[94,11.0],[90,12.0],[95,10.8],
        [97,12.9],[89,11.2],[93,13.3],[96,10.9],[92,12.6]
    ])
    y = np.array([92,89,96,90,97,98,88,93,99,94])
    model = LinearRegression().fit(X, y)
    pred = model.predict([[purity, moisture]])[0]
    return model, X, y, pred

@st.cache_data
def call_gemini_ai(prompt_text: str) -> str:
    # Call Google Gemini for text generation
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    res = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=15)
    res.raise_for_status()
    data = res.json()
    candidates = data.get('candidates', [])
    return candidates[0].get('content','') if candidates else ""

@st.cache_data
def call_agentic_ai(prompt_text: str) -> str:
    # Call OpenAI GPT-4o Mini for contextual Q&A and analysis
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are AgroOptik, an expert in Irish agri-diagnostics and seed classification."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.6
    )
    return resp.choices[0].message.content

@st.cache_data
def analyze_stress(image: Image.Image) -> tuple[str, float]:
    # Phenotypic stress detection via Gemini from image
    buf = io.BytesIO()
    image.convert('RGB').save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = (
        "Analyze the following base64-encoded crop image and identify any phenotypic stress symptoms. "
        f"Image(base64)={img_b64}"
    )
    response = call_gemini_ai(prompt)
    parts = response.strip().split('(')
    label = parts[0].strip()
    conf = 0.0
    if len(parts) > 1 and parts[1].endswith(')'):
        try:
            conf = float(parts[1][:-1])
        except:
            conf = 0.0
    return label, conf

# ----------------------------
# UI LAYOUT
# ----------------------------
st.title("AgroOptik Ireland | AI-Augmented Crop Health & Econometric Seed Quality Analysis")

# Sidebar: IoT Sensor Feed
with st.sidebar:
    st.header("📡 IoT Sensor Feed")
    for k, v in get_iot_data().items():
        st.metric(k, v)
    st.markdown("---")
    st.info("Simulated sensor data unless live sensors are connected.")

# Main tabs
tabs = st.tabs(["Crop ID & Analysis","Seed Quality","Inspector Panel"])

# Tab 1: Crop Identification & Auto-Detection
with tabs[0]:
    st.header("Crop Identification & AI Insights")
    img_file = st.file_uploader("Upload Crop Image", type=["jpg","png","jpeg"])
    if img_file:
        img = Image.open(img_file).convert('RGB')
        st.image(img, use_container_width=True)

        # Auto-detect crop species
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        prompt_detect = (
            "Identify the crop species shown in this base64-encoded image. "
            f"Image(base64)={img_b64}"
        )
        detected = call_gemini_ai(prompt_detect)
        st.success(f"Detected Crop: {detected}")

        # Fetch scientific details
        prompt_details = (
            f"Provide scientific details (scientific name, family, key traits) for the crop '{detected}' in Ireland."
        )
        details = call_gemini_ai(prompt_details)
        st.markdown("#### Crop Details (via Gemini AI)")
        st.write(details)

        # Phenotypic stress analysis
        stress, sc = analyze_stress(img)
        st.warning(f"Phenotypic Stress: {stress} ({sc*100:.1f}% confidence)")

        # Contextual Q&A via OpenAI
        question = st.text_input("Ask AgroOptik AI about this crop or stress:")
        if question:
            response = call_agentic_ai(f"Crop: {detected}. Stress: {stress}. {question}")
            st.markdown("#### AgroOptik AI Response:")
            st.write(response)

# Tab 2: Seed Purity & OLS 3D Regression
with tabs[1]:
    st.header("Seed Purity & OLS 3D Regression")
    purity = st.slider("Seed Purity (%)", 85, 100, 92)
    moisture = st.slider("Moisture (%)", 10, 16, 12)
    model, X_train, y_train, prediction = run_econometric_model(purity, moisture)
    st.metric("Predicted Germination Rate (%)", f"{prediction:.1f}")

    # Interactive 3D regression using Plotly
    import plotly.graph_objects as go

    # Prepare training scatter points
    scatter = go.Scatter3d(
        x=X_train[:,0], y=X_train[:,1], z=y_train,
        mode='markers', marker=dict(size=5), name='Data Points'
    )

    # Create grid for surface
    purity_range = np.linspace(85, 100, 20)
    moisture_range = np.linspace(10, 16, 20)
    P, M = np.meshgrid(purity_range, moisture_range)
    Z = model.predict(np.column_stack((P.ravel(), M.ravel()))).reshape(P.shape)

    # Surface plot
    surface = go.Surface(
        x=P, y=M, z=Z, colorscale='Viridis', opacity=0.6, name='Regression Surface'
    )

    # Highlight current input
    current = go.Scatter3d(
        x=[purity], y=[moisture], z=[prediction],
        mode='markers', marker=dict(size=8, color='red'), name='Current Input'
    )

    fig = go.Figure(data=[scatter, surface, current])
    fig.update_layout(
        scene=dict(
            xaxis_title='Purity (%)',
            yaxis_title='Moisture (%)',
            zaxis_title='Germination Rate (%)',
            bgcolor='#212121',
            xaxis=dict(backgroundcolor='#212121', gridcolor='white', showbackground=True, zerolinecolor='white'),
            yaxis=dict(backgroundcolor='#212121', gridcolor='white', showbackground=True, zerolinecolor='white'),
            zaxis=dict(backgroundcolor='#212121', gridcolor='white', showbackground=True, zerolinecolor='white')
        ),
        paper_bgcolor='#212121',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

    # AI Q&A for seed metrics
    q2 = st.text_input("Ask AI about seed metrics:")
    if q2:
        resp2 = call_agentic_ai(f"Purity={purity}%, Moisture={moisture}%. {q2}")
        st.markdown("#### Seed AI Response:")
        st.write(resp2)

# Tab 3: Inspector Panel & AI Guidance
with tabs[2]:
    st.header("Inspector Panel & AI Guidance")
    if st.checkbox("Inspector Login (Simulated)"):
        batch = st.text_input("Batch ID")
        notes = st.text_area("Field Notes")
        decision = st.radio("Decision", ["Approved", "Pending", "Rejected"])
        st.success("Entry saved.")
        q3 = st.text_input("Ask AI for regulatory guidance:")
        if q3:
            resp3 = call_agentic_ai(f"Batch={batch}, Notes={notes}. {q3}")
            st.markdown("#### Inspector AI Response:")
            st.write(resp3)

st.caption("Prototype Developed by Jit | Gemini + OpenAI, Econometrics, IoT integrated")
