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
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
st.set_page_config(page_title="AgroOptik Ireland", layout="wide")

# ----------------------------
# HELPERS & CACHES
# ----------------------------
def extract_gemini_text(raw):
    if isinstance(raw, dict) and "parts" in raw:
        return "\n\n".join(p.get("text","") for p in raw["parts"])
    if isinstance(raw, str):
        return raw
    return str(raw)

@st.cache_data(show_spinner=False)
def call_agentic_ai(prompt_text: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content":"You are AgroOptik, expert in Irish agri-diagnostics."},
            {"role":"user",   "content":prompt_text}
        ],
        temperature=0.6
    )
    return resp.choices[0].message.content or ""

@st.cache_data(show_spinner=False)
def call_gemini_ai(prompt_text: str):
    headers = {"Content-Type":"application/json", "X-goog-api-key":GEMINI_API_KEY}
    payload = {"contents":[{"parts":[{"text":prompt_text}]}]}
    r = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()  # return full JSON so we can extract parts

@st.cache_data(show_spinner=False)
def run_econometric_model(purity: float, moisture: float):
    X = np.array([[91,12.5],[88,13.1],[94,11.0],[90,12.0],[95,10.8],
                  [97,12.9],[89,11.2],[93,13.3],[96,10.9],[92,12.6]])
    y = np.array([92,89,96,90,97,98,88,93,99,94])
    model = LinearRegression().fit(X,y)
    pred  = model.predict([[purity, moisture]])[0]
    return model, X, y, pred

@st.cache_data(show_spinner=False)
def get_ai_explanation(purity: float, moisture: float, pred: float) -> str:
    prompt = (
        f"Interpret seed purity {purity}% and moisture {moisture}% "
        f"yielding germination rate {pred:.1f}%. Provide one concise agronomic insight."
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
st.title("AgroOptik Ireland | AI & Econometrics in AgTech")

with st.sidebar:
    st.header("📡 IoT Sensor Feed")
    for k,v in get_iot_data().items():
        st.metric(k, v)
    st.markdown("---")
    st.info("Simulated sensor data")

tabs = st.tabs(["Crop ID & Analysis","Seed Quality","Inspector Panel"])

# ---- TAB 1: Crop ID & Analysis ----
with tabs[0]:
    st.header("Crop Identification & Insights")
    img_file = st.file_uploader("Upload Crop Image", type=["jpg","png","jpeg"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_container_width=True)

        # Base64 encode
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Gemini detect
        prompt_detect = f"Identify this crop from the base64 image: Image(base64)={img_b64}"
        raw_detect = call_gemini_ai(prompt_detect)
        text_detect = extract_gemini_text(raw_detect).strip()
        if not text_detect:
            st.error("⚠️ Detection failed. Please upload a clearer image.")
        else:
            st.success(f"Detected Crop: **{text_detect}**")

            # Gemini details
            prompt_details = (
                f"Provide scientific name, family and key traits for '{text_detect}' in Ireland."
            )
            raw_det = call_gemini_ai(prompt_details)
            details = extract_gemini_text(raw_det)
            st.markdown("**Crop Details (Gemini):**")
            st.write(details)

            # Gemini stress
            prompt_stress = f"Analyze phenotypic stress in the image: Image(base64)={img_b64}"
            raw_stress = call_gemini_ai(prompt_stress)
            stress_text = extract_gemini_text(raw_stress) or "No stress detected"
            st.warning(f"Phenotypic Stress:\n\n{stress_text}")

            # Agentic Q&A
            q = st.text_input("Ask AgroOptik AI about this crop/stress:")
            if q:
                resp = call_agentic_ai(f"Crop: {text_detect}. Stress: {stress_text}. {q}")
                st.markdown("**AgroOptik AI Response:**")
                st.write(resp)

# ---- TAB 2: Seed Quality & 3D OLS ----
with tabs[1]:
    st.header("Seed Purity & 3D Econometric Model")
    purity  = st.slider("Seed Purity (%)", 85,100,92)
    moisture= st.slider("Moisture (%)", 10,16,12)

    model, X_train, y_train, pred = run_econometric_model(purity, moisture)
    st.metric("Predicted Germination Rate (%)", f"{pred:.1f}")

    # Hover text
    hover_train = [
        f"P:{x:.1f}%, M:{m:.1f}%, G:{y:.1f}%"
        for x,m,y in zip(X_train[:,0], X_train[:,1], y_train)
    ]
    insight = get_ai_explanation(purity, moisture, pred)
    hover_curr = f"P:{purity}%, M:{moisture}%, Pred:{pred:.1f}%\nInsight:{insight}"

    # 3D plot
    P, M = np.meshgrid(np.linspace(85,100,20), np.linspace(10,16,20))
    Z = model.predict(np.column_stack((P.ravel(), M.ravel()))).reshape(P.shape)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_train[:,0], y=X_train[:,1], z=y_train,
        mode='markers', marker=dict(size=4),
        name='Data', hovertext=hover_train, hoverinfo='text'
    ))
    fig.add_trace(go.Surface(x=P,y=M,z=Z,opacity=0.5,showscale=False,name='OLS Surface'))
    fig.add_trace(go.Scatter3d(
        x=[purity], y=[moisture], z=[pred],
        mode='markers', marker=dict(size=8,symbol='diamond'),
        name='Current', hovertext=[hover_curr], hoverinfo='text'
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='Purity (%)', yaxis_title='Moisture (%)', zaxis_title='Germination (%)',
            bgcolor='#212121', xaxis=dict(color='white'), yaxis=dict(color='white'), zaxis=dict(color='white')
        ),
        paper_bgcolor='#212121', font_color='white',
        width=900, height=600, title="3D OLS Regression with AI Hover"
    )
    st.plotly_chart(fig, use_container_width=False)

    q2 = st.text_input("Ask AI about seed metrics:")
    if q2:
        res2 = call_agentic_ai(f"Purity={purity}%, Moisture={moisture}%. {q2}")
        st.markdown("**Seed AI Response:**")
        st.write(res2)

# ---- TAB 3: Inspector Panel ----
with tabs[2]:
    st.header("Inspector Panel & AI Guidance")
    if st.checkbox("Inspector Login (Simulated)"):
        batch = st.text_input("Batch ID")
        notes = st.text_area("Field Notes")
        decision = st.radio("Decision", ["Approved","Pending","Rejected"])
        st.success("Entry saved.")
        q3 = st.text_input("Ask Inspector AI for guidance:")
        if q3:
            res3 = call_agentic_ai(f"Batch={batch}. Notes={notes}. {q3}")
            st.markdown("**Inspector AI Response:**")
            st.write(res3)

st.caption("Prototype by Jit | AI, Econometrics, IoT integrated")
