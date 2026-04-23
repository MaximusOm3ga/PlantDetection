from pathlib import Path
import json
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from cnn_model import CNN
import csv


st.set_page_config(
    page_title="Plant Disease Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(900px 500px at 12% 20%, rgba(125, 211, 252, 0.10), transparent 60%),
            radial-gradient(800px 460px at 86% 18%, rgba(196, 181, 253, 0.12), transparent 60%),
            radial-gradient(1000px 580px at 50% 100%, rgba(255, 255, 255, 0.06), transparent 70%),
            linear-gradient(160deg, #070b14 0%, #0b1222 50%, #0f172a 100%);
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.0rem;
        padding-bottom: 1.0rem;
    }

    .orb {
        position: fixed;
        border-radius: 999px;
        filter: blur(2px);
        pointer-events: none;
        z-index: 0;
    }

    .orb-1 {
        width: 180px;
        height: 180px;
        left: 6%;
        top: 22%;
        background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.55), rgba(255,255,255,0.05) 60%, rgba(255,255,255,0.01) 100%);
        box-shadow: 0 0 80px rgba(255,255,255,0.08);
    }

    .orb-2 {
        width: 130px;
        height: 130px;
        right: 8%;
        top: 28%;
        background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.55), rgba(255,255,255,0.06) 65%, rgba(255,255,255,0.01) 100%);
        box-shadow: 0 0 70px rgba(255,255,255,0.08);
    }

    .orb-3 {
        width: 260px;
        height: 260px;
        left: 45%;
        bottom: -40px;
        background: radial-gradient(circle at 35% 35%, rgba(255,255,255,0.30), rgba(255,255,255,0.06) 55%, rgba(255,255,255,0.01) 100%);
        box-shadow: 0 0 120px rgba(255,255,255,0.08);
    }

    .window-shell {
        position: relative;
        z-index: 1;
        margin: 18px auto 20px auto;
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.20);
        background: linear-gradient(
            180deg,
            rgba(255,255,255,0.14) 0%,
            rgba(255,255,255,0.08) 18%,
            rgba(255,255,255,0.05) 100%
        );
        backdrop-filter: blur(18px) saturate(130%);
        -webkit-backdrop-filter: blur(18px) saturate(130%);
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.45);
        overflow: hidden;
    }

    .window-top {
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 16px;
        border-bottom: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.03);
    }

    .dots {
        display: flex;
        gap: 8px;
        align-items: center;
    }

    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        box-shadow: 0 0 8px rgba(0,0,0,0.25) inset;
    }

    .dot.red { background: #ef4444; }
    .dot.yellow { background: #eab308; }
    .dot.green { background: #22c55e; }

    .window-url {
        font-size: 12px;
        color: #94a3b8;
        letter-spacing: 0.02em;
    }

    .window-body {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        padding: 20px 20px 12px 20px;
    }

    .headline h1 {
        font-size: 40px;
        line-height: 1.05;
        margin: 0;
        color: #f8fafc;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .headline p {
        margin-top: 12px;
        margin-bottom: 0;
        color: #cbd5e1;
        font-size: 22px;
        letter-spacing: 0.01em;
    }

    .top-nav {
        margin-top: 6px;
        display: flex;
        gap: 18px;
        color: #a8b3c7;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .glass-card {
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.16);
        background: linear-gradient(
            180deg,
            rgba(255,255,255,0.10) 0%,
            rgba(255,255,255,0.06) 100%
        );
        backdrop-filter: blur(16px) saturate(130%);
        -webkit-backdrop-filter: blur(16px) saturate(130%);
        box-shadow: 0 10px 34px rgba(0,0,0,0.35);
        padding: 16px 16px 10px 16px;
        margin-bottom: 14px;
        position: relative;
        z-index: 1;
    }

    .section-label {
        font-size: 12px;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #9fb0cc;
        margin-bottom: 12px;
    }

    .pred-label {
        font-size: 12px;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #9fb0cc;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .pred-value {
        font-size: 24px;
        line-height: 1.2;
        color: #f8fafc;
        font-weight: 650;
        margin-bottom: 8px;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(255,255,255,0.30);
        border-radius: 14px;
        padding: 8px;
        background: rgba(255,255,255,0.03);
    }

    [data-testid="stFileUploader"] section {
        background: transparent;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 12px;
        padding: 8px 12px;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.24);
        background: linear-gradient(180deg, rgba(255,255,255,0.16), rgba(255,255,255,0.08));
        color: #f8fafc;
        font-weight: 600;
        height: 42px;
    }

    .stButton > button:hover {
        border-color: rgba(255,255,255,0.42);
        background: linear-gradient(180deg, rgba(255,255,255,0.22), rgba(255,255,255,0.10));
    }

    .stAlert {
        border-radius: 12px;
    }

    .footer-text {
        color: #9fb0cc;
        font-size: 12px;
        margin-top: 4px;
    }
    </style>

    """,
    unsafe_allow_html=True
)

def load_class_names(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r["class_index"]), r["class_name"]))
    rows.sort(key=lambda x: x[0])
    return [name for _, name in rows]


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR /"models" / "cnn_classifier_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def pretty_name(name: str) -> str:
    return name.replace("___", " - ").replace("_", " ")
CLASS_CSV = BASE_DIR / "class_order.csv"
CURES_JSON = BASE_DIR / "cures.json"


@st.cache_resource
def load_artifacts():
    class_names = load_class_names(CLASS_CSV)
    model = CNN(num_classes=len(class_names)).to(device)
    state = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, class_names


@st.cache_data
def load_cures(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_label(text: str) -> str:
    return text.replace("_", " ").strip().title()


def render_cure_details(pred_class: str, cures_data: dict):
    entry = cures_data.get(pred_class)
    if not isinstance(entry, dict):
        st.info("No treatment details found for this class.")
        return

    if entry.get("status") == "healthy":
        st.success("Plant appears healthy.")
        advice = entry.get("advice", [])
        if advice:
            st.markdown("**Advice**")
            st.markdown("\n".join(f"- {item}" for item in advice))
        return

    disease_type = entry.get("type")
    if disease_type:
        st.markdown(f"**Type:** {format_label(str(disease_type))}")

    symptoms = entry.get("symptoms", [])
    if symptoms:
        st.markdown("**Symptoms**")
        st.markdown("\n".join(f"- {item}" for item in symptoms))

    treatment = entry.get("treatment", {})
    if isinstance(treatment, dict) and treatment:
        st.markdown("**Treatment**")
        for key, value in treatment.items():
            st.markdown(f"**{format_label(str(key))}**")
            if isinstance(value, list) and value:
                st.markdown("\n".join(f"- {item}" for item in value))
            elif value:
                st.markdown(f"- {value}")

    prevention = entry.get("prevention", [])
    if prevention:
        st.markdown("**Prevention**")
        st.markdown("\n".join(f"- {item}" for item in prevention))

def predict(image: Image.Image, model, class_names):
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
    return class_names[idx], conf

try:
    model, class_names = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model/classes: {e}")
    st.stop()

try:
    cures_data = load_cures(CURES_JSON)
except Exception:
    cures_data = {}

st.markdown(
    """
    <div class="window-shell">
      <div class="window-body">
        <div class="headline">
          <h1>Plant Disease Classifier</h1>
        </div>
        <div class="top-nav">
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown('<div class="glass-card"><div class="section-label">Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
    else:
        image = None
        st.info("Upload an image to start.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card"><div class="section-label">Inference</div>', unsafe_allow_html=True)
    run = st.button("Run prediction", use_container_width=True)
    if run and image is None:
        st.warning("Please upload an image first.")
    if run and image is not None:
        pred_class, confidence = predict(image, model, class_names)
        st.markdown('<div class="pred-label">Predicted class</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pred-value">{pretty_name(pred_class)}</div>', unsafe_allow_html=True)
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.markdown('<div class="pred-label">Relevant details</div>', unsafe_allow_html=True)
        render_cure_details(pred_class, cures_data)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f'<div class="footer-text">Device: {device} | Classes: {len(class_names)}</div>', unsafe_allow_html=True)
