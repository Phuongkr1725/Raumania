import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Medical Image Classifier",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* ===== BACKGROUND ===== */
    .stApp {
        background: radial-gradient(circle at top, #eef3ff 0%, #ffffff 60%);
    }

    /* ===== MAIN CARD ===== */
    .block-container {
        max-width: 1100px;
        margin: auto;
        background: white;
        padding: 3rem 3rem 3.5rem;
        border-radius: 26px;
        box-shadow: 0 30px 90px rgba(0, 0, 0, 0.08);
    }

    /* ===== TITLE GRADIENT ===== */
    .title-gradient {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f8cff, #6a5cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    /* ===== SUBTITLE ===== */
    .subtitle-gradient {
        text-align: center;
        font-size: 1.15rem;
        font-weight: 500;
        color: #5f6c8a;
        margin-bottom: 1.6rem;
    }

    /* ===== INFO BOX ===== */
    .stAlert {
        border-radius: 14px;
        font-size: 0.95rem;
    }

    /* ===== UPLOAD BOX ===== */
    section[data-testid="stFileUploader"] {
        background: #f7f9fc;
        border-radius: 16px;
        padding: 1.3rem;
        border: 1px dashed #c7d2fe;
        margin-bottom: 2rem;
    }

    /* ===== IMAGE CARD ===== */
    img {
        border-radius: 18px;
    }

    div[data-testid="column"] {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 22px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    return YOLO("model/best.pt")

st.markdown(
    """
    <h1 class="title-gradient">Medical Image Classifier</h1>
    <h3 class="subtitle-gradient">Disease Detection from Medical Images (YOLO)</h3>
    """,
    unsafe_allow_html=True
)

st.info("⚠️ This system is for educational support only. Not a medical diagnosis.")

model = load_model()

uploaded_file = st.file_uploader(
    "📤 Upload Medical Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(image, use_container_width=True, caption="Original Image")

    with col2:
        result = model(img_np)[0]
        annotated = result.plot()

        annotated = cv2.resize(
            annotated,
            (img_np.shape[1], img_np.shape[0])
        )

        st.image(
            annotated,
            use_container_width=True,
            caption="Detection Result"
        )
