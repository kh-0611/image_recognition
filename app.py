"""
Image Recognition App
Classifies an image into 1,000 ImageNet categories using a pretrained
Vision Transformer / ResNet model from Hugging Face. Runs fully offline
after the first model download. No API key needed.
"""


import streamlit as st
from PIL import Image
from transformers import pipeline
from pathlib import Path


"""
Image Recognition App
Classifies an image into 1,000 ImageNet categories using a pretrained
Vision Transformer / ResNet model from Hugging Face.
Runs fully offline after the first model download. No API key needed.
"""

import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(
    page_title="Image Recognition",
    page_icon="🖼️",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* ---------- Main Background ---------- */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.18), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(168, 85, 247, 0.15), transparent 30%),
        linear-gradient(135deg, #080b16 0%, #0d1222 50%, #080b16 100%);
    color: #f8fafc;
}

/* ---------- Main Container ---------- */

.block-container {
    max-width: 900px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* ---------- Title ---------- */

h1 {
    font-size: 3rem !important;
    font-weight: 800 !important;
    text-align: center;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem !important;
}

/* ---------- Subtitle ---------- */

.stApp p {
    color: #b8c1d9;
}

/* ---------- Main Description ---------- */

div[data-testid="stCaptionContainer"] {
    color: #94a3b8;
}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(15, 23, 42, 0.98),
            rgba(8, 11, 22, 0.98)
        );
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

section[data-testid="stSidebar"] h2 {
    color: #e2e8f0;
    font-weight: 700;
}

section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}

/* ---------- Tabs ---------- */

button[data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #a78bfa !important;
}

div[data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
}

/* ---------- File Uploader ---------- */

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.55);
    border: 1px dashed rgba(129, 140, 248, 0.55);
    border-radius: 18px;
    padding: 12px;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #a78bfa;
    background: rgba(30, 41, 59, 0.7);
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.15);
}

/* ---------- Camera ---------- */

[data-testid="stCameraInput"] {
    border-radius: 18px;
    overflow: hidden;
}

/* ---------- Image ---------- */

[data-testid="stImage"] {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.15);
    box-shadow:
        0 20px 50px rgba(0, 0, 0, 0.35),
        0 0 35px rgba(99, 102, 241, 0.08);
}

/* ---------- Primary Button ---------- */

.stButton > button {
    width: 100%;
    border: none !important;
    border-radius: 14px;
    padding: 0.75rem 1rem;
    font-size: 16px;
    font-weight: 700;
    color: white !important;
    background: linear-gradient(
        135deg,
        #6366f1,
        #8b5cf6,
        #a855f7
    ) !important;
    box-shadow:
        0 8px 25px rgba(99, 102, 241, 0.25);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 12px 35px rgba(139, 92, 246, 0.4);
}

.stButton > button:active {
    transform: scale(0.98);
}

/* ---------- Success Prediction Card ---------- */

div[data-testid="stAlert"] {
    border-radius: 16px !important;
    border: 1px solid rgba(52, 211, 153, 0.25) !important;
    background: rgba(16, 185, 129, 0.08) !important;
}

/* ---------- Section Headers ---------- */

h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

/* ---------- Progress Bar ---------- */

div[data-testid="stProgress"] > div {
    background: rgba(51, 65, 85, 0.7);
    border-radius: 20px;
    height: 9px;
}

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(
        90deg,
        #6366f1,
        #8b5cf6,
        #c084fc
    );
    border-radius: 20px;
}

/* ---------- Selectbox ---------- */

div[data-baseweb="select"] > div {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 12px;
}

/* ---------- Slider ---------- */

div[data-testid="stSlider"] {
    padding-top: 10px;
}

/* ---------- Info Box ---------- */

div[data-testid="stAlert"][kind="info"] {
    border-radius: 15px;
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(96, 165, 250, 0.2);
}

/* ---------- Spinner ---------- */

div[data-testid="stSpinner"] {
    color: #a78bfa;
}

/* ---------- Prediction Text ---------- */

.stProgress + div {
    color: #cbd5e1;
}

/* ---------- Footer ---------- */

.footer {
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
    color: #64748b;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODELS
# =========================================================

MODELS = {
    "ViT Base (Google)": "google/vit-base-patch16-224",
    "ResNet-50 (Microsoft)": "microsoft/resnet-50",
}


@st.cache_resource(
    show_spinner="Loading model (first run downloads it, please wait)..."
)
def load_classifier(model_id: str):
    return pipeline(
        "image-classification",
        model=model_id
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Choose AI Model",
    list(MODELS.keys())
)

top_k = st.sidebar.slider(
    "Number of Predictions",
    1,
    10,
    5
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **🧠 About the Models**

    These pretrained models are trained on
    **ImageNet**, containing 1,000 everyday
    object categories.

    **ViT** → Vision Transformer  
    **ResNet** → Residual Neural Network
    """
)

st.sidebar.markdown("---")

st.sidebar.caption("🔒 No API key required")
st.sidebar.caption("⚡ Runs locally after model download")


# =========================================================
# MAIN UI
# =========================================================

st.title("🖼️ Image Recognition")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        font-size:16px;
        margin-bottom:30px;
    ">
        Upload an image or take a photo and let AI identify what's inside.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# IMAGE INPUT
# =========================================================

tab_upload, tab_camera = st.tabs(
    ["📁  Upload Image", "📷  Take Photo"]
)

image = None


with tab_upload:

    file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp"
        ]
    )

    if file:
        image = Image.open(file).convert("RGB")


with tab_camera:

    photo = st.camera_input(
        "Take a picture"
    )

    if photo:
        image = Image.open(photo).convert("RGB")


# =========================================================
# RECOGNITION
# =========================================================

if image is not None:

    st.markdown("### 📸 Input Image")

    st.image(
        image,
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "🔍  Recognize Image",
        type="primary"
    ):

        classifier = load_classifier(
            MODELS[model_name]
        )

        with st.spinner(
            "🤖 AI is analyzing your image..."
        ):

            results = classifier(
                image,
                top_k=top_k
            )

        # -----------------------------------------
        # Best Prediction
        # -----------------------------------------

        best = results[0]

        st.markdown("### 🎯 AI Result")

        st.success(
            f"**Top Prediction:** {best['label']}  "
            f"({best['score'] * 100:.2f}%)"
        )

        # -----------------------------------------
        # All Predictions
        # -----------------------------------------

        st.markdown("### 📊 Other Predictions")

        for index, r in enumerate(results):

            percentage = r["score"] * 100

            st.write(
                f"**{index + 1}. {r['label']}**"
            )

            st.progress(
                float(r["score"])
            )

            st.caption(
                f"Confidence: {percentage:.2f}%"
            )

else:

    st.info(
        "👆 Upload an image or use the camera to get started."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🧠 Powered by Hugging Face Transformers &nbsp;•&nbsp;
        🔒 Local AI Image Recognition
    </div>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="Image Recognition", page_icon="🖼️", layout="centered")

MODELS = {
    "ViT Base (Google)": "google/vit-base-patch16-224",
    "ResNet-50 (Microsoft)": "microsoft/resnet-50",
}


@st.cache_resource(show_spinner="Loading model (first run downloads it, please wait)...")
def load_classifier(model_id: str):
    return pipeline("image-classification", model=model_id)


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
top_k = st.sidebar.slider("Number of predictions", 1, 10, 5)
st.sidebar.markdown("---")
st.sidebar.caption("Models are trained on ImageNet (1,000 everyday object classes).")

# ---------------- Main ----------------
st.title("🖼️ Image Recognition")
st.write("Upload an image or take a photo, and the model will tell you what it sees.")

tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])
image = None

with tab_upload:
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if file:
        image = Image.open(file).convert("RGB")

with tab_camera:
    photo = st.camera_input("Take a picture")
    if photo:
        image = Image.open(photo).convert("RGB")

if image is not None:
    st.image(image, caption="Input image")

    if st.button("🔍 Recognize", type="primary"):
        classifier = load_classifier(MODELS[model_name])
        with st.spinner("Analyzing image..."):
            results = classifier(image, top_k=top_k)

        best = results[0]
        st.success(f"**Top prediction:** {best['label']}  ({best['score'] * 100:.2f}%)")

        st.subheader("All predictions")
        for r in results:
            st.write(f"{r['label']} — {r['score'] * 100:.2f}%")
            st.progress(float(r["score"]))
else:
    st.info("👆 Upload an image or use the camera to get started.")