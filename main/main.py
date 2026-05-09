import streamlit as st
import numpy as np
import pickle
from PIL import Image
from helper import load_models, recognize_face


# ================================
# Load Models (cached — loaded only once per session)
# ================================
@st.cache_resource
def get_models():
    return load_models()


# ================================
# Load Face Database (cached)
# ================================
@st.cache_resource
def load_database():
    with open("face_database.pkl", "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["labels"]


# ================================
# UI
# ================================
st.set_page_config(page_title="Face Recognition", layout="centered")

st.title("🔍 Face Recognition System")
st.write("Upload an image to recognize the person.")

threshold = st.slider("Similarity Threshold", 0.3, 0.9, 0.6, 0.01)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Open image — no exif_transpose here either, matching the training pipeline
    img = Image.open(uploaded_file).convert('RGB')

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Load resources
    mtcnn, model = get_models()
    X_emb, y = load_database()

    with st.spinner("Processing..."):
        name, score = recognize_face(img, X_emb, y, mtcnn, model, threshold)

    st.subheader("Result:")

    if name == "No face detected":
        st.warning(f"⚠️ No face detected in the image.")
    elif name == "Unknown":
        st.error(f"❌ Unknown Person (Score: {score:.4f})")
    else:
        st.success(f"✅ {name} (Score: {score:.4f})")