# ================================
# Imports
# ================================
import os
import numpy as np
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
from facenet_pytorch import MTCNN, InceptionResnetV1


# ================================
# Device
# ================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================================
# Load Models
# ================================
def load_models():
    """
    Load MTCNN and FaceNet models. Call this once and cache with
    @st.cache_resource in main.py to avoid re-loading on every rerun.
    """
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, model


# ================================
# Cosine Similarity
# ================================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ================================
# Generate Embeddings (TRAINING)
# Matches notebook exactly — no exif_transpose, same pipeline.
# ================================
def generate_embeddings(df, mtcnn, model):
    embeddings = []
    labels = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_path']
        person = row['person']

        try:
            # ✅ Match notebook: no exif_transpose during training
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        face = mtcnn(img)
        if face is None:
            continue

        face = face.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(face)

        embeddings.append(emb.squeeze().cpu().numpy())
        labels.append(person)

    X_emb = np.array(embeddings)
    y = np.array(labels)

    # Normalize embeddings
    normalizer = Normalizer(norm='l2')
    X_emb = normalizer.transform(X_emb)

    return X_emb, y


# ================================
# Get Embedding (NEW IMAGE)
# Must match the training pipeline exactly.
# ================================
def get_embedding(img, mtcnn, model):
    """
    Extract a normalized L2 embedding from a PIL Image.

    IMPORTANT: No exif_transpose is applied here, matching the training
    notebook which also did NOT apply exif_transpose during embedding
    generation. Applying it at inference but not training would cause
    embedding space mismatch and degraded recognition accuracy.

    Args:
        img:   PIL.Image (already opened and converted to RGB by caller)
        mtcnn: MTCNN face detector
        model: InceptionResnetV1 FaceNet model in eval mode

    Returns:
        numpy array of shape (512,) or None if no face detected
    """
    try:
        img = img.convert('RGB')
    except Exception:
        return None

    # ✅ No exif_transpose — matches notebook training pipeline
    face = mtcnn(img)

    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(face)

    emb = emb.squeeze().cpu().numpy()

    # Normalize (L2) — matches training normalizer
    emb = emb / np.linalg.norm(emb)

    return emb


# ================================
# Face Recognition
# ================================
def recognize_face(img, X_emb, y, mtcnn, model, threshold=0.6):
    """
    Identify the person in `img` against the database (X_emb, y).

    Args:
        img:       PIL.Image — the uploaded image
        X_emb:     np.ndarray of shape (N, 512) — stored embeddings
        y:         np.ndarray of shape (N,)     — corresponding labels
        mtcnn:     MTCNN face detector
        model:     InceptionResnetV1 in eval mode
        threshold: float — cosine similarity threshold (default 0.6)

    Returns:
        (name: str, score: float)
    """
    query_emb = get_embedding(img, mtcnn, model)

    if query_emb is None:
        return "No face detected", 0.0

    best_score = -1.0
    best_label = "Unknown"

    for emb, label in zip(X_emb, y):
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_label = label

    if best_score >= threshold:
        return best_label, best_score
    else:
        return "Unknown", best_score