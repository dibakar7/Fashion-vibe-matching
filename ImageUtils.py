import open_clip
import torch
from torchvision import transforms
import faiss
import numpy as np
from PIL import Image
import pickle

# --- Load CLIP Model (globally) ---
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32-quickgelu', pretrained='openai'
)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

# --- Load FAISS Index and Metadata ---
FAISS_INDEX_PATH = "faiss_utils/faiss_catalog.index"
META_DATA_PATH = "faiss_utils/catalog_meta.pkl"

faiss_index = faiss.read_index(FAISS_INDEX_PATH)

with open(META_DATA_PATH, "rb") as f:
    catalog_meta = pickle.load(f)

# --- Functions ---

def crop_from_frame(frame_path, bbox):
    """
    Crop a region from the frame using YOLO bounding box [x, y, w, h]
    """
    image = Image.open(frame_path).convert("RGB")
    x, y, w, h = map(int, bbox)
    cropped = image.crop((x, y, x + w, y + h))
    return cropped

def get_clip_embedding(image: Image.Image):
    """
    Generate normalized CLIP embedding from PIL image
    """
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

def match_with_faiss(embedding, top_k=1):
    """
    Search the FAISS index for the closest product embedding match.
    """
    embedding = embedding / np.linalg.norm(embedding)
    D, I = faiss_index.search(embedding.reshape(1, -1).astype("float32"), top_k)
    best_score = float(D[0][0])
    best_idx = int(I[0][0])

    if best_score > 0.9:
        match_type = "exact"
    elif best_score > 0.75:
        match_type = "similar"
    else:
        match_type = "no_match"

    return {
        "match_type": match_type,
        "matched_product_id": catalog_meta[best_idx]["id"],
        "confidence": best_score,
        "product_info": catalog_meta[best_idx]
    }
