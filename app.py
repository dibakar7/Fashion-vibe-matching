from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uuid
import os
import shutil
from FrameExtractor import FrameExtractor
from YOLOv8Engine import FashionItemsDetector
from VibeClassifier import classify_vibes
from ImageUtils import crop_from_frame, get_clip_embedding, match_with_faiss
import whisper

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



whisper_model = whisper.load_model("base")

@app.post("/upload-reel/")
async def upload_reel(file: UploadFile = File(...), caption: str = str(None)):
    # Generate unique ID for video
    os.makedirs("video_inputs", exist_ok=True)
    video_id = str(uuid.uuid4())
    video_path = f"video_inputs/{video_id}.mp4"

    # Save uploaded file
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    # Text/Caption handling
    result = whisper_model.transcribe(video_path)
    transcript = result.get("text", "")

    # Append caption if provided
    if caption:
        transcript += f" {caption}"


    # Run vibe classification
    vibes = classify_vibes(transcript, use_transformer=True)


    # Extract frames
    frame_folder = f"frames/{video_id}"
    os.makedirs(frame_folder, exist_ok=True)
    frame_count = FrameExtractor(video_path, frame_folder)

    product_matches = []

    for i in range(frame_count):
        frame_path = os.path.join(frame_folder, f"frame_{i:05d}.jpg")
        detections = FashionItemsDetector(frame_path, i)

        for det in detections:
            cropped = crop_from_frame(frame_path, det['bbox'])
            embedding = get_clip_embedding(cropped)
            match = match_with_faiss(embedding)  # returns dict: {match_type, matched_product_id, confidence}
            product_matches.append({
                "type": det["class_name"],
                "color": "unknown",  # Optional: add color detection
                "match_type": match["match_type"],
                "matched_product_id": match["matched_product_id"],
                "confidence": match["confidence"]
            })

    return JSONResponse({
        "video_id": video_id,
        "vibes": vibes,
        "products": product_matches
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
