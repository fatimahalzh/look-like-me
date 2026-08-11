# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
FastAPI application entry point.

AI service: accepts an image, validates it, and — if
accepted — computes and returns its embedding, all in a single call.

Endpoint:
    POST /embed -> validates an image and returns its embedding.


"""

import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pipeline.detector import FaceDetector
from pipeline.aligner import FaceAligner
from pipeline.embedder import FaceEmbedder
from schemas import EmbeddingResponse


BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "vggface2_mtcnn_filtered.pth")


app = FastAPI(
    title="Look Like Me API",
    description="API for validating faces and computing embeddings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


print("=" * 60)
print("Look Like Me API — Initializing pipeline components...")
print("=" * 60)

detector = FaceDetector()
aligner  = FaceAligner()
embedder = FaceEmbedder(weights_path=WEIGHTS_PATH)

print("=" * 60)
print("All components loaded successfully. API is ready.")
print("=" * 60)


def _compute_embedding(img_pil: Image.Image, detection) -> list:
    """Aligns a detected face and extracts its embedding."""
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    aligned_bgr = aligner.align(img_bgr, detection.landmarks)
    embedding = embedder.embed(aligned_bgr)
    return embedding.tolist()


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_photo(profile_photo: UploadFile = File(...)):
    """
        Validates an uploaded image and, if accepted, computes and
        returns its embedding — all in a single call.
    """
    try:
        contents = await profile_photo.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return EmbeddingResponse(
            accepted=False,
            details="Could not read the uploaded file. Please make sure it is a valid image (jpg, jpeg, png, or webp).",
            reason="invalid_file",
            face_count=0,
        )

    detection = detector.detect(img_pil)

    if not detection.success:
        return EmbeddingResponse(
            accepted=False,
            details=detection.details,
            reason=detection.reason,
            face_count=detection.face_count,
            confidence=detection.confidence,
        )

    embedding = _compute_embedding(img_pil, detection)

    return EmbeddingResponse(
        accepted=True,
        details="Face validated and embedding computed successfully.",
        reason=None,
        face_count=detection.face_count,
        confidence=detection.confidence,
        embedding=embedding,
    )

# trigger CI/CD test