# pipeline/detector.py
"""
Face detection and validation.

Runs MTCNN on an image and accepts it only if exactly one face is
found, with sufficient detection confidence (>= 0.99) and a large enough face
region (60 pixels). Detected boxes below the confidence threshold are filtered
out before counting faces, so low-confidence false positives (e.g.
background artifacts, distant people) are not mistaken for a second
face.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN


@dataclass
class DetectionResult:
    success:    bool
    details:    str
    reason:     Optional[str]        = None
    box:        Optional[tuple]      = None
    landmarks:  Optional[np.ndarray] = None
    confidence: Optional[float]      = None
    face_count: int                  = 0

class FaceDetector:

    CONFIDENCE_THRESHOLD = 0.99
    MIN_FACE_SIZE         = 60  # pixels

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
        )
        print(f"[FaceDetector] MTCNN loaded on {self.device}")

    def detect(self, img_pil: Image.Image) -> DetectionResult:
        """
        Args:
            img_pil : RGB PIL image
        Returns:
            DetectionResult — the outcome of detection, plus the
            landmarks needed for alignment if a face was accepted.
        """
        boxes, probs, landmarks = self.mtcnn.detect(img_pil, landmarks=True)

        # No candidate found at all — nothing resembling a face.
        if boxes is None or len(boxes) == 0:
            return DetectionResult(
                success=False,
                details="No face was detected in the image.",
                reason="no_face",
                face_count=0,
            )

        # Candidates exist, but filter out any below the confidence
        # threshold before counting — low-confidence detections (e.g.
        # background artifacts) should not be mistaken for a real face.
        keep_indices = [i for i, p in enumerate(probs) if float(p) >= self.CONFIDENCE_THRESHOLD]

        # Candidates were found, but none were confident enough to trust.
        if len(keep_indices) == 0:
            return DetectionResult(
                success=False,
                details="Face detection confidence was too low.",
                reason="unclear",
                face_count=len(boxes),
            )

        boxes     = [boxes[i] for i in keep_indices]
        probs     = [probs[i] for i in keep_indices]
        landmarks = [landmarks[i] for i in keep_indices]

        face_count = len(boxes)

        if face_count > 1:
            return DetectionResult(
                success=False,
                details=f"More than one face was detected ({face_count} faces found).",
                reason="multi_face",
                face_count=face_count,
            )

        confidence = float(probs[0])
        box = tuple(boxes[0])
        x1, y1, x2, y2 = box
        face_size = min(x2 - x1, y2 - y1)

        if face_size < self.MIN_FACE_SIZE:
            return DetectionResult(
                success=False,
                details="The detected face is too small.",
                reason="face_too_small",
                face_count=face_count,
            )

        return DetectionResult(
            success=True,
            details="Face detected successfully.",
            reason=None,
            box=box,
            landmarks=landmarks[0].astype(np.float32),
            confidence=confidence,
            face_count=face_count,
        )