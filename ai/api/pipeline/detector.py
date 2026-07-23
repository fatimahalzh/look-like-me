# pipeline/detector.py
"""
Face detection and validation.

Runs MTCNN on an image and accepts it only if exactly one face is
found, with sufficient detection confidence. A face that is either
missing or detected with low confidence is treated as "no face
detected" — both return the same message.
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
    message:    str
    box:        Optional[tuple]      = None   # (x1, y1, x2, y2)
    landmarks:  Optional[np.ndarray] = None   # shape (5, 2)
    confidence: Optional[float]      = None
    face_count: int                  = 0      # number of faces found, regardless of acceptance


class FaceDetector:

    CONFIDENCE_THRESHOLD = 0.90

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

        # No face found at all.
        if boxes is None or len(boxes) == 0:
            return DetectionResult(
                success=False,
                message="No face was detected in the image. Please provide a photo with a clear face.",
                face_count=0,
            )

        face_count = len(boxes)

        # More than one face — reject, since the pipeline expects a
        # single person per photo.
        if face_count > 1:
            return DetectionResult(
                success=False,
                message=f"More than one face was detected ({face_count} faces found). Please provide a photo of a single person.",
                face_count=face_count,
            )

        confidence = float(probs[0])

        # Low-confidence detection is treated the same as no face found.
        if confidence < self.CONFIDENCE_THRESHOLD:
            return DetectionResult(
                success=False,
                message="No face was detected in the image. Please provide a photo with a clear face.",
                face_count=0,
            )

        box = tuple(boxes[0])

        return DetectionResult(
            success=True,
            message="Face detected successfully.",
            box=box,
            landmarks=landmarks[0].astype(np.float32),
            confidence=confidence,
            face_count=face_count,
        )