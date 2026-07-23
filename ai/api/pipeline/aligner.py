# pipeline/aligner.py
"""
Face alignment.

Uses five facial landmarks (eyes, nose tip, mouth corners) to warp a
detected face into a fixed 224x224 reference frame, so the same
facial points always land in the same position regardless of the
original photo's angle or tilt. This produces the exact input size
and orientation expected by the embedding model.
"""

import numpy as np
import cv2
from skimage.transform import SimilarityTransform


# Reference (destination) landmark positions for a 224x224 output.
VGGFACE2_DST_224 = np.array([
    [60.5892, 103.3926],   # Left eye
    [131.0636, 103.0028],  # Right eye
    [96.0504, 143.4732],   # Nose tip
    [67.0986, 184.7310],   # Left mouth corner
    [125.4598, 184.4082],  # Right mouth corner
], dtype=np.float32)


class FaceAligner:
    """Aligns a face to 224x224 using a Similarity Transform."""

    OUTPUT_SIZE = 224

    def align(self, img_bgr: np.ndarray, src_landmarks: np.ndarray) -> np.ndarray:
        """
        Args:
            img_bgr       : full BGR image (as read by OpenCV)
            src_landmarks : (5, 2) array of facial landmarks from MTCNN
        Returns:
            aligned_bgr   : aligned face image, 224x224, BGR
        """
        tform = SimilarityTransform()
        tform.estimate(src_landmarks, VGGFACE2_DST_224)

        aligned_bgr = cv2.warpAffine(
            img_bgr,
            tform.params[0:2],
            (self.OUTPUT_SIZE, self.OUTPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned_bgr
