# pipeline/embedder.py
"""
Embedding extraction.

Converts an aligned 224x224 BGR face image into a 2048-dimensional,
L2-normalized feature vector, using a fine-tuned VGGFace2 ResNet50.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class FaceEmbedder:
    """
    Loads the fine-tuned VGGFace2 ResNet50 once at server startup,
    then extracts an embedding from any aligned face image.
    """

    OUTPUT_DIM = 2048

    def __init__(self, weights_path: str):
        """
        Args:
            weights_path : path to the fine-tuned model weights file
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the ResNet50 architecture, then load the fine-tuned
        # weights, which fully replace all layer values.
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, 8631)
        self.model.fc = nn.Identity()

        ft_weights = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ft_weights)

        self.model.eval().to(self.device)
        print(f"[FaceEmbedder] Fine-tuned VGGFace2 ResNet50 loaded on {self.device}")

    def embed(self, face_bgr_224: np.ndarray) -> np.ndarray:
        """
        Args:
            face_bgr_224 : aligned face image, BGR, shape (224, 224, 3)
        Returns:
            L2-normalized (2048,) embedding vector
        """
        if face_bgr_224.shape != (224, 224, 3):
            raise ValueError(
                f"Expected shape (224,224,3), got {face_bgr_224.shape}"
            )

        x = torch.from_numpy(face_bgr_224).float().permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([91.4953, 103.8827, 131.0912]).view(1, 3, 1, 1)
        x = (x - mean).to(self.device)

        with torch.no_grad():
            emb = self.model(x)[0].cpu().numpy()

        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
