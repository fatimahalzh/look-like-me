# schemas.py
"""
Request/response data shape for the API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingResponse(BaseModel):
    """Response for /embed."""
    accepted:   bool
    message:    str
    face_count: int = Field(..., description="Number of faces found in the image")
    confidence: Optional[float] = None
    embedding:  Optional[List[float]] = Field(
        None, description="2048-dimensional L2-normalized embedding vector"
    )
