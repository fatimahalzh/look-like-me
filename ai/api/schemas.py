# schemas.py
"""
Request/response data shape for the API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class RejectionReason(str, Enum):
    NO_FACE      = "no_face"
    MULTI_FACE   = "multi_face"
    UNCLEAR      = "unclear"
    INVALID_FILE = "invalid_file"
    FACE_TOO_SMALL = "face_too_small" 

class EmbeddingResponse(BaseModel):
    """Response for /embed."""
    accepted:   bool
    details:    str
    reason:     Optional[RejectionReason] = Field(
        None, description="Fixed short code identifying the rejection type, for frontend use"
    )
    face_count: int = Field(..., description="Number of faces found in the image")
    confidence: Optional[float] = None
    embedding:  Optional[List[float]] = Field(
        None, description="2048-dimensional L2-normalized embedding vector"
    )