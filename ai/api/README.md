# Look Like Me AI Service API

A FastAPI service for validating faces and computing embeddings,
using a fine-tuned VGGFace2 ResNet50.

---

## Structure

```
look_like_me_api/
├── main.py                          <- Entry point, /embed endpoint
├── schemas.py                       <- Response data shape
├── pipeline/
│   ├── detector.py                  <- Face detection & validation (MTCNN)
│   ├── aligner.py                   <- Alignment to 224x224
│   └── embedder.py                  <- Embedding extraction (VGGFace2)
├── models/
│   └── vggface2_mtcnn_filtered.pth  <- Fine-tuned model weights
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```
---

## Endpoint

### `POST /embed`

Accepts an image, validates it, and
returns its embedding if accepted.

**Acceptance criteria:**
- Exactly one face detected
- Detection confidence ≥ 0.99
- Detected face size ≥ 60px

Low-confidence detections are filtered out before faces are counted,
so background artifacts or distant people are not mistaken for an
additional face.

---

### Response Fields

| Field        | Type            | Description |
|--------------|-----------------|-------------|
| `accepted`   | bool            | Whether the image was accepted |
| `details`    | string          | Human-readable explanation, for logging/debugging |
| `reason`     | string \| null  | Fixed code identifying the rejection type |
| `face_count` | int             | Number of faces found in the image |
| `confidence` | float \| null   | Detection confidence of the accepted face |
| `embedding`  | array \| null   | 2048-dimensional L2-normalized embedding vector |

**`reason` values** (present only when `accepted` is `false`):

| Value            | Meaning |
|------------------|---------|
| `no_face`        | No face was detected in the image |
| `unclear`        | A candidate was detected, but confidence was too low to trust |
| `multi_face`     | More than one face was detected |
| `face_too_small` | A face was detected, but is smaller than the minimum accepted size |
| `invalid_file`   | The uploaded file could not be read as a valid image |

---

### Example Responses

**Accepted:**
```json
{
  "accepted": true,
  "details": "Face detected successfully.",
  "reason": null,
  "face_count": 1,
  "confidence": 0.998,
  "embedding": [0.0123, -0.0456, ...]
}
```

**Rejected — no face detected:**
```json
{
  "accepted": false,
  "details": "No face was detected in the image.",
  "reason": "no_face",
  "face_count": 0,
  "confidence": null,
  "embedding": null
}
```

**Rejected — low detection confidence:**
```json
{
  "accepted": false,
  "details": "Face detection confidence was too low.",
  "reason": "unclear",
  "face_count": 1,
  "confidence": null,
  "embedding": null
}
```

**Rejected — multiple faces:**
```json
{
  "accepted": false,
  "details": "More than one face was detected (2 faces found).",
  "reason": "multi_face",
  "face_count": 2,
  "confidence": null,
  "embedding": null
}
```

**Rejected — face too small:**
```json
{
  "accepted": false,
  "details": "The detected face is too small.",
  "reason": "face_too_small",
  "face_count": 1,
  "confidence": 0.998,
  "embedding": null
}
```

**Rejected — invalid file:**
```json
{
  "accepted": false,
  "details": "Could not read the uploaded file. Please make sure it is a valid image (jpg, jpeg, png, or webp).",
  "reason": "invalid_file",
  "face_count": 0,
  "confidence": null,
  "embedding": null
}
```