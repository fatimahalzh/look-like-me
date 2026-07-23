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
│   └── vggface2_mtcnn_filtered.pth  <- Fine-tuned model weights (copied manually)
└── requirements.txt
```


## Endpoint

### `POST /embed`

Accepts an image (field name: `profile_photo`), validates it, and
returns its embedding if accepted.

Example response (accepted):
```json
{
  "accepted": true,
  "message": "Face validated and embedding computed successfully.",
  "face_count": 1,
  "confidence": 0.998,
  "embedding": [0.0123, -0.0456, ...]
}
```

Example response (rejected — multiple faces):
```json
{
  "accepted": false,
  "message": "More than one face was detected (2 faces found). Please provide a photo of a single person.",
  "face_count": 2,
  "confidence": null,
  "embedding": null
}
```
