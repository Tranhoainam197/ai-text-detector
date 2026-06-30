from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detector.features import stylometry_features
from detector.perplexity import get_scorer
from detector.classifier import get_classifier
from detector.explainer import explain
from detector.ensemble import combine
from detector.extractors import extract

app = FastAPI(title="AI Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str


def _analyze_text(text: str) -> dict:
    """Shared analysis pipeline used by both /analyze and /upload."""
    text = text.strip()
    if len(text.split()) < 20:
        raise HTTPException(400, "Provide at least 20 words for reliable analysis.")

    feats = stylometry_features(text)
    ppl = get_scorer().score(text)
    clf_out = get_classifier().predict(text)

    ensemble = combine(
        perplexity=ppl,
        feats=feats,
        classifier_ai_prob=clf_out["ai_prob"],
        classifier_loaded=clf_out["model_loaded"],
    )
    explanation = explain(feats, ppl, ensemble["ai_probability"])

    return {
        "ai_probability": round(ensemble["ai_probability"], 4),
        "human_probability": round(ensemble["human_probability"], 4),
        "confidence": round(ensemble["confidence"], 4),
        "perplexity": round(ppl, 2),
        "features": feats,
        "components": ensemble["components"],
        "explanation": explanation,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return _analyze_text(req.text)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Limit upload size (5 MB)
    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(413, "File too large. Maximum 5 MB.")

    try:
        text = extract(file.filename, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to read file: {e}")

    if not text.strip():
        raise HTTPException(400, "The file appears to be empty or unreadable.")

    result = _analyze_text(text)
    # Include the extracted text so the frontend can show it to the user
    result["extracted_text"] = text
    result["filename"] = file.filename
    return result