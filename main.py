import io
import os
import json
import joblib
from dotenv import load_dotenv

# 1. تفعيل load_dotenv في أول الفايل خالص عشان يقرا المفتاح صح
load_dotenv()

import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# استدعاء الـ SDK الجديد والمستقر بالكامل
from google import genai
from google.genai import types

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
VERSION = "3.1.0"
MCV_NORMAL_THRESHOLD = float(os.getenv("MCV_NORMAL_THRESHOLD", "81.0"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 2. تعريف الـ Client بالطريقة الجديدة الصح
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("⚠️ Warning: GEMINI_API_KEY is not set in environment!")

# ─────────────────────────────────────────
# Load ML Model
# ─────────────────────────────────────────
model = None
label_encoder = None

try:
    model = joblib.load("thalassemia_expert_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model load error: {e}")

# ─────────────────────────────────────────
# Score & Recommendation Maps
# ─────────────────────────────────────────
SCORE_MAP = {
    "Normal": 1,
    "Iron Deficiency Anemia": 2,
    "Borderline": 3,
    "Possible Thalassemia": 5,
    "Alpha Thalassemia Trait": 6,
    "Beta Thalassemia Trait": 6,
    "Beta-Thalassemia Minor": 6,
    "Thalassemia Intermedia": 7,
    "Beta Thalassemia": 8,
    "Beta-Thalassemia Major": 9,
    "Thalassemia Major": 10,
}

REC_MAP = {
    "Normal": "CBC values are within normal range. No signs of thalassemia detected. Routine follow-up recommended.",
    "Iron Deficiency Anemia": "Results suggest iron deficiency anemia. Iron studies (serum ferritin, TIBC) are recommended.",
    "Borderline": "Borderline results detected. Repeat CBC in 3 months and consider hemoglobin electrophoresis.",
    "Possible Thalassemia": "Possible thalassemia indicated. Hemoglobin electrophoresis is strongly recommended.",
    "Alpha Thalassemia Trait": "Alpha thalassemia trait detected. Genetic counselling recommended, especially if planning a family.",
    "Beta Thalassemia Trait": "Beta thalassemia minor (trait) detected. Hemoglobin electrophoresis and genetic counselling recommended.",
    "Beta-Thalassemia Minor": "Beta thalassemia minor (trait) detected. Hemoglobin electrophoresis and genetic counselling recommended.",
    "Thalassemia Intermedia": "Thalassemia intermedia detected. Specialist evaluation and monitoring required.",
    "Beta Thalassemia": "Beta thalassemia detected. Urgent specialist referral recommended.",
    "Beta-Thalassemia Major": "Beta thalassemia major detected. Urgent haematology referral required.",
    "Thalassemia Major": "Thalassemia major strongly indicated. Immediate specialist referral and comprehensive workup required.",
}

# ─────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────
app = FastAPI(title="Thalassemia Diagnosis API", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────
class PatientData(BaseModel):
    hgb: float
    mcv: float
    mch: float
    rbc: float

    @field_validator("hgb")
    @classmethod
    def validate_hgb(cls, v):
        if not (3 <= v <= 25):
            raise ValueError("HGB must be between 3 and 25 g/dL")
        return v

    @field_validator("mcv")
    @classmethod
    def validate_mcv(cls, v):
        if not (40 <= v <= 130):
            raise ValueError("MCV must be between 40 and 130 fL")
        return v

    @field_validator("mch")
    @classmethod
    def validate_mch(cls, v):
        if not (10 <= v <= 50):
            raise ValueError("MCH must be between 10 and 50 pg")
        return v

    @field_validator("rbc")
    @classmethod
    def validate_rbc(cls, v):
        if not (1 <= v <= 8):
            raise ValueError("RBC must be between 1 and 8 ×10¹²/L")
        return v

# ─────────────────────────────────────────
# Core Prediction Logic
# ─────────────────────────────────────────
def run_prediction(hgb: float, mcv: float, mch: float, rbc: float) -> dict:
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="ML model is not available. Check server logs.")

    if rbc == 0:
        raise HTTPException(status_code=422, detail="RBC cannot be zero.")

    mentzer_index = round(mcv / rbc, 2)
    shine_lal = round((mcv ** 2 * mch) / 100, 2)

    features = pd.DataFrame(
        [[mcv, mch, hgb, rbc, mentzer_index, shine_lal]],
        columns=["MCV", "MCH", "HGB", "RBC", "Mentzer_Index", "Shine_Lal"],
    )

    pred_idx = model.predict(features)[0]
    prediction = label_encoder.inverse_transform([pred_idx])[0]

    try:
        proba = model.predict_proba(features)[0]
        confidence = round(float(np.max(proba)), 3)
    except Exception:
        confidence = 1.0

    if mcv >= MCV_NORMAL_THRESHOLD and prediction != "Normal":
        prediction = "Normal"
        confidence = 1.0

    thalassemia_score = SCORE_MAP.get(prediction, 5)
    recommendation = REC_MAP.get(prediction, "Please consult a haematologist for further evaluation.")

    mentzer_note = "supports thalassemia" if mentzer_index < 13 else "suggests IDA or Normal"
    explanation = (
        f"CBC values: HGB={hgb} g/dL, MCV={mcv} fL, MCH={mch} pg, RBC={rbc} x10^12/L. "
        f"Mentzer Index: {mentzer_index} ({mentzer_note}). "
        f"Shine-Lal Index: {shine_lal}. "
        f"Model confidence: {round(confidence * 100)}%."
    )

    return {
        "prediction": prediction,
        "thalassemia_score": thalassemia_score,
        "confidence": confidence,
        "recommendation": recommendation,
        "explanation": explanation,
        "indicators": {
            "mentzer_index": mentzer_index,
            "shine_lal": shine_lal,
        },
    }

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None,
        "version": VERSION,
    }


@app.post("/predict/manual")
def predict_manual(data: PatientData):
    return run_prediction(data.hgb, data.mcv, data.mch, data.rbc)


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    allowed = [".csv", ".xls", ".xlsx"]

    if ext not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported type '{ext}'. Use: {allowed}")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {e}")

    df.columns = [c.strip().upper() for c in df.columns]
    required = {"HGB", "MCV", "MCH", "RBC"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {sorted(missing)}")

    results = []
    for _, row in df.iterrows():
        patient_name = str(row.get("NAME", row.get("name", "Unknown")))
        try:
            result = run_prediction(
                float(row["HGB"]), float(row["MCV"]),
                float(row["MCH"]), float(row["RBC"])
            )
        except HTTPException as e:
            result = {"prediction": f"ERROR: {e.detail}"}
        except Exception as e:
            result = {"prediction": f"ERROR: {str(e)}"}
        results.append({"patient": patient_name, "result": result})

    return results


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Please upload a valid lab report image.")

    # تعديل التحقق ليتماشى مع الـ Client الجديد
    if client is None:
        raise HTTPException(status_code=503, detail="AI image extraction is not configured. Use manual entry.")

    image_bytes = await file.read()

    try:
        # تحويل الصورة لتتوافق مع الـ SDK الجديد باستخدام PIL
        image = Image.open(io.BytesIO(image_bytes))

        prompt = """
        You are a medical data extraction expert.
        Analyze this CBC lab report image and extract exactly these 4 values:
        - HGB (Hemoglobin)
        - MCV (Mean Corpuscular Volume)
        - MCH (Mean Corpuscular Hemoglobin)
        - RBC (Red Blood Cell Count)

        Rules:
        1. Return ONLY a valid JSON object with keys: hgb, mcv, mch, rbc
        2. Values must be floats. If a value cannot be read, use null.
        3. No markdown, no explanation, just raw JSON.
        """

        # استخدام الـ SDK المطور (gemini-2.5-flash هو الأحدث والأسرع)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        extracted = json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI extraction error: {str(e)}")

    hgb = extracted.get("hgb")
    mcv = extracted.get("mcv")
    mch = extracted.get("mch")
    rbc = extracted.get("rbc")

    missing_fields = [k for k, v in {"HGB": hgb, "MCV": mcv, "MCH": mch, "RBC": rbc}.items() if v is None]

    if missing_fields:
        return {
            "status": "partial",
            "message": f"Could not extract: {missing_fields}. Try manual entry instead.",
            "detected": {k: v for k, v in {"HGB": hgb, "MCV": mcv, "MCH": mch, "RBC": rbc}.items() if v is not None},
            "missing_fields": missing_fields,
        }

    try:
        result = run_prediction(float(hgb), float(mcv), float(mch), float(rbc))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "success",
        "diagnosis": result["prediction"],
        "result": result,
        "extracted_values": {"HGB": hgb, "MCV": mcv, "MCH": mch, "RBC": rbc},
    }
