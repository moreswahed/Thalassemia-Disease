import joblib
import pandas as pd
import numpy as np
import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import pytesseract
import re
from fastapi import File, UploadFile
from PIL import Image
import os
import json
import google.generativeai as genai  # المكتبة الجديدة للـ Agent
from pydantic import BaseModel

GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GENAI_API_KEY)

# الـ Schema اللي بتجبر الـ Agent يطلع البيانات مظبوطة
class ExtractedCBC(BaseModel):
    hgb: float = None
    mcv: float = None
    mch: float = None
    rbc: float = None


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI(title="Thalassemia Diagnosis System")

try:
    model = joblib.load('thalassemia_expert_model.pkl')
    le = joblib.load('label_encoder.pkl')
except Exception as e:
    print(f" error: {e}")




# التعديل هنا: غيرنا .joblib لـ .pkl
model = joblib.load('thalassemia_expert_model.pkl') 

# اتأكد برضه لو الـ encoder امتداده .pkl غيره زيهم
label_encoder = joblib.load('label_encoder.pkl')
# 2. 
class PatientData(BaseModel):
    hgb: float
    mcv: float
    mch: float
    rbc: float

# --- دالة الحسابات الطبية (الـ 6 Features) ---
def process_prediction(hgb, mcv, mch, rbc):
    mentzer = mcv / rbc
    shine = (mcv**2 * mch) / 100
    
    input_df = pd.DataFrame([[mcv, mch, hgb, rbc, mentzer, shine]], 
                             columns=['MCV', 'MCH', 'HGB', 'RBC', 'Mentzer_Index', 'Shine_Lal'])
    
    pred_idx = model.predict(input_df)[0]
    diagnosis = le.inverse_transform([pred_idx])[0]
    
    # تحسين منطقي بسيط
    if mcv >= 81 and diagnosis != 'Normal':
        diagnosis = 'Normal'
        
    return {
        "diagnosis": diagnosis,
        "indicators": {"mentzer_index": round(mentzer, 2), "shine_lal": round(shine, 2)}
    }



# 1.  (Manual)
@app.post("/predict/manual")
async def predict_manual(data: PatientData):
    return process_prediction(data.hgb, data.mcv, data.mch, data.rbc)

# 2.  (Excel/CSV)
@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(contents))
    
    results = []
    for _, row in df.iterrows():
        res = process_prediction(row['HGB'], row['MCV'], row['MCH'], row['RBC'])
        results.append({"patient": row.get('Name', 'Unknown'), "result": res})
    return results

@app.post("/predict/image")  # سبنا نفس الاسم عشان الفرونت إند يفضل شغال تمام
async def predict_with_agent(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a valid lab report image.")
    
    try:
        # قراءة الصورة كـ Bytes وباصيتها للـ Agent
        image_bytes = await file.read()
        
        ai_model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """
        You are an expert medical data extraction agent. 
        Analyze this CBC lab report image and extract exactly these 4 features:
        - HGB (Hemoglobin)
        - MCV (Mean Corpuscular Volume)
        - MCH (Mean Corpuscular Hemoglobin)
        - RBC (Red Blood Cell Count)
        
        CRITICAL RULES:
        1. Extract the values as float numbers.
        2. If a value is missing or unreadable, set it to null.
        3. Respond ONLY with a valid JSON object matching the requested schema. No conversational text, no markdown block wrappers.
        """
        
        response = ai_model.generate_content(
            contents=[
                {"mime_type": file.content_type, "data": image_bytes},
                prompt
            ],
            generation_config={"response_mime_type": "application/json"}
        )
        
        extracted_data = json.loads(response.text)
        
        hgb = extracted_data.get("hgb")
        mcv = extracted_data.get("mcv")
        mch = extracted_data.get("mch")
        rbc = extracted_data.get("rbc")
        
        # لو الـ Agent معرفش يلقط الـ 4 قيم كاملين بيرد بـ partial عشان الـ Front يحس بيه
        if None in [hgb, mcv, mch, rbc]:
            return {
                "status": "partial",
                "message": "The AI Agent could not confidently extract all critical values. Please review or use manual entry.",
                "detected": extracted_data
            }
            
        # الحسابات التلقائية للمؤشرات (الـ Feature Engineering بتاعك)
        mentzer_index = mcv / rbc
        shine_lal = (mcv ** 2 * mch) / 100
        
        # ترتيب الـ Features بالظبط زي ما الموديل الخبير متعود
        features = [[mcv, mch, hgb, rbc, mentzer_index, shine_lal]]
        
        # التوقع بالموديل الخبير الجديد بتاعك
        prediction_encoded = model.predict(features)[0]
        
        return {
            "status": "success",
            "agent_extraction": "Advanced Vision AI Agent",
            "diagnosis": str(prediction_encoded),  # هيرجع التشخيص فوراً
            "extracted_values": {
                "HGB": hgb, "MCV": mcv, "MCH": mch, "RBC": rbc
            },
            "indicators": {
                "mentzer_index": round(mentzer_index, 2),
                "shine_lal": round(shine_lal, 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")


   
