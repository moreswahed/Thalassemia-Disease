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

# 3. (OCR)
@app.post("/predict/image")
async def predict_from_image(file: UploadFile = File(...)):
    try:
        # 1. قراءة محتوى الملف المرفوع
        contents = await file.read()
        
        
        
        # تحويل البايتات لصور يفهمها OpenCV
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # تحويل لرمادي وتكبير الصورة لتحسين الدقة
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        # --- 3. نبعت الصورة "المعالجـة" لـ Tesseract ---

        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        raw_text = pytesseract.image_to_string(thresh, config=custom_config)
        
        # --- من أول هنا ابدأ الاستبدال ---
        
       # 1. عرف الدالة الأول (تأكد إنها بتبدأ من أول السطر جوه الدالة الكبيرة)
       # 1. دالة الاستخراج (زي ما هي مع تحسين بسيط)
        def extract_value(short_name, text):
            patterns = {
                "HGB": r"H[GgB60b]{2}.*?(\d+\.?\d*)",
                "MCV": r"MC[Vv0uU7].*?(\d+\.?\d*)",
                "MCH": r"MCH.*?(\d+\.?\d*)",
                "RBC": r"RBC.*?(\d+\.?\d*)"
            }
            match = re.search(patterns[short_name], text, re.IGNORECASE | re.DOTALL)
            return float(match.group(1)) if match else None

        # 2. فلتر التصحيح الذكي (عشان لو الـ OCR نسي العلامة العشرية)
        def fix_decimal(value, test_type):
            if value is None: return None
            # الهيموجلوبين (HGB) والـ MCH والـ RBC غالباً بيكونو رقم وجنبه منزلة عشرية واحدة
            if test_type == "HGB" and value > 25: return value / 10  # لو قرأ 134 يخليها 13.4
            if test_type == "RBC" and value > 10: return value / 10   # لو قرأ 46 يخليها 4.6
            if test_type == "MCH" and value > 60: return value / 10   # لو قرأ 267 يخليها 26.7
            return value

       # 1. سحب القيم وتمريرها على الفلتر فوراً (عشان نصلح العلامة العشرية)
        hgb = fix_decimal(extract_value("HGB", raw_text), "HGB")
        mcv = extract_value("MCV", raw_text) 
        mch = fix_decimal(extract_value("MCH", raw_text), "MCH")
        rbc = fix_decimal(extract_value("RBC", raw_text), "RBC")

        # 2. خطة طوارئ للهيموجلوبين (لو لسه null يدور بكلمة Hemoglobin كاملة)
        if hgb is None:
            hgb_alt = re.search(r"Hemoglobin.*?(\d+\.?\d*)", raw_text, re.IGNORECASE | re.DOTALL)
            if hgb_alt:
                hgb = fix_decimal(float(hgb_alt.group(1)), "HGB")

        # 3. خطة طوارئ للـ MCV (لو لسه null يدور على رقم منطقي في النص)
        if mcv is None:
            potential_numbers = re.findall(r"(\d{2,3}\.\d?)", raw_text)
            mcv = next((float(n) for n in potential_numbers if 50 <= float(n) <= 115), None)

        # 1. التأكد من وجود الأرقام الأساسية
        if all(v is not None for v in [hgb, mcv, mch, rbc]):
            
            # 2. حساب المعادلات اللي الموديل مستنيها (مهم جداً!)
            mentzer_index = mcv / rbc if rbc != 0 else 0
            shine_lal = (mcv * mcv * mch) / 100
            
            # 3. ترتيب الـ Features "بالظبط" زي ما الموديل اتدرب في الـ Notebook
            # الترتيب: [MCV, MCH, HGB, RBC, Mentzer, Shine]
            features = np.array([[mcv, mch, hgb, rbc, mentzer_index, shine_lal]])
            
            # 4. التوقع
            prediction_numeric = model.predict(features)[0]
            diagnosis_name = label_encoder.inverse_transform([prediction_numeric])[0]

            return {
                "status": "success",
                "diagnosis": diagnosis_name,
                "extracted_values": {
                    "HGB": hgb, "MCV": mcv, "MCH": mch, "RBC": rbc,
                    "Mentzer_Index": round(mentzer_index, 2),
                    "Shine_Lal": round(shine_lal, 2)
                }
            }
        else:
            return {
                "status": "error",
                "message": "Values missing. Please check the terminal for raw text.",
                "detected": {"HGB": hgb, "MCV": mcv, "MCH": mch, "RBC": rbc},
                "raw_debug": raw_text[:500] # هيظهرلك أول 500 حرف من اللي قراهم عشان تفهم التايه فين
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) # لازم يكون فيه مسافة (Tab) هنا


   