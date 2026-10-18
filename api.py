import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from pydantic import BaseModel

from measurement_engine import MeasurementEngine
from size_predictor import predict_shirt_size, get_brands, get_fits

app = FastAPI(
    title="AI Shirt Size Detector API",
    description="AI-powered body measurement and shirt size recommendation system",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

measurement_engine = MeasurementEngine()

# Serve the frontend folder at /frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
def home():
    return RedirectResponse(url="/frontend/index.html")

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return {
            "success": False,
            "message": "Could not decode image."
        }

    measurements = measurement_engine.process_frame(frame)

    if measurements is None:
        return {
            "success": False,
            "message": "No person detected."
        }

    recommended_size = predict_shirt_size(
        shoulder_cm=measurements["shoulder_cm"],
        chest_width_cm=measurements["chest_width_cm"],
        brand="generic",
        fit="regular_fit"
    )

    return {
        "success": True,
        "measurements": measurements,
        "recommendation": {
            "brand": "generic",
            "fit": "regular_fit",
            "size": recommended_size
        }
    }


# ============================================================
# STAGE 4: SIZE RECOMMENDATION ENDPOINTS
# ============================================================

class SizeRequest(BaseModel):
    shoulder_cm: float
    chest_width_cm: float
    brand: str = "generic"
    fit: str = "regular_fit"


@app.get("/api/brands")
def get_available_brands():
    """Returns available brands and fits from size_charts.json."""
    brands = get_brands()
    brand_list = []
    for b in brands:
        fits = get_fits(b)
        display_name = "Generic" if b == "generic" else "Demo Brand (Test Data)"
        fit_items = [
            {"id": f, "name": f.replace("_", " ").title()}
            for f in fits
        ]
        brand_list.append({
            "id": b,
            "name": display_name,
            "fits": fit_items
        })
    return {"brands": brand_list}


@app.post("/api/recommend-size")
def recommend_size(req: SizeRequest):
    """Predicts shirt size from shoulder and chest measurements."""
    try:
        size = predict_shirt_size(
            shoulder_cm=req.shoulder_cm,
            chest_width_cm=req.chest_width_cm,
            brand=req.brand,
            fit=req.fit
        )
        display_brand = "Generic" if req.brand == "generic" else "Demo Brand (Test Data)"
        display_fit = req.fit.replace("_", " ").title()
        return {
            "success": True,
            "size": size,
            "brand": req.brand,
            "brand_name": display_brand,
            "fit": req.fit,
            "fit_name": display_fit,
            "shoulder_cm": req.shoulder_cm,
            "chest_width_cm": req.chest_width_cm,
            "explanation": "Based on your estimated shoulder and chest measurements."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.on_event("shutdown")
def shutdown():
    measurement_engine.close()
