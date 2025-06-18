from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import os

# Load paths
model = joblib.load("model/price_predictor.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
dataset_df = pd.read_csv("dataset/lahore_dataset.csv")

# Known locations for nearest area calculation
known_locations = {
    "DHA": (31.4894, 74.3531), "Valencia": (31.3962, 74.2293), "Johar Town": (31.4700, 74.2990),
    "Model Town": (31.4833, 74.3167), "Gulberg": (31.5204, 74.3587), "Shadman": (31.5372, 74.3402),
    "Iqbal Town": (31.5000, 74.3167), "Township": (31.4584, 74.2731), "Cantt": (31.5206, 74.4000),
    "Bahria Town": (31.3672, 74.2395), "Sultanpura": (31.5881, 74.3644), "Askari": (31.4987, 74.4061),
    "Garhi Shahu": (31.5652, 74.3283), "Garden Town": (31.4930, 74.3075), "Wapda Town": (31.4465, 74.2728),
    "Androon Lahore": (31.5833, 74.3667), "Misri Shah": (31.5871, 74.3487)
}

app = FastAPI()

class ServiceData(BaseModel):
    service_type: str
    main_category: str
    latitude: float
    longitude: float
    distance_from_center: float
    time_of_day_minutes: int
    day_of_week: str
    peak_hour: int
    demand_level: str
    weather: str

def find_nearest_area(lat, lon):
    user_loc = (lat, lon)
    nearest = min(known_locations.items(), key=lambda item: geodesic(user_loc, item[1]).km)
    return nearest[0]

@app.post("/predict_price")
async def predict_price(data: ServiceData, request: Request):
    try:
        print("📥 Request:", await request.json())

        # Determine nearest area from lat/lon
        area_name = find_nearest_area(data.latitude, data.longitude)

        # Normalize inputs for encoding
        service_type = data.service_type.strip().lower()
        main_category = data.main_category.strip().lower()
        area_name_clean = area_name.strip().lower()
        day_of_week = data.day_of_week.strip().lower()
        demand_level = data.demand_level.strip().lower()
        weather = data.weather.strip().lower()

        # Encode each categorical value with error handling
        try:
            service_enc = label_encoders["service_type"].transform([service_type])[0]
        except ValueError:
            return {"error": f"Unknown service_type: '{service_type}'"}

        try:
            main_cat_enc = label_encoders["main_category"].transform([main_category])[0]
        except ValueError:
            return {"error": f"Unknown main_category: '{main_category}'"}

        try:
            area_enc = label_encoders["area_name"].transform([area_name_clean])[0]
        except ValueError:
            return {"error": f"Unknown area_name: '{area_name_clean}'"}

        try:
            day_enc = label_encoders["day_of_week"].transform([day_of_week])[0]
        except ValueError:
            return {"error": f"Unknown day_of_week: '{day_of_week}'"}

        try:
            demand_enc = label_encoders["demand_level"].transform([demand_level])[0]
        except ValueError:
            return {"error": f"Unknown demand_level: '{demand_level}'"}

        try:
            weather_enc = label_encoders["weather"].transform([weather])[0]
        except ValueError:
            return {"error": f"Unknown weather: '{weather}'"}

        # Prepare input array for prediction
        input_data = [
            service_enc,
            main_cat_enc,
            area_enc,
            data.distance_from_center,
            data.time_of_day_minutes,
            day_enc,
            data.peak_hour,
            demand_enc,
            weather_enc
        ]

        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        return {
            "predicted_price": round(prediction, 2),
            "location_used": area_name
        }

    except Exception as e:
        print("❌ Error:", str(e))
        return {"error": str(e)}
