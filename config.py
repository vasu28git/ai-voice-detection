import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "63cb790554517a0936916c09a1224f92da0e11577d9710aa")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

SAMPLE_RATE = 16000
SKIP_SECONDS = 3
WINDOW_SECONDS = 12
SKIP_SAMPLES = SAMPLE_RATE * SKIP_SECONDS 
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS 

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]


N_MFCC = 40  
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
