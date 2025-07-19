# uvicorn api_server:app --reload --port 8000

# api_server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pathlib
import pandas as pd
from google import genai
from google.genai import types
import shutil
import os
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# --- CORS Configuration ---
# Get frontend URL from environment variable
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173") # Default for local development

origins = [
    FRONTEND_URL,
    # Add other frontend URLs if you have multiple, e.g.,
    # "https://your-production-frontend.vercel.app",
    # "https://another-staging-frontend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # You can restrict this to ["GET", "POST", "PUT", "DELETE"] for more security
    allow_headers=["*"], # You can restrict this to specific headers
)
# --- End CORS Configuration ---

MIME_MAP = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.pdf': 'application/pdf'
}
PROMPT = """
You are a data extraction specialist. Extract the following information:
1. Names of people/employees
2. Present days (attendance days)
3. Total days (total working days) or Absent (absent or absent days) or zero (0) if not present
Rules:
- Extract ALL names and their corresponding attendance data
- If data is in table format, extract from each row
- If data is handwritten/unstructured, identify patterns like \"Name X/Y\" or \"Name X Y\"
- Handle variations in handwriting and formatting
- Return ONLY valid JSON format
Required JSON format:
{
    \"extracted_data\": [
        {
            \"name\": \"Person Name\",
            \"present_day\": number,
            \"total_day\": number // or \"absent_day\": number
        }
    ]
}
If you cannot extract certain information, use null for missing values.
"""
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Gemini API Key Setup ---
# Get Gemini API Key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyBw8r_4jXeuxeMiUW2-AfaBJg_wfG5Z0qg"
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=GEMINI_API_KEY)
# ---------------------------

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    print("Received upload request.")
    if not file.filename:
        print("No filename provided.")
        raise HTTPException(status_code=400, detail="No filename provided.")
    print(f"Filename: {file.filename}")
    ext = pathlib.Path(file.filename).suffix.lower()
    print(f"File extension: {ext}")
    if ext not in MIME_MAP and ext != '.xlsx':
        print(f"Unsupported file type: {ext}")
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Use a temporary file path instead of saving to 'uploads' directory
    # Serverless functions often have read-only file systems or limited storage.
    # The 'uploads' directory might not persist between invocations.
    # A better approach for serverless is to process the file directly from memory
    # or use cloud storage like AWS S3 or Google Cloud Storage if persistence is needed.
    # For this simple case, we can process directly from the UploadFile.
    
    try:
        if ext == '.xlsx':
            print("Processing as XLSX...")
            # Read directly from the UploadFile object's file-like object
            df = pd.read_excel(file.file)
            csv_text = df.to_csv(index=False)
            full_prompt = f"{PROMPT}\n\nSpreadsheet data:\n{csv_text}"
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[full_prompt]
            )
        else:
            print(f"Processing as {ext} with MIME type {MIME_MAP[ext]}...")
            mime_type = MIME_MAP[ext]
            # Read bytes directly from the UploadFile object's file-like object
            file_bytes = await file.read() # Use await for async read
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    PROMPT
                ]
            )
        print("Gemini response received.")
        print("Gemini response text:", response.text)
        return JSONResponse(content={"result": response.text})
    except Exception as e:
        print("UPLOAD ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))