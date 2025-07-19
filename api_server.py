# uvicorn api_server:app --reload --port 8000

import json
import pathlib
import pandas as pd
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS Configuration
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
origins = [FRONTEND_URL]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Gemini API Key Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)

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

    try:
        if ext == '.xlsx':
            print("Processing as XLSX...")
            df = pd.read_excel(file.file)
            csv_text = df.to_csv(index=False)
            full_prompt = f"{PROMPT}\n\nSpreadsheet data:\n{csv_text}"
            model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
            response = model.generate_content(full_prompt)
        else:
            print(f"Processing as {ext} with MIME type {MIME_MAP[ext]}...")
            file_bytes = await file.read()
            model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
            response = model.generate_content([
                {"mime_type": MIME_MAP[ext], "data": file_bytes},  # Updated format for file bytes
                PROMPT
            ])
        print("Gemini response received.")
        try:
            json_result = json.loads(response.text)
            print("Gemini response parsed as JSON:", json_result)
            return JSONResponse(content={"result": json_result})
        except json.JSONDecodeError:
            print("Gemini response is not valid JSON. Returning raw text.")
            return JSONResponse(content={"result": response.text})

    except Exception as e:
        print("UPLOAD ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
