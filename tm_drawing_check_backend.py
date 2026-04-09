from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import base64
import mimetypes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.get("/api/tm/health")
def health():
    return {"status": "ok"}

@app.post("/api/tm/check-drawing")
async def check_drawing(
    drawing: UploadFile = File(...),
    standard: str = Form("uk_red_book"),
    agent_mode: str = Form("false"),
    client: str = Form("mobile_app")
):
    try:
        file_bytes = await drawing.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        mime_type = drawing.content_type or "application/octet-stream"
        encoded = base64.b64encode(file_bytes).decode()

        # 🔥 FIX: handle PDF vs IMAGE properly
        if "pdf" in mime_type:
            content = [
                {"type": "input_text", "text": "Check this UK TM drawing for Red Book compliance."},
                {
                    "type": "input_file",
                    "filename": drawing.filename,
                    "file_data": f"data:{mime_type};base64,{encoded}"
                }
            ]
        else:
            content = [
                {"type": "input_text", "text": "Check this UK TM drawing for Red Book compliance."},
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{encoded}"
                }
            ]

        response = client.responses.create(
            model="gpt-5",
            input=[{"role": "user", "content": content}]
        )

        return {
            "status": "Needs Review",
            "summary": response.output_text,
            "findings": [],
            "agent_steps": [
                {"title": "AI reviewed drawing", "outcome": "Basic analysis complete"}
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
