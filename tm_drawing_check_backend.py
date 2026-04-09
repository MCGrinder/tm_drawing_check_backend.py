from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import base64

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
    client_type: str = Form("web")
):
    file_bytes = await drawing.read()
    encoded = base64.b64encode(file_bytes).decode()

    response = client.responses.create(
        model="gpt-5",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Check this UK traffic management drawing for Red Book compliance."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}
            ]
        }]
    )

    return {
        "status": "Needs Review",
        "summary": response.output_text,
        "findings": [],
        "agent_steps": [
            {"title": "AI reviewed drawing", "outcome": "Basic analysis complete"}
        ]
    }
