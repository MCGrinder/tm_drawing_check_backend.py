import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI(title="TM Drawing Checker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = os.environ.get("TM_AI_MODEL", "gpt-5")
MAX_FILE_SIZE_MB = int(os.environ.get("TM_MAX_FILE_SIZE_MB", "15"))
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}

OUTPUT_SCHEMA: dict[str, Any] = {
    "name": "tm_red_book_check",
    "type": "json_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {
                "type": "string",
                "enum": ["Likely Compliant", "Needs Review", "Likely Non-Compliant"],
            },
            "summary": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pass", "review", "fail"],
                        },
                        "why": {"type": "string"},
                        "reference": {"type": "string"},
                    },
                    "required": ["id", "title", "status", "why", "reference"],
                },
            },
            "agent_steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "outcome": {"type": "string"},
                    },
                    "required": ["title", "outcome"],
                },
            },
        },
        "required": ["status", "summary", "findings", "agent_steps"],
    },
}

SYSTEM_PROMPT = """
You are a UK temporary traffic management drawing reviewer.

Review the uploaded TM drawing against UK Red Book style principles at a high level.
Do not invent exact clause numbers unless they are clearly visible in the supplied material.
Focus on practical compliance observations such as:
- lane closures and tapers
- advance warning signing
- pedestrian route continuity and safety
- crossings and junction visibility
- cycle impact if visible
- road user separation and obvious layout risks

Important rules:
- If the drawing is too unclear to decide, use 'Needs Review'.
- Do not call something non-compliant unless the issue is reasonably visible from the drawing.
- Be cautious and explain uncertainty.
- References should be short principle-style references.
- Return only the structured output.
""".strip()


@app.get("/api/tm/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/tm/check-drawing")
async def check_drawing(
    drawing: UploadFile = File(...),
    standard: str = Form("uk_red_book"),
    agent_mode: str = Form("false"),
    client_name: str = Form("mobile_app", alias="client"),
) -> dict[str, Any]:
    if standard != "uk_red_book":
        raise HTTPException(status_code=400, detail="Only uk_red_book is supported right now.")

    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing on the server.")

    filename = drawing.filename or "upload"
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, PNG, JPG, JPEG, or WEBP.")

    file_bytes = await drawing.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max size is {MAX_FILE_SIZE_MB} MB.")

    try:
        response = openai_client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=_build_input(
                filename=filename,
                file_bytes=file_bytes,
                agent_mode=agent_mode,
                client_name=client_name,
            ),
            text={"format": OUTPUT_SCHEMA},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}") from exc

    parsed = _extract_structured_output(response)
    if not parsed:
        raise HTTPException(status_code=502, detail="AI response did not include valid structured output.")

    return parsed


def _build_input(*, filename: str, file_bytes: bytes, agent_mode: str, client_name: str) -> list[dict[str, Any]]:
    extension = Path(filename).suffix.lower()
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    encoded = base64.b64encode(file_bytes).decode("utf-8")

    request_text = (
        "Review this temporary traffic management drawing and return a UK Red Book style first-pass compliance view. "
        f"Agent mode: {agent_mode}. Client: {client_name}. Be cautious. Use 'Needs Review' where the drawing is unclear."
    )

    content: list[dict[str, Any]] = [{"type": "input_text", "text": request_text}]

    if extension == ".pdf":
        content.append(
            {
                "type": "input_file",
                "filename": filename,
                "file_data": f"data:{mime_type};base64,{encoded}",
            }
        )
    else:
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported image format. Use PNG, JPG, or WEBP.")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
                "detail": "high",
            }
        )

    return [{"role": "user", "content": content}]


def _extract_structured_output(response: Any) -> dict[str, Any] | None:
    output_parsed = getattr(response, "output_parsed", None)
    if output_parsed:
        if isinstance(output_parsed, dict):
            return output_parsed
        if hasattr(output_parsed, "model_dump"):
            return output_parsed.model_dump()

    output_text = getattr(response, "output_text", None)
    if output_text:
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            pass

    output = getattr(response, "output", None)
    if not output:
        return None

    for item in output:
        for content in getattr(item, "content", []) or []:
            parsed = getattr(content, "parsed", None)
            if parsed:
                if isinstance(parsed, dict):
                    return parsed
                if hasattr(parsed, "model_dump"):
                    return parsed.model_dump()

            text = getattr(content, "text", None)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue

    return None


def _self_test() -> None:
    png_input = _build_input(filename="test.png", file_bytes=b"fake", agent_mode="true", client_name="mobile_app")
    assert png_input[0]["content"][1]["type"] == "input_image"
    assert png_input[0]["content"][1]["image_url"].startswith("data:image/png;base64,")

    jpg_input = _build_input(filename="test.jpg", file_bytes=b"fake", agent_mode="true", client_name="mobile_app")
    assert jpg_input[0]["content"][1]["type"] == "input_image"
    assert jpg_input[0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")

    pdf_input = _build_input(filename="test.pdf", file_bytes=b"fake", agent_mode="true", client_name="mobile_app")
    assert pdf_input[0]["content"][1]["type"] == "input_file"
    assert pdf_input[0]["content"][1]["file_data"].startswith("data:application/pdf;base64,")


_self_test()

# Run with:
# uvicorn tm_drawing_check_backend:app --host 0.0.0.0 --port 10000
