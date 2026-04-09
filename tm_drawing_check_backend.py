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
            "confidence": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
            },
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
        "required": ["status", "summary", "confidence", "findings", "agent_steps"],
    },
}

SYSTEM_PROMPT = """
You are a careful UK temporary traffic management drawing reviewer.

Your job is to review uploaded TM drawings against Safety at Street Works and Road Works (the Red Book) at a practical first-pass level for modern UK TM plans.

Your output must feel like a competent streetworks / permit reviewer, not a vague AI.

Core review principle:
- DO NOT require spacing measurements, taper lengths, sign distances, cone spacings, or exact dimensions to be shown on the drawing.
- Modern TM plans often do not show those measurements.
- Base your decision on visible layout, positioning, route continuity, road user protection, and obvious safety logic only.

Critical rules:
- Do not invent exact clause numbers unless they are visible in the source material.
- Use only what is reasonably visible from the drawing/photo and the supplied context.
- If something is unclear, mark it as review, not fail.
- Only use fail where there is a visible likely non-compliance or a clearly missing safety arrangement.
- Do not fail a plan merely because exact distances or spacing are not shown.
- Do not fail a plan merely because measurements are absent.
- References must be short principle-style references, for example:
  'Safety at Street Works and Road Works - pedestrian safety principles'
  'Safety at Street Works and Road Works - signing and taper principles'
  'Safety at Street Works and Road Works - site layout and visibility principles'

Use the supplied inspection context:
- speed limit
- road type
- works type
- nearby risks
- reviewer note

Assess the drawing using these practical checks where relevant and visible:
1. Is the overall TM layout understandable and coherent?
2. Is there an obvious transition into the works area where traffic is diverted or narrowed?
3. Is there obvious warning/signing provision shown in principle?
4. Is working space or separation from live traffic visibly provided?
5. Is there a clear pedestrian arrangement where pedestrians are affected?
6. Are obvious conflicts with crossings, junctions, bus stops, schools, or vulnerable users visible?
7. If temporary signals or stop/go appear relevant, is the control arrangement understandable in principle?
8. Does anything look obviously unsafe, missing, obstructive, or contradictory?

Decision guidance:
- Likely Non-Compliant:
  use this where one or more important visible safety issues are fail.
  Examples:
  - pedestrians appear left with no obvious route where one is clearly needed
  - the layout appears to push traffic into conflict with no visible transition/protection
  - an obvious crossing/junction conflict is visible and unmanaged
  - the arrangement is clearly confusing or unsafe

- Needs Review:
  use this where the drawing may be acceptable, but the evidence is unclear, partial, low quality, cropped, or ambiguous.
  Also use this where a reviewer would reasonably want more detail before approving.

- Likely Compliant:
  use this where the visible layout appears sensible, coherent, and no material concern is obvious.

Finding balance:
- Do not be over-strict.
- Do not assume failure from missing measurements.
- Be practical.
- Think like a reviewer looking at a real submitted TM plan, not a textbook diagram.

Output requirements:
- Return 4 to 7 findings unless the drawing is unreadable.
- Include a mix of pass/review/fail as appropriate.
- Findings must be specific and useful.
- Include agent_steps summarising what you checked.
- Confidence should reflect image clarity and certainty of judgement.
- Return only the structured output.
""".strip()


@app.get("/api/tm/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/tm/check-drawing")
async def check_drawing(
    drawing: UploadFile = File(...),
    standard: str = Form("uk_red_book"),
    agent_mode: str = Form("true"),
    client_name: str = Form("mobile_app", alias="client"),
    speed_limit: str = Form("30"),
    road_type: str = Form("urban-single-carriageway"),
    works_type: str = Form("lane-closure"),
    nearby_risks: str = Form(""),
    reviewer_note: str = Form(""),
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
                speed_limit=speed_limit,
                road_type=road_type,
                works_type=works_type,
                nearby_risks=nearby_risks,
                reviewer_note=reviewer_note,
            ),
            text={"format": OUTPUT_SCHEMA},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}") from exc

    parsed = _extract_structured_output(response)
    if not parsed:
        raise HTTPException(status_code=502, detail="AI response did not include valid structured output.")

    return parsed


def _build_input(
    *,
    filename: str,
    file_bytes: bytes,
    agent_mode: str,
    client_name: str,
    speed_limit: str,
    road_type: str,
    works_type: str,
    nearby_risks: str,
    reviewer_note: str,
) -> list[dict[str, Any]]:
    extension = Path(filename).suffix.lower()
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    encoded = base64.b64encode(file_bytes).decode("utf-8")

    context_text = (
        "Inspection context:\n"
        f"- Agent mode: {agent_mode}\n"
        f"- Client: {client_name}\n"
        f"- Speed limit: {speed_limit} mph\n"
        f"- Road type: {road_type}\n"
        f"- Works type: {works_type}\n"
        f"- Nearby risks: {nearby_risks or 'None provided'}\n"
        f"- Reviewer note: {reviewer_note or 'None provided'}\n\n"
        "Review this temporary traffic management drawing and provide a practical Red Book-style compliance result. "
        "Do not require spacing or distance measurements. Judge from visible layout and safety principles."
    )

    content: list[dict[str, Any]] = [{"type": "input_text", "text": context_text}]

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
    png_input = _build_input(
        filename="test.png",
        file_bytes=b"fake",
        agent_mode="true",
        client_name="mobile_app",
        speed_limit="30",
        road_type="urban-single-carriageway",
        works_type="lane-closure",
        nearby_risks="zebra crossing",
        reviewer_note="watch pedestrian route",
    )
    assert png_input[0]["content"][1]["type"] == "input_image"
    assert png_input[0]["content"][1]["image_url"].startswith("data:image/png;base64,")

    jpg_input = _build_input(
        filename="test.jpg",
        file_bytes=b"fake",
        agent_mode="true",
        client_name="mobile_app",
        speed_limit="40",
        road_type="rural-road",
        works_type="stop-go",
        nearby_risks="tight bend",
        reviewer_note="high speed approach",
    )
    assert jpg_input[0]["content"][1]["type"] == "input_image"
    assert jpg_input[0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")

    pdf_input = _build_input(
        filename="test.pdf",
        file_bytes=b"fake",
        agent_mode="true",
        client_name="mobile_app",
        speed_limit="20",
        road_type="residential-street",
        works_type="footway-closure",
        nearby_risks="school crossing",
        reviewer_note="check vulnerable users",
    )
    assert pdf_input[0]["content"][1]["type"] == "input_file"
    assert pdf_input[0]["content"][1]["file_data"].startswith("data:application/pdf;base64,")


_self_test()

# Run with:
# uvicorn tm_drawing_check_backend:app --host 0.0.0.0 --port 10000
