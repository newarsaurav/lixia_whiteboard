"""
Single-file whiteboard backend. One HTTP endpoint: POST a prompt, get back a
scene (shapes + connectors + freeform sketch paths + charts) as plain JSON.

This version can also fetch REAL data before drawing — e.g. "chart of the
last 5 days temperature in Kathmandu" actually calls a live weather API
(Open-Meteo, free, no key needed) and charts the real numbers, instead of
the model guessing.

Install:
    pip install fastapi uvicorn google-genai pydantic python-dotenv httpx

Run:
    export GEMINI_API_KEY=your_key_here
    uvicorn server:app --reload --port 8000

Image generation backend (for create_sketch_image):
    IMAGE_BACKEND=gemini (default) — hosted Gemini image model, costs per image.
    IMAGE_BACKEND=local            — your own self-hosted SDXL model on your GPU,
                                      see local_image_gen.py for setup + requirements.
        export IMAGE_BACKEND=local

Call:
    POST http://localhost:8000/draw
    body: {"prompt": "chart the last 5 days temperature in Kathmandu"}
    response: {"shapes": [...], "connectors": [...], "paths": [...], "charts": [...]}
"""

import asyncio
import base64
import os
from typing import Optional, Literal, List

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
IMAGE_MODEL_NAME = "gemini-2.5-flash-image"  # "Nano Banana" — used when IMAGE_BACKEND="gemini"

# "gemini" = hosted Gemini image model (needs GEMINI_API_KEY, costs per image).
# "local"  = your own self-hosted SDXL model on your GPU (see local_image_gen.py) —
#            free, private, and retrainable. Set via: export IMAGE_BACKEND=local
IMAGE_BACKEND = os.environ.get("IMAGE_BACKEND", "gemini").lower()
if IMAGE_BACKEND == "local":
    from local_image_gen import generate_sketch_image_local

# Structured-diagram grid (boxes/diamonds/ellipses + connectors)
GRID_W, GRID_H = 24, 16
GRID_PX = 40

# Freeform sketch canvas (raw SVG path data, in pixels) — must match the
# <svg viewBox="0 0 960 620"> in the frontend.
CANVAS_W, CANVAS_H = 960, 620

MAX_TOOL_ROUNDS = 4  # cap on data-fetch <-> model round trips per request


# ─────────────────────────────────────────────────────────────────────────
# REAL SKETCH IMAGE — actually generates a picture (Nano Banana / Gemini
# image model), styled to look like a hand-drawn pencil sketch on paper,
# instead of the LLM guessing raw SVG coordinates.
# ─────────────────────────────────────────────────────────────────────────

async def generate_sketch_image(subject: str, style_notes: str = "") -> dict:
    if IMAGE_BACKEND == "local":
        # GPU-bound and synchronous — run off the event loop thread.
        return await asyncio.to_thread(generate_sketch_image_local, subject, style_notes)
    return await _generate_sketch_image_gemini(subject, style_notes)


async def _generate_sketch_image_gemini(subject: str, style_notes: str = "") -> dict:
    prompt = (
        f"A simple black-and-white pencil sketch drawn by hand on plain white paper, "
        f"depicting: {subject}. {style_notes or ''} "
        "Loose, expressive graphite pencil linework with visible texture and natural "
        "imperfect lines, like a quick doodle in a sketchbook — NOT digital art, NOT "
        "vector graphics, NOT a photo, NOT clip art. A real hand-drawn pencil sketch look. "
        "Plain white or cream paper background, no color unless explicitly requested, "
        "one clearly centered subject, minimal shading."
    ).strip()
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=IMAGE_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            ),
        )
    except Exception as e:
        return {"error": str(e)}

    for part in getattr(response, "parts", None) or []:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            if isinstance(data, bytes):
                data = base64.b64encode(data).decode("utf-8")
            return {"mime_type": inline.mime_type or "image/png", "data": data}
    return {"error": "Image generation returned no image."}


# ─────────────────────────────────────────────────────────────────────────
# REAL DATA TOOLS — these actually execute (real HTTP calls), unlike the
# drawing tools below which just get parsed into JSON for the frontend.
# ─────────────────────────────────────────────────────────────────────────

async def fetch_weather_history(city: str, days: int = 5) -> dict:
    """Geocode `city` then pull real daily max/min temps for the last `days`
    days (plus today) from Open-Meteo. No API key required."""
    days = max(1, min(int(days or 5), 16))
    async with httpx.AsyncClient(timeout=10) as client:
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
        )
        geo_data = geo.json()
        results = geo_data.get("results") or []
        if not results:
            return {"error": f"Could not find a location named '{city}'."}
        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        resolved_name = ", ".join(
            p for p in [loc.get("name"), loc.get("admin1"), loc.get("country")] if p
        )

        wx = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min",
                "past_days": days,
                "forecast_days": 1,
                "timezone": "auto",
            },
        )
        wx_data = wx.json()
        daily = wx_data.get("daily", {})
        return {
            "location": resolved_name,
            "unit": wx_data.get("daily_units", {}).get("temperature_2m_max", "°C"),
            "dates": daily.get("time", []),
            "temp_max": daily.get("temperature_2m_max", []),
            "temp_min": daily.get("temperature_2m_min", []),
        }


DATA_TOOL_IMPLS = {
    "fetch_weather_history": fetch_weather_history,
}

DATA_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="fetch_weather_history",
                description=(
                    "Fetch REAL recent daily max/min temperature history for a city from a "
                    "live weather API. Call this BEFORE create_chart whenever a request needs "
                    "actual temperature/weather numbers — never invent weather data yourself."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "city": types.Schema(type=types.Type.STRING, description="City name, e.g. 'Kathmandu'"),
                        "days": types.Schema(type=types.Type.INTEGER, description="How many past days, default 5"),
                    },
                    required=["city"],
                ),
            ),
        ]
    )
]


# ─────────────────────────────────────────────────────────────────────────
# DRAWING TOOLS — parsed into JSON, never executed server-side.
# ─────────────────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = f"""You are a whiteboard drawing assistant with a few complementary
ways of drawing, plus real data/image tools. For every request, decide what fits, then call
the matching tool(s). Always end by calling at least one DRAWING tool — never respond with
only text.

DATA TOOL (call first, only when needed):
- fetch_weather_history — call this BEFORE drawing whenever the request needs real recent
  temperature/weather numbers (e.g. "chart the last 5 days temp of Kathmandu"). Never invent
  weather numbers yourself; always fetch them. Once you get the result, use its real dates
  and temperatures as the x_labels/values for create_chart.

DRAWING TOOLS (call one or more of these to actually put something on the board):
1. create_shape + create_connector — for a process, flowchart, org chart, architecture
   diagram, mind map, or anything made of discrete labeled boxes/circles/diamonds linked by
   arrows. Think in a {GRID_W}x{GRID_H} grid (grid_x 0-{GRID_W}, grid_y 0-{GRID_H}), leave at
   least 1 grid unit of gap between shapes, keep everything within bounds.

2. create_sketch_image — the DEFAULT choice for any object, animal, person, place, or scene
   (e.g. "draw a cat", "sketch a mountain", "draw a house"). This generates a REAL picture —
   an actual hand-drawn-style pencil sketch on paper — not guessed vector coordinates, so it
   looks genuinely like something a person drew. Use this whenever someone asks to "draw" or
   "sketch" a real-world thing and no diagram/chart is implied.

3. create_sketch_path — a lightweight fallback ONLY for tiny simple icons/glyphs embedded
   inside a diagram box (e.g. a small icon next to a label), or if the request explicitly
   asks for plain line-art/outline rather than a real sketch. Emit one or more SVG path `d`
   strings (M/L/C/Q/A/Z) in a {CANVAS_W}x{CANVAS_H} pixel canvas, origin top-left.

4. create_chart — for any request asking to chart/plot/graph numeric data over categories or
   time (temperature, prices, counts, comparisons). Use "bar" for comparisons across discrete
   categories, "line" for a trend over time. Use REAL numbers — from fetch_weather_history when
   available, or well-known figures for other topics; do not fabricate precise statistics you
   are not confident in — if you cannot get real numbers for a non-weather data request, say so
   in a short text reply instead of calling create_chart with made-up figures.

You may combine tools (e.g. a labeled diagram with a create_sketch_image icon inside a box)."""

DRAWING_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="create_shape",
                description="Create a structured diagram shape (box/circle/diamond) on the grid.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "id": types.Schema(type=types.Type.STRING, description="Short unique id you choose, e.g. 's1'"),
                        "shape_type": types.Schema(type=types.Type.STRING, enum=["rect", "ellipse", "diamond"]),
                        "grid_x": types.Schema(type=types.Type.INTEGER),
                        "grid_y": types.Schema(type=types.Type.INTEGER),
                        "grid_width": types.Schema(type=types.Type.INTEGER),
                        "grid_height": types.Schema(type=types.Type.INTEGER),
                        "fill": types.Schema(type=types.Type.STRING, description="Hex color"),
                        "label": types.Schema(type=types.Type.STRING),
                    },
                    required=["id", "shape_type", "grid_x", "grid_y", "grid_width", "grid_height"],
                ),
            ),
            types.FunctionDeclaration(
                name="create_connector",
                description="Connect two shapes already created by their ids.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "from_shape_id": types.Schema(type=types.Type.STRING),
                        "to_shape_id": types.Schema(type=types.Type.STRING),
                        "label": types.Schema(type=types.Type.STRING),
                    },
                    required=["from_shape_id", "to_shape_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="create_sketch_image",
                description=(
                    "Generate a REAL hand-drawn-style pencil sketch image (an actual picture, "
                    "not vector shapes) of an object, animal, person, place, or scene. This is "
                    "the default tool whenever someone asks to draw/sketch a real-world thing."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "id": types.Schema(type=types.Type.STRING, description="Short unique id you choose, e.g. 'img1'"),
                        "subject": types.Schema(type=types.Type.STRING, description="What to draw, described clearly, e.g. 'a sleeping tabby cat curled up'"),
                        "style_notes": types.Schema(type=types.Type.STRING, description="Optional extra style guidance, e.g. 'side profile', 'in a garden'"),
                        "caption": types.Schema(type=types.Type.STRING, description="Optional short caption to show under the sketch"),
                    },
                    required=["id", "subject"],
                ),
            ),
            types.FunctionDeclaration(
                name="create_sketch_path",
                description=(
                    "Draw one freeform stroke/region of a sketch (object, animal, scene, icon) "
                    "as a raw SVG path 'd' string on the pixel canvas. Call this multiple times "
                    "for a multi-stroke drawing."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "id": types.Schema(type=types.Type.STRING, description="Short unique id you choose, e.g. 'p1'"),
                        "d": types.Schema(type=types.Type.STRING, description="SVG path data, e.g. 'M100,100 L200,100 ...'"),
                        "stroke": types.Schema(type=types.Type.STRING, description="Hex stroke color, e.g. '#4f8ef7'"),
                        "fill": types.Schema(type=types.Type.STRING, description="Hex fill color, or 'none' for outline-only"),
                        "stroke_width": types.Schema(type=types.Type.NUMBER, description="Stroke width in px, default 2.5"),
                        "label": types.Schema(type=types.Type.STRING, description="Optional short caption"),
                    },
                    required=["id", "d"],
                ),
            ),
            types.FunctionDeclaration(
                name="create_chart",
                description="Draw a bar or line chart of real numeric data on the board.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "id": types.Schema(type=types.Type.STRING, description="Short unique id, e.g. 'c1'"),
                        "chart_type": types.Schema(type=types.Type.STRING, enum=["bar", "line"]),
                        "title": types.Schema(type=types.Type.STRING),
                        "y_unit": types.Schema(type=types.Type.STRING, description="e.g. '°C', '$', 'count'"),
                        "x_labels": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.STRING),
                            description="Category/date labels along the x-axis",
                        ),
                        "series": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(
                                type=types.Type.OBJECT,
                                properties={
                                    "name": types.Schema(type=types.Type.STRING),
                                    "color": types.Schema(type=types.Type.STRING, description="Hex color"),
                                    "values": types.Schema(
                                        type=types.Type.ARRAY,
                                        items=types.Schema(type=types.Type.NUMBER),
                                    ),
                                },
                                required=["name", "values"],
                            ),
                            description="One or more data series, each with one value per x_label",
                        ),
                    },
                    required=["id", "chart_type", "x_labels", "series"],
                ),
            ),
        ]
    )
]

ALL_TOOLS = DATA_TOOLS + DRAWING_TOOLS


class ShapeArgs(BaseModel):
    id: str
    shape_type: Literal["rect", "ellipse", "diamond"]
    grid_x: int = Field(ge=0, le=GRID_W)
    grid_y: int = Field(ge=0, le=GRID_H)
    grid_width: int = Field(ge=1, le=GRID_W)
    grid_height: int = Field(ge=1, le=GRID_H)
    fill: Optional[str] = "#4f8ef7"
    label: Optional[str] = None


class ConnectorArgs(BaseModel):
    from_shape_id: str
    to_shape_id: str
    label: Optional[str] = None


class SketchImageArgs(BaseModel):
    id: str
    subject: str
    style_notes: Optional[str] = None
    caption: Optional[str] = None


class SketchPathArgs(BaseModel):
    id: str
    d: str
    stroke: Optional[str] = "#e6e8ec"
    fill: Optional[str] = "none"
    stroke_width: Optional[float] = 2.5
    label: Optional[str] = None


class ChartSeriesArgs(BaseModel):
    name: str
    color: Optional[str] = None
    values: List[float]


class ChartArgs(BaseModel):
    id: str
    chart_type: Literal["bar", "line"]
    title: Optional[str] = None
    y_unit: Optional[str] = None
    x_labels: List[str]
    series: List[ChartSeriesArgs]


class DrawRequest(BaseModel):
    prompt: str


@app.post("/draw")
async def draw(req: DrawRequest):
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, tools=ALL_TOOLS)

    contents: List[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=req.prompt)])
    ]

    known_ids = set()
    shapes, connectors, paths, charts, images = [], [], [], [], []
    assistant_text_parts: List[str] = []

    for _round in range(MAX_TOOL_ROUNDS):
        response = client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
        cand_parts = response.candidates[0].content.parts if response.candidates else []
        if not cand_parts:
            break

        contents.append(types.Content(role="model", parts=cand_parts))

        data_calls = [p.function_call for p in cand_parts if p.function_call and p.function_call.name in DATA_TOOL_IMPLS]
        drawing_calls = [p.function_call for p in cand_parts if p.function_call and p.function_call.name not in DATA_TOOL_IMPLS]

        for p in cand_parts:
            if p.text:
                assistant_text_parts.append(p.text)

        for fc in drawing_calls:
            try:
                if fc.name == "create_shape":
                    args = ShapeArgs(**dict(fc.args or {}))
                    known_ids.add(args.id)
                    shapes.append(args.model_dump())
                elif fc.name == "create_connector":
                    args = ConnectorArgs(**dict(fc.args or {}))
                    if args.from_shape_id in known_ids and args.to_shape_id in known_ids:
                        connectors.append(args.model_dump())
                elif fc.name == "create_sketch_image":
                    args = SketchImageArgs(**dict(fc.args or {}))
                    img_result = await generate_sketch_image(args.subject, args.style_notes or "")
                    if "error" not in img_result:
                        images.append({
                            "id": args.id,
                            "subject": args.subject,
                            "caption": args.caption,
                            "mime_type": img_result["mime_type"],
                            "data": img_result["data"],
                        })
                    else:
                        assistant_text_parts.append(
                            f" (couldn't generate a sketch of '{args.subject}': {img_result['error']}) "
                        )
                elif fc.name == "create_sketch_path":
                    args = SketchPathArgs(**dict(fc.args or {}))
                    paths.append(args.model_dump())
                elif fc.name == "create_chart":
                    args = ChartArgs(**dict(fc.args or {}))
                    charts.append(args.model_dump())
            except Exception:
                continue  # skip malformed tool calls rather than failing the whole request

        if not data_calls:
            break  # nothing left to fetch — this round's drawing calls are the final answer

        # Execute real data tool calls and feed results back for another round.
        response_parts = []
        for fc in data_calls:
            impl = DATA_TOOL_IMPLS[fc.name]
            try:
                result = await impl(**dict(fc.args or {}))
            except Exception as e:
                result = {"error": str(e)}
            response_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=response_parts))
        # loop again so the model can call create_chart with the real data

    return {
        "shapes": shapes,
        "connectors": connectors,
        "paths": paths,
        "charts": charts,
        "images": images,
        "assistant_text": "".join(assistant_text_parts),
    }