"""
Single-file, text-only whiteboard backend. No WebSocket, no Live API -
just one HTTP endpoint: POST a prompt, get back a scene (shapes + connectors)
as plain JSON.

Install:
    pip install fastapi uvicorn google-genai pydantic

Run:
    export GEMINI_API_KEY=your_key_here
    uvicorn whiteboard_server:app --reload --port 8000

Call:
    POST http://localhost:8000/draw
    body: {"prompt": "draw a 3-step approval flow"}
    response: {"shapes": [...], "connectors": [...]}
"""

import os
import uuid
from typing import Optional, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")   
GRID_W, GRID_H = 24, 16

SYSTEM_INSTRUCTION = """You are a whiteboard drawing assistant. Given a request,
call create_shape for each shape needed and create_connector to link related
shapes by id. Think in a 24x16 grid (grid_x 0-24, grid_y 0-16), leave at least
1 grid unit of gap between shapes, and keep everything within bounds."""

DRAWING_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="create_shape",
                description="Create a shape on the whiteboard.",
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
        ]
    )
]


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


class DrawRequest(BaseModel):
    prompt: str


@app.post("/draw")
def draw(req: DrawRequest):
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        # model="gemini-2.0-flash",
        model="gemini-3.5-flash",
        # model = "gemini-3-flash-preview",
        contents=req.prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=DRAWING_TOOLS,
        ),
    )

    parts = response.candidates[0].content.parts if response.candidates else []
    known_ids = set()
    shapes = []
    connectors = []

    for part in parts:
        fc = part.function_call
        if not fc:
            continue
        try:
            if fc.name == "create_shape":
                args = ShapeArgs(**dict(fc.args or {}))
                known_ids.add(args.id)
                shapes.append(args.model_dump())
            elif fc.name == "create_connector":
                args = ConnectorArgs(**dict(fc.args or {}))
                if args.from_shape_id in known_ids and args.to_shape_id in known_ids:
                    connectors.append(args.model_dump())
        except Exception:
            continue  # skip malformed tool calls rather than failing the whole request

    text = "".join(p.text or "" for p in parts if p.text)
    return {"shapes": shapes, "connectors": connectors, "assistant_text": text}