"""
Gemini Live whiteboard backend: function-calling drawing engine.

This handles the Python/backend side of the architecture described in the
report - it defines the drawing tools, runs a Gemini Live session, validates
incoming tool calls, maintains canvas state, and forwards validated shape
events to the frontend (e.g. over a WebSocket) for react-konva/tldraw to render.

Install:
    pip install google-genai pydantic websockets
"""

import asyncio
import json
import uuid
from enum import Enum
from typing import Optional, Literal

from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# 1. Tool schemas (sent to Gemini so it knows what it can call)
# ---------------------------------------------------------------------------

DRAWING_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="create_shape",
                description="Create a shape on the whiteboard canvas.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "shape_type": types.Schema(
                            type=types.Type.STRING,
                            enum=["rect", "ellipse", "diamond", "line", "arrow"],
                        ),
                        "grid_x": types.Schema(type=types.Type.INTEGER, description="X position in grid units (0-24)"),
                        "grid_y": types.Schema(type=types.Type.INTEGER, description="Y position in grid units (0-16)"),
                        "grid_width": types.Schema(type=types.Type.INTEGER),
                        "grid_height": types.Schema(type=types.Type.INTEGER),
                        "fill": types.Schema(type=types.Type.STRING, description="Hex color, e.g. #4F8EF7"),
                        "stroke": types.Schema(type=types.Type.STRING),
                        "label": types.Schema(type=types.Type.STRING, description="Optional text label inside the shape"),
                    },
                    required=["shape_type", "grid_x", "grid_y", "grid_width", "grid_height"],
                ),
            ),
            types.FunctionDeclaration(
                name="create_text",
                description="Place standalone text on the canvas (not inside a shape).",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "grid_x": types.Schema(type=types.Type.INTEGER),
                        "grid_y": types.Schema(type=types.Type.INTEGER),
                        "text": types.Schema(type=types.Type.STRING),
                        "font_size": types.Schema(type=types.Type.INTEGER),
                    },
                    required=["grid_x", "grid_y", "text"],
                ),
            ),
            types.FunctionDeclaration(
                name="create_connector",
                description="Draw an arrow/line connecting two existing shapes by their IDs.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "from_shape_id": types.Schema(type=types.Type.STRING),
                        "to_shape_id": types.Schema(type=types.Type.STRING),
                        "style": types.Schema(type=types.Type.STRING, enum=["solid", "dashed"]),
                        "label": types.Schema(type=types.Type.STRING),
                    },
                    required=["from_shape_id", "to_shape_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="position_relative_to",
                description="Create a shape positioned relative to an existing shape instead of absolute coordinates.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "anchor_shape_id": types.Schema(type=types.Type.STRING),
                        "direction": types.Schema(type=types.Type.STRING, enum=["above", "below", "left", "right"]),
                        "gap": types.Schema(type=types.Type.INTEGER, description="Grid units of spacing"),
                        "shape_type": types.Schema(type=types.Type.STRING, enum=["rect", "ellipse", "diamond"]),
                        "grid_width": types.Schema(type=types.Type.INTEGER),
                        "grid_height": types.Schema(type=types.Type.INTEGER),
                        "label": types.Schema(type=types.Type.STRING),
                    },
                    required=["anchor_shape_id", "direction", "shape_type", "grid_width", "grid_height"],
                ),
            ),
            types.FunctionDeclaration(
                name="update_shape",
                description="Update properties of an existing shape.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "shape_id": types.Schema(type=types.Type.STRING),
                        "fill": types.Schema(type=types.Type.STRING),
                        "label": types.Schema(type=types.Type.STRING),
                        "grid_x": types.Schema(type=types.Type.INTEGER),
                        "grid_y": types.Schema(type=types.Type.INTEGER),
                    },
                    required=["shape_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="delete_shape",
                description="Remove a shape from the canvas.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"shape_id": types.Schema(type=types.Type.STRING)},
                    required=["shape_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="clear_canvas",
                description="Remove all shapes from the canvas.",
                parameters=types.Schema(type=types.Type.OBJECT, properties={}),
            ),
        ]
    )
]


# ---------------------------------------------------------------------------
# 2. Pydantic validation models (defense against malformed tool-call args)
# ---------------------------------------------------------------------------

class ShapeType(str, Enum):
    rect = "rect"
    ellipse = "ellipse"
    diamond = "diamond"
    line = "line"
    arrow = "arrow"


class CreateShapeArgs(BaseModel):
    shape_type: ShapeType
    grid_x: int = Field(ge=0, le=24)
    grid_y: int = Field(ge=0, le=16)
    grid_width: int = Field(ge=1, le=24)
    grid_height: int = Field(ge=1, le=16)
    fill: Optional[str] = "#4F8EF7"
    stroke: Optional[str] = "#1A1A1A"
    label: Optional[str] = None


class CreateTextArgs(BaseModel):
    grid_x: int = Field(ge=0, le=24)
    grid_y: int = Field(ge=0, le=16)
    text: str
    font_size: Optional[int] = 16


class CreateConnectorArgs(BaseModel):
    from_shape_id: str
    to_shape_id: str
    style: Optional[Literal["solid", "dashed"]] = "solid"
    label: Optional[str] = None


class PositionRelativeArgs(BaseModel):
    anchor_shape_id: str
    direction: Literal["above", "below", "left", "right"]
    gap: Optional[int] = 1
    shape_type: ShapeType
    grid_width: int = Field(ge=1, le=24)
    grid_height: int = Field(ge=1, le=16)
    label: Optional[str] = None


class UpdateShapeArgs(BaseModel):
    shape_id: str
    fill: Optional[str] = None
    label: Optional[str] = None
    grid_x: Optional[int] = None
    grid_y: Optional[int] = None


class DeleteShapeArgs(BaseModel):
    shape_id: str


VALIDATORS = {
    "create_shape": CreateShapeArgs,
    "create_text": CreateTextArgs,
    "create_connector": CreateConnectorArgs,
    "position_relative_to": PositionRelativeArgs,
    "update_shape": UpdateShapeArgs,
    "delete_shape": DeleteShapeArgs,
}

GRID_UNIT_PX = 40  # 1 grid unit = 40px on the frontend canvas


# ---------------------------------------------------------------------------
# 3. Canvas state - the backend's source of truth, reconciled to the frontend
# ---------------------------------------------------------------------------

class CanvasState:
    def __init__(self):
        self.shapes: dict[str, dict] = {}

    def create_shape(self, args: CreateShapeArgs) -> dict:
        shape_id = str(uuid.uuid4())[:8]
        shape = {
            "id": shape_id,
            "kind": "shape",
            "shape_type": args.shape_type.value,
            "x": args.grid_x * GRID_UNIT_PX,
            "y": args.grid_y * GRID_UNIT_PX,
            "width": args.grid_width * GRID_UNIT_PX,
            "height": args.grid_height * GRID_UNIT_PX,
            "fill": args.fill,
            "stroke": args.stroke,
            "label": args.label,
        }
        self.shapes[shape_id] = shape
        return {"event": "shape_created", "shape": shape}

    def create_text(self, args: CreateTextArgs) -> dict:
        shape_id = str(uuid.uuid4())[:8]
        shape = {
            "id": shape_id,
            "kind": "text",
            "x": args.grid_x * GRID_UNIT_PX,
            "y": args.grid_y * GRID_UNIT_PX,
            "text": args.text,
            "font_size": args.font_size,
        }
        self.shapes[shape_id] = shape
        return {"event": "text_created", "shape": shape}

    def create_connector(self, args: CreateConnectorArgs) -> dict:
        if args.from_shape_id not in self.shapes or args.to_shape_id not in self.shapes:
            return {"event": "error", "message": "connector references unknown shape id"}
        connector_id = str(uuid.uuid4())[:8]
        connector = {
            "id": connector_id,
            "kind": "connector",
            "from": args.from_shape_id,
            "to": args.to_shape_id,
            "style": args.style,
            "label": args.label,
        }
        self.shapes[connector_id] = connector
        return {"event": "connector_created", "connector": connector}

    def position_relative_to(self, args: PositionRelativeArgs) -> dict:
        anchor = self.shapes.get(args.anchor_shape_id)
        if anchor is None or anchor.get("kind") != "shape":
            return {"event": "error", "message": "unknown anchor shape id"}

        gap_px = args.gap * GRID_UNIT_PX
        w, h = args.grid_width * GRID_UNIT_PX, args.grid_height * GRID_UNIT_PX

        if args.direction == "below":
            x, y = anchor["x"], anchor["y"] + anchor["height"] + gap_px
        elif args.direction == "above":
            x, y = anchor["x"], anchor["y"] - h - gap_px
        elif args.direction == "right":
            x, y = anchor["x"] + anchor["width"] + gap_px, anchor["y"]
        else:  # left
            x, y = anchor["x"] - w - gap_px, anchor["y"]

        shape_id = str(uuid.uuid4())[:8]
        shape = {
            "id": shape_id,
            "kind": "shape",
            "shape_type": args.shape_type.value,
            "x": x, "y": y, "width": w, "height": h,
            "fill": "#4F8EF7", "stroke": "#1A1A1A", "label": args.label,
        }
        self.shapes[shape_id] = shape
        return {"event": "shape_created", "shape": shape}

    def update_shape(self, args: UpdateShapeArgs) -> dict:
        shape = self.shapes.get(args.shape_id)
        if shape is None:
            return {"event": "error", "message": "unknown shape id"}
        for field in ("fill", "label"):
            val = getattr(args, field)
            if val is not None:
                shape[field] = val
        if args.grid_x is not None:
            shape["x"] = args.grid_x * GRID_UNIT_PX
        if args.grid_y is not None:
            shape["y"] = args.grid_y * GRID_UNIT_PX
        return {"event": "shape_updated", "shape": shape}

    def delete_shape(self, args: DeleteShapeArgs) -> dict:
        self.shapes.pop(args.shape_id, None)
        return {"event": "shape_deleted", "shape_id": args.shape_id}

    def clear_canvas(self) -> dict:
        self.shapes.clear()
        return {"event": "canvas_cleared"}


HANDLERS = {
    "create_shape": CanvasState.create_shape,
    "create_text": CanvasState.create_text,
    "create_connector": CanvasState.create_connector,
    "position_relative_to": CanvasState.position_relative_to,
    "update_shape": CanvasState.update_shape,
    "delete_shape": CanvasState.delete_shape,
}


# ---------------------------------------------------------------------------
# 4. Tool-call dispatch with validation + debounced event forwarding
# ---------------------------------------------------------------------------

async def handle_tool_call(
    canvas: CanvasState,
    name: str,
    raw_args: dict,
    send_to_frontend,
    stagger_seconds: float = 0.08,
):
    """Validate one Gemini tool call, apply it to canvas state, forward the
    resulting event to the frontend with a small stagger so shapes appear
    to be drawn live rather than popping in all at once."""

    if name == "clear_canvas":
        result = canvas.clear_canvas()
        await send_to_frontend(result)
        return {"status": "ok"}

    validator = VALIDATORS.get(name)
    if validator is None:
        return {"status": "error", "message": f"unknown tool '{name}'"}

    try:
        args = validator(**raw_args)
    except ValidationError as e:
        # Don't crash the Live session - tell the model what was wrong so it can retry.
        return {"status": "error", "message": f"invalid arguments: {e.errors()}"}

    handler = HANDLERS[name]
    result = handler(canvas, args)

    await asyncio.sleep(stagger_seconds)
    await send_to_frontend(result)

    if result.get("event") == "error":
        return {"status": "error", "message": result["message"]}
    return {"status": "ok", "shape_id": result.get("shape", {}).get("id") or result.get("connector", {}).get("id")}


# ---------------------------------------------------------------------------
# 5. Gemini Live session wiring
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are a whiteboard drawing assistant. When the user
asks you to draw something, call the drawing tools to build it.

Rules:
- Think in a 24x16 grid (grid_x: 0-24, grid_y: 0-16), not pixels.
- For diagrams with multiple connected shapes (flowcharts, org charts), create
  each shape first, then use create_connector with the returned shape IDs -
  never guess coordinates for arrows.
- Prefer position_relative_to over absolute coordinates when placing a shape
  next to one you just created.
- Keep shapes within the grid bounds and leave at least 1 grid unit of gap
  between shapes so connectors are readable.
"""


async def run_whiteboard_session(api_key: str, send_to_frontend):
    """
    send_to_frontend: async callable(dict) -> None, e.g. a WebSocket .send()
    wrapper that forwards shape events to your react-konva/tldraw frontend.
    """
    client = genai.Client(api_key=api_key)
    canvas = CanvasState()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_INSTRUCTION,
        tools=DRAWING_TOOLS,
    )

    async with client.aio.live.connect(model="gemini-2.0-flash-live-001", config=config) as session:
        async for response in session.receive():
            if response.tool_call:
                function_responses = []
                for fc in response.tool_call.function_calls:
                    outcome = await handle_tool_call(
                        canvas, fc.name, dict(fc.args or {}), send_to_frontend
                    )
                    function_responses.append(
                        types.FunctionResponse(id=fc.id, name=fc.name, response=outcome)
                    )
                await session.send_tool_response(function_responses=function_responses)

            # Forward audio/text response chunks to the client as needed
            if response.server_content and response.server_content.model_turn:
                for part in response.server_content.model_turn.parts:
                    if part.text:
                        await send_to_frontend({"event": "assistant_text", "text": part.text})


# ---------------------------------------------------------------------------
# 6. Example: minimal WebSocket server wiring (FastAPI)
# ---------------------------------------------------------------------------
"""
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/whiteboard")
async def whiteboard_ws(ws: WebSocket):
    await ws.accept()

    async def send_to_frontend(payload: dict):
        await ws.send_text(json.dumps(payload))

    await run_whiteboard_session(api_key="YOUR_GEMINI_API_KEY", send_to_frontend=send_to_frontend)
"""