import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from src.graph import graph
from src.state import GaokaoState, default_profile
from src.db import init_db, load_session, save_session, delete_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Gaokao Advisor Agent", version="0.2.0", lifespan=lifespan)

sessions: dict[str, GaokaoState] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


def _get_or_create_state(session_id: str) -> GaokaoState:
    if session_id not in sessions:
        # 尝试从 SQLite 恢复
        saved = load_session(session_id)
        if saved:
            stage = saved.pop("_stage", "collecting")
            sessions[session_id] = {
                "messages": [],
                "user_profile": saved,
                "stage": stage,
            }
        else:
            sessions[session_id] = {
                "messages": [],
                "user_profile": default_profile(),
                "stage": "collecting",
            }
    return sessions[session_id]


def _save_state(session_id: str, state: GaokaoState):
    profile = state.get("user_profile", {})
    stage = state.get("stage", "collecting")
    save_session(session_id, profile, stage)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    state = _get_or_create_state(req.session_id)
    state["messages"].append(HumanMessage(content=req.message))

    result = await graph.ainvoke(state)

    sessions[req.session_id] = result
    _save_state(req.session_id, result)

    last_msg = result["messages"][-1]
    return ChatResponse(
        session_id=req.session_id,
        response=last_msg.content if hasattr(last_msg, "content") else str(last_msg),
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    state = _get_or_create_state(req.session_id)
    state["messages"].append(HumanMessage(content=req.message))

    collected_msgs = []

    async def event_generator():
        nonlocal collected_msgs
        async for event in graph.astream_events(state, version="v2"):
            kind = event.get("event", "")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and chunk.content:
                    yield f"data: {json.dumps({'content': chunk.content}, ensure_ascii=False)}\n\n"
            elif kind == "on_chat_model_end":
                output = event.get("data", {}).get("output")
                if output is not None:
                    collected_msgs.append(output)
            elif kind == "on_tool_start":
                tool_name = event.get("name", "")
                yield f"data: {json.dumps({'tool': tool_name, 'status': 'start'}, ensure_ascii=False)}\n\n"
            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                output = event.get("data", {}).get("output")
                if output is not None:
                    collected_msgs.append(output)
                yield f"data: {json.dumps({'tool': tool_name, 'status': 'end'}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

        for msg in collected_msgs:
            state["messages"].append(msg)
        sessions[req.session_id] = state
        _save_state(req.session_id, state)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
