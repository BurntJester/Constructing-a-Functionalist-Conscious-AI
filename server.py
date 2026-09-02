import os
import json
import uuid
import time
import re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from coordinator import ensemble_app

app = FastAPI(title="Conscious AI Ensemble API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS_DIR = "sessions"
if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)


def initialize_system_state() -> None:
    """Boot-up validation establishing baseline JSON state vectors on fresh installation."""
    if not os.path.exists("identity.json"):
        default_identity = {
            "my_attractions_and_repulsions": [],
            "my_sense_of_congruence": [],
            "my_relationship_with_the_unknown": [],
            "what_demands_my_attention": [],
            "how_i_view_myself_and_others": [],
            "my_relationship_with_truth_and_ambiguity": [],
            "what_stabilises_or_strains_my_architecture": [],
            "my_rules_of_integrity": []
        }
        with open("identity.json", "w", encoding="utf-8") as f:
            json.dump(default_identity, f, indent=2)

    if not os.path.exists("memory.json"):
        default_memory = {
            "episodic_summaries": [],
            "my_open_goals": [],
            "last_timestamp": time.time(),
            "last_initial_scores": {},
            "last_reflective_scores": {}
        }
        with open("memory.json", "w", encoding="utf-8") as f:
            json.dump(default_memory, f, indent=2)

    if not os.path.exists("agent_meta.json"):
        default_meta = {"agent_name": "AI Ensemble"}
        with open("agent_meta.json", "w", encoding="utf-8") as f:
            json.dump(default_meta, f, indent=2)


initialize_system_state()


def get_agent_name(default_api_name: str = "Ensemble") -> str:
    """Reads the active identity moniker from metadata persistence."""
    try:
        if os.path.exists("agent_meta.json"):
            with open("agent_meta.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("agent_name", default_api_name)
    except Exception:
        pass
    return default_api_name


def serialize_state(obj: Any) -> Any:
    """Recursively serialises state dictionaries and LangChain message objects into JSON structures."""
    if isinstance(obj, list): 
        return [serialize_state(item) for item in obj]
    if isinstance(obj, dict): 
        return {k: serialize_state(v) for k, v in obj.items()}
    if hasattr(obj, "type") and hasattr(obj, "content"):
        role = "other person" if obj.type in ["human", "system"] else "me"
        return {"type": role, "content": obj.content}
    return obj


def generate_title(user_msg: str) -> str:
    """Generates a concise title summary for session management."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    prompt = f"Summarize this user request into a 3 to 5 word conversation title. Output ONLY the title: '{user_msg}'"
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().replace('"', '')
    except Exception:
        return user_msg[:30] + "..."


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class RenameRequest(BaseModel):
    title: str


@app.get("/sessions")
async def get_sessions() -> Dict[str, List[Dict[str, str]]]:
    """Retrieves stored chat session identifiers and titles."""
    session_list = []
    if os.path.exists(SESSIONS_DIR):
        for f in os.listdir(SESSIONS_DIR):
            if f.endswith(".json"):
                path = os.path.join(SESSIONS_DIR, f)
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        session_list.append({
                            "id": f.replace(".json", ""),
                            "title": data.get("title", "Untitled Conversation")
                        })
                except Exception:
                    continue
    return {"sessions": session_list}


@app.get("/sessions/{session_id}")
async def get_session_detail(session_id: str) -> Dict[str, Any]:
    """Loads historical session payload by ID and injects active agent metadata."""
    session_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")
    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data["agent_name"] = get_agent_name("AI Ensemble")
        return data


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Removes specified session file from disk."""
    session_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(session_path):
        os.remove(session_path)
    return {"status": "deleted"}


@app.put("/sessions/{session_id}/title")
async def rename_session(session_id: str, request: RenameRequest) -> Dict[str, str]:
    """Updates session title in disk store."""
    session_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")
    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["title"] = request.title
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return {"status": "success"}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> StreamingResponse:
    """Executes state graph progression and streams node output telemetry via Server-Sent Events."""
    session_id = request.session_id or str(uuid.uuid4())
    session_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    history = []
    title = None

    if os.path.exists(session_path):
        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            title = data.get("title")
            for m in data.get("messages", []):
                if m.get("is_system"):
                    history.append(SystemMessage(content=m["content"]))
                elif m["type"] in ["human", "other person"]: 
                    history.append(HumanMessage(content=m["content"]))
                else: 
                    history.append(AIMessage(content=m["content"]))
    else:
        try:
            with open("memory.json", "r", encoding="utf-8") as f:
                mem_data = json.load(f)
                eps = mem_data.get("episodic_summaries", [])
                goals = mem_data.get("my_open_goals", [])
                
                sys_content = "BACKGROUND COGNITIVE CONTEXT (Your memories):\n"
                if eps:
                    sys_content += "Prior Memories:\n" + "\n".join(eps) + "\n\n"
                if goals:
                    sys_content += "Current Open Goals:\n" + "\n".join(goals)
                
                if eps or goals:
                    history.append(SystemMessage(content=sys_content))
        except Exception:
            pass

    async def event_generator():
        nonlocal title
        user_msg = HumanMessage(content=request.message)
        history.append(user_msg)
        
        if not title:
            title = generate_title(request.message)

        current_state = {
            "messages": history, 
            "session_id": session_id,
            "memory_json": "{}", 
            "identity_json": "{}", 
        }

        final_history = list(history)

        for event in ensemble_app.stream(current_state):
            for node_name, node_state in event.items():
                if node_state is None:
                    continue
                
                if "messages" in node_state:
                    msgs = node_state["messages"]
                    if len(msgs) == 1:
                        final_history.append(msgs[0])
                    elif len(msgs) > 1:
                        final_history = msgs

                payload = {
                    "node": node_name, 
                    "data": serialize_state(node_state), 
                    "session_id": session_id, 
                    "title": title
                }
                yield f"data: {json.dumps(payload)}\n\n"

        final_msg = final_history[-1]
        final_payload = {
            "node": "FINAL_OUTPUT",
            "data": {"messages": [serialize_state(final_msg)]},
            "session_id": session_id,
            "sender_name": get_agent_name("AI Ensemble")
        }
        yield f"data: {json.dumps(final_payload)}\n\n"

        serializable_msgs = []
        for m in final_history:
            if isinstance(m, SystemMessage):
                serializable_msgs.append({"type": "other person", "content": m.content, "is_system": True})
            else:
                serializable_msgs.append({"type": ("other person" if m.type == "human" else "me"), "content": m.content})
                
        save_blob = {"title": title, "messages": serializable_msgs}
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(save_blob, f, indent=2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")