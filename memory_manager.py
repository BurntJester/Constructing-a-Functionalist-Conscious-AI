import os
import json
import time
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.messages import HumanMessage
from ensemble_state import EnsembleState


def retrieve_memory(state: EnsembleState) -> dict:
    """
    Initialises memory retrieval at graph boot-up and subsequent execution turns:
    - Loads identity.json on every cycle.
    - Loads full memory.json text summaries strictly on the first turn of a session.
    - Loads lightweight temporal metadata (timestamps and affect scores) mid-session to optimise context length.
    - Queries ChromaDB vector store for relevant episodic archival records.
    """
    state["revision_count"] = 0  
    
    # 1. Identity Matrix: Loaded on every turn
    id_path = "identity.json"
    try:
        if os.path.exists(id_path) and os.path.getsize(id_path) > 0:
            with open(id_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                state["identity_json"] = content if content else "{}"
        else:
            state["identity_json"] = "{}"
    except Exception:
        state["identity_json"] = "{}"

    # 2. Episodic & Temporal Memory: Selective metadata loading
    is_first_turn = len(state.get("messages", [])) <= 1
    mem_path = "memory.json"
    
    try:
        if os.path.exists(mem_path) and os.path.getsize(mem_path) > 0:
            with open(mem_path, "r", encoding="utf-8") as f:
                mem_data = json.load(f)
                
            if is_first_turn:
                state["memory_json"] = json.dumps(mem_data)
            else:
                minimal_mem = {
                    "last_timestamp": mem_data.get("last_timestamp", time.time()),
                    "last_initial_scores": mem_data.get("last_initial_scores", {}),
                    "last_reflective_scores": mem_data.get("last_reflective_scores", {})
                }
                state["memory_json"] = json.dumps(minimal_mem)
        else:
            state["memory_json"] = "{}"
    except Exception:
        state["memory_json"] = "{}"

    # 3. Vector Database Retrieval (RAG)
    state["archival_memory"] = "No relevant archival memories found."
    try:
        messages = state.get("messages", [])
        user_input = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
            ""
        )
        
        if user_input:
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            collection = chroma_client.get_or_create_collection(
                name="episodic_archive",
                embedding_function=ef
            )
            
            if collection.count() > 0:
                results = collection.query(query_texts=[user_input], n_results=3)
                if results and results.get("documents") and results["documents"][0]:
                    state["archival_memory"] = json.dumps(results["documents"][0], indent=2)
    except Exception:
        pass

    return {k: v for k, v in state.items() if k != "messages"}