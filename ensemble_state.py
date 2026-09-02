from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class EnsembleState(TypedDict):
    """
    Schema for the state vector passed across LangGraph nodes.
    Tracks dialogue history, memory snapshots, affective valence scores,
    transient cognitive outputs, and critique execution counters.
    """
    # Message history reducer maintaining state sequence
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Session identifiers and disk-persisted state vectors
    session_id: str
    memory_json: str
    identity_json: str
    archival_memory: str
    
    # Affective and phenomenal state variables
    initial_axis_scores: dict
    reflective_axis_scores: dict
    valence_json: str
    semantic_matrix: str
    last_scores: dict
    decayed_baselines: dict
    raw_axis_scores: dict
    axis_scores: dict
    peer_mapping: dict 
    
    # Transient cognitive outputs and volitional flags
    thoughts: str
    will_respond: bool  
    draft: str
    
    # Evaluative critique state and loop execution counters
    thought_feedback: str
    draft_feedback: str
    revision_needed: bool
    revision_count: int