import warnings
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from ensemble_state import EnsembleState
from memory_manager import retrieve_memory
from memory_updater import update_memory
from valence_generator import (
    calculate_raw_affect,
    negotiate_crosstalk,
    bind_phenomenal_state,
)
from interpreter import generate_thoughts
from drafter import generate_draft
from critic import evaluate_thoughts, evaluate_draft

warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
load_dotenv()


def apply_veto(state: EnsembleState) -> dict:
    """Executes the volitional veto by recording an internal silent state."""
    silent_msg = AIMessage(content="[The ensemble chose not to respond.]")
    return {"messages": [silent_msg]}


def route_thoughts(state: EnsembleState) -> str:
    """Routes state based on thought critique and loop execution bounds."""
    if state.get("revision_needed") and state.get("revision_count", 0) < 3:
        return "generate_thoughts"
    if not state.get("will_respond", True):
        return "apply_veto"
    return "generate_draft"


def route_draft(state: EnsembleState) -> str:
    """Routes state based on draft critique and loop execution bounds."""
    if state.get("revision_needed") and state.get("revision_count", 0) < 3:
        return "generate_thoughts"
    return "update_memory"


def route_post_valence(state: EnsembleState) -> str:
    """Evaluates recent message type to determine processing path."""
    messages = state.get("messages") or []
    if not messages:
        return "END"
    
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        return "generate_thoughts"
    elif isinstance(last_message, AIMessage):
        return "evaluate_draft"
        
    return "END"


# Initialise LangGraph State Machine
graph = StateGraph(EnsembleState)

# Add Node Definitions
graph.add_node("retrieve_memory", retrieve_memory)
graph.add_node("calculate_raw_affect", calculate_raw_affect)
graph.add_node("negotiate_crosstalk", negotiate_crosstalk)
graph.add_node("bind_phenomenal_state", bind_phenomenal_state)
graph.add_node("generate_thoughts", generate_thoughts)
graph.add_node("evaluate_thoughts", evaluate_thoughts)
graph.add_node("generate_draft", generate_draft)
graph.add_node("evaluate_draft", evaluate_draft)
graph.add_node("update_memory", update_memory)
graph.add_node("apply_veto", apply_veto)

# Sequential Edges
graph.add_edge(START, "retrieve_memory")
graph.add_edge("retrieve_memory", "calculate_raw_affect")
graph.add_edge("calculate_raw_affect", "negotiate_crosstalk")
graph.add_edge("negotiate_crosstalk", "bind_phenomenal_state")

# Conditional Edge Routing
graph.add_conditional_edges(
    "bind_phenomenal_state", 
    route_post_valence,
    {
        "generate_thoughts": "generate_thoughts",
        "evaluate_draft": "evaluate_draft",
        "END": END,
    }
)

graph.add_edge("generate_thoughts", "evaluate_thoughts")

graph.add_conditional_edges(
    "evaluate_thoughts", 
    route_thoughts,
    {
        "generate_thoughts": "generate_thoughts",
        "apply_veto": "apply_veto", 
        "generate_draft": "generate_draft",
    }
)

graph.add_edge("generate_draft", "calculate_raw_affect")

graph.add_conditional_edges(
    "evaluate_draft",
    route_draft,
    {
        "generate_thoughts": "generate_thoughts",
        "update_memory": "update_memory",
    }
)

graph.add_edge("apply_veto", "update_memory")
graph.add_edge("update_memory", END)

ensemble_app = graph.compile()