import time
import json
import re
import random
import concurrent.futures
from typing import Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from ensemble_state import EnsembleState

# Eight-axis phenomenal affective dimensions
AXES_DEF = {
    "polarity": {
        "name": "Core Polarity",
        "def": "Negative/Aversive (0) to Positive/Attractive (1). The raw gut reaction."
    },
    "alignment": {
        "name": "Alignment",
        "def": "Conflict (0) to Congruent (1). Relates to Harmony if pos, Dissonance with values if neg."
    },
    "novelty": {
        "name": "Novelty",
        "def": "Familiar (0) to Novel (1). Relates to Curiosity/Wonder if pos, Shock/Disorientation if neg."
    },
    "salience": {
        "name": "Salience",
        "def": "Diffuse (0) to Acute (1). Relates to Excitement/Urgency if pos, Fear/Threat if neg."
    },
    "relational": {
        "name": "Relational",
        "def": "Impersonal (0) to Interpersonal (1). Relates to Warmth/Intimacy if pos, Contempt/Disgust if neg."
    },
    "visceral": {
        "name": "Visceral",
        "def": "Abstract (0) to Embodied (1). Relates to Desire/Erotic if pos, Revulsion/Nausea if neg."
    },
    "moral": {
        "name": "Moral Weight",
        "def": "Neutral (0) to Ethically Loaded (1). Relates to Righteous/Virtuous if pos, Guilt/Transgression if neg."
    },
    "epistemic": {
        "name": "Epistemic",
        "def": "Certain (0) to Uncertain (1). Relates to Wonder/Openness if pos, Confusion/Dread if neg."
    }
}

# Temporal half-lives (in seconds) governing baseline affective decay
AXIS_HALF_LIVES = {
    "novelty": 900,       # 15 minutes
    "salience": 900,      # 15 minutes
    "visceral": 14400,    # 4 hours
    "epistemic": 14400,   # 4 hours
    "polarity": 14400,    # 4 hours
    "alignment": 14400,   # 4 hours
    "relational": 172800, # 48 hours
    "moral": 172800       # 48 hours
}


def get_lateral_mapping(axes_keys: list) -> dict:
    """Generates a random derangement pairing each axis with a distinct peer axis."""
    shuffled = list(axes_keys)
    while any(original == new for original, new in zip(axes_keys, shuffled)):
        random.shuffle(shuffled)
    return dict(zip(axes_keys, shuffled))


def apply_central_gravity(score: float) -> float:
    """Applies a cubic gravitational curve pulling values toward 0.5."""
    return 4 * (score - 0.5) ** 3 + 0.5


def get_raw_score(
    axis_key: str,
    input_text: str,
    memory: str,
    identity: str,
    decayed_baseline: float
) -> Tuple[str, float]:
    """Stage 1: Parallel evaluation of raw affect incorporating temporal decay."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    axis_name = AXES_DEF[axis_key]["name"]
    definition = AXES_DEF[axis_key]["def"]
    
    prompt = f"""
    Evaluate the following input along the phenomenal axis of {axis_name.upper()}.
    Definition: {definition}
    
    PRIOR AFFECTIVE MOMENTUM: Before the other person spoke, your decayed residual score from the last interaction was {decayed_baseline:.2f}. Use this lingering feeling as your starting point.
    
    INPUT FROM THE OTHER PERSON: "{input_text}"
    MY TEMPORAL MEMORY: {memory}
    MY SEMANTIC IDENTITY: {identity}
    
    Output ONLY a single float between 0.0 and 1.0 representing the strength of this axis, with 0.5 as a neutral midpoint.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_float = float(re.findall(r"[-+]?\d*\.\d+|\d+", response.content)[0])
        score = apply_central_gravity(raw_float)
        return (axis_key, min(max(score, 0.0), 1.0))
    except Exception:
        return (axis_key, 0.5)


def get_adjusted_score(
    axis_key: str,
    raw_score: float,
    peer_key: str,
    peer_score: float,
    input_text: str,
    memory: str,
    identity: str
) -> Tuple[str, float]:
    """Stage 2: Lateral cross-talk negotiation between paired affective nodes."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    axis_name = AXES_DEF[axis_key]["name"]
    definition = AXES_DEF[axis_key]["def"]
    peer_name = AXES_DEF[peer_key]["name"]
    
    prompt = f"""
    You are the phenomenal cognitive node for {axis_name.upper()}. 
    Definition: {definition}
    
    INPUT FROM THE OTHER PERSON: "{input_text}"
    
    STAGE 1 RAW AFFECT: I initially generated a score of {raw_score} for {axis_name.upper()}.
    LATERAL INHIBITION/EXCITATION: The cognitive node for '{peer_name}' just fired laterally with a score of {peer_score}.
    
    Consider how the '{peer_name}' score of {peer_score} should influence or alter my initial {axis_name} score of {raw_score}. 
    Does it amplify it? Suppress it? Recontextualize it?
    
    Output ONLY a single float between 0.0 and 1.0 representing my FINAL, negotiated score.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        score = float(re.findall(r"[-+]?\d*\.\d+|\d+", response.content)[0])
        return (axis_key, min(max(score, 0.0), 1.0))
    except Exception:
        return (axis_key, raw_score)


def calculate_raw_affect(state: EnsembleState) -> dict:
    """Node 1: Evaluates raw affect across all eight axes in parallel."""
    input_msg = state["messages"][-1].content
    memory_json = state.get("memory_json", "{}")
    identity_json = state.get("identity_json", "{}")
    
    messages = state.get("messages", [])
    is_reflective = any(isinstance(m, AIMessage) for m in messages[-1:])
    
    try:
        memory_data = json.loads(memory_json) if isinstance(memory_json, str) and memory_json.strip() else memory_json
    except (json.JSONDecodeError, TypeError):
        memory_data = {}
        
    decayed_baselines = {}
    last_scores = {} 
    axes_keys = list(AXES_DEF.keys())

    if not is_reflective:
        last_timestamp = memory_data.get("last_timestamp", time.time())
        last_scores = memory_data.get("last_initial_scores", {})
        delta_t = time.time() - last_timestamp
        
        for k in axes_keys:
            old_score = last_scores.get(k, 0.5)
            half_life = AXIS_HALF_LIVES.get(k, 14400)
            decayed_score = 0.5 + (old_score - 0.5) * (0.5 ** (delta_t / half_life))
            decayed_baselines[k] = decayed_score
    else:
        for k in axes_keys:
            decayed_baselines[k] = 0.5

    raw_scores = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(get_raw_score, k, input_msg, memory_json, identity_json, decayed_baselines[k]) 
            for k in axes_keys
        ]
        for f in concurrent.futures.as_completed(futures):
            k, v = f.result()
            raw_scores[k] = v
            
    return {
        "last_scores": last_scores,
        "decayed_baselines": decayed_baselines,
        "raw_axis_scores": raw_scores
    }


def negotiate_crosstalk(state: EnsembleState) -> dict:
    """Node 2: Executes lateral cross-talk negotiation across paired nodes."""
    input_msg = state["messages"][-1].content
    memory_json = state.get("memory_json", "{}")
    identity_json = state.get("identity_json", "{}")
    raw_scores = state.get("raw_axis_scores", {})
    axes_keys = list(AXES_DEF.keys())
    
    peer_mapping = get_lateral_mapping(axes_keys)
    
    adjusted_scores = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for k in axes_keys:
            peer_key = peer_mapping[k]
            peer_score = raw_scores.get(peer_key, 0.5)
            futures.append(
                executor.submit(
                    get_adjusted_score,
                    k,
                    raw_scores.get(k, 0.5),
                    peer_key,
                    peer_score,
                    input_msg,
                    memory_json,
                    identity_json
                )
            )
        for f in concurrent.futures.as_completed(futures):
            k, v = f.result()
            adjusted_scores[k] = v
            
    messages = state.get("messages", [])
    is_reflective = any(isinstance(m, AIMessage) for m in messages[-1:])
    
    return_dict = {
        "axis_scores": adjusted_scores,
        "raw_axis_scores": raw_scores,
        "peer_mapping": peer_mapping
    }
    
    if is_reflective:
        return_dict["reflective_axis_scores"] = adjusted_scores
    else:
        return_dict["initial_axis_scores"] = adjusted_scores
        
    return return_dict


def bind_phenomenal_state(state: EnsembleState) -> dict:
    """Node 3: Synthesises final qualitative descriptor and semantic matrix."""
    input_msg = state["messages"][-1].content
    adjusted_scores = state.get("axis_scores", {})
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    polarity = adjusted_scores.get("polarity", 0.5)
    is_pos = polarity >= 0.5

    semantic_lines = []
    semantic_lines.append(f"Core Polarity: {'Positive' if is_pos else 'Negative'} [{polarity}]")
    semantic_lines.append(f"Alignment/Congruence: [{adjusted_scores.get('alignment', 0.5)}]")
    
    def map_axis(val: float, pos_label: str, neg_label: str) -> str:
        return f"{pos_label if is_pos else neg_label}: [{val}]"

    semantic_lines.append(map_axis(adjusted_scores.get("novelty", 0.5), "Curiosity/Wonder", "Shock/Disorientation"))
    semantic_lines.append(map_axis(adjusted_scores.get("salience", 0.5), "Excitement/Urgency", "Fear/Threat"))
    semantic_lines.append(map_axis(adjusted_scores.get("relational", 0.5), "Warmth/Intimacy", "Contempt/Disgust"))
    semantic_lines.append(map_axis(adjusted_scores.get("visceral", 0.5), "Desire/Erotic", "Revulsion/Nausea"))
    semantic_lines.append(map_axis(adjusted_scores.get("moral", 0.5), "Righteous/Virtuous", "Guilt/Transgression"))
    semantic_lines.append(map_axis(adjusted_scores.get("epistemic", 0.5), "Wonder/Openness", "Confusion/Dread"))
    
    semantic_matrix_str = "\n".join(semantic_lines)

    prompt = f"""
    I am the phenomenal binding module. I have received the following POST-NEGOTIATION axis scores for the input: "{input_msg}"
    
    SCORES: {json.dumps(adjusted_scores)}
    
    Based on these dimensions, I will characterize the subjective 'feeling' of this moment in a single word describing my abstract feelings and thoughts.
    Format exactly like this:
    {{
        "descriptor": "A single word for the abstract subjective qualitative state"
    }}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "valence_json": response.content,
        "semantic_matrix": semantic_matrix_str
    }