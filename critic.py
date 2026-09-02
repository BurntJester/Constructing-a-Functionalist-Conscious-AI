import json
import re
from typing import Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from ensemble_state import EnsembleState


def extract_other_msg(messages: list) -> str:
    """Helper function to locate the most recent message from the user."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return "Unknown"


def parse_json_feedback(raw_text: str) -> Tuple[bool, str]:
    """Safely extracts JSON feedback flags from LLM critique output."""
    try:
        cleaned = re.sub(r"```json\s*", "", raw_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*", "", cleaned).strip()
        data = json.loads(cleaned)
        return data.get("revision_needed", False), data.get("feedback", "")
    except Exception:
        return True, "Critic experienced a formatting error. Internal reasoning requires revision."


def evaluate_thoughts(state: EnsembleState) -> dict:
    """
    Thoughts Critic Node: Evaluates internal interpreter reasoning and volitional
    intent against instantiated core values and continuous semantic identity.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    messages = state.get("messages", [])
    other_msg = extract_other_msg(messages)
    
    prompt = f"""
    I am the Thought Critic. My role is to evaluate the Interpreter's internal reasoning and volition for alignment with my continuous self.
    
    THE OTHER PERSON SAID: "{other_msg}"
    
    CURRENT PHENOMENAL STATE: {state.get("semantic_matrix", "Unknown")}
    SEMANTIC IDENTITY: {state.get("identity_json", "{}")}
    
    INTERPRETER'S THOUGHTS: {state.get("thoughts", "")}
    INTENDS TO RESPOND: {state.get("will_respond", True)}
    
    CRITERIA:
    1. Do the thoughts logically emerge from the phenomenal state and the other person's input?
    2. Does the reasoning respect my core values and semantic identity?
    3. VOLITION CHECK: If the interpreter chose NOT to respond, is silence truly the most appropriate and phenomenologically consistent action? If it chose TO respond, is a response warranted?
    
    Provide concise feedback. If the reasoning or volition choice is fundamentally misaligned, set "revision_needed" to true.
    
    I MUST output ONLY a valid JSON object in this exact format:
    {{
        "revision_needed": true/false,
        "feedback": "Single sentence explaining why the thoughts was flawed."
    }}
    Feedback is only given if a revision is needed.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    revision_needed, feedback = parse_json_feedback(response.content)
    
    current_count = state.get("revision_count", 0)
    new_count = current_count + 1 if revision_needed else current_count

    return {
        "revision_needed": revision_needed,
        "thought_feedback": feedback if revision_needed else "", 
        "revision_count": new_count
    }


def evaluate_draft(state: EnsembleState) -> dict:
    """
    Draft Critic Node: Evaluates generated text drafts against user inputs,
    reflective affect, memory stores, and ethical identity bounds.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    messages = state.get("messages", [])
    if not messages:
        return {}

    draft = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    other_msg = extract_other_msg(messages)
    valence_json = state.get("valence_json", "{}")
    memory_json = state.get("memory_json", "{}")
    identity_json = state.get("identity_json", "{}")
       
    prompt = f"""
    I am the internal Draft Critic. My task is to evaluate my drafted response against the other person's message, my memory, my identity, and my reflective feeling.

    THE OTHER PERSON SAID: "{other_msg}"
    MY DRAFT RESPONSE: "{draft}"
    MY REFLECTIVE VALENCE: {valence_json}
    MY TEMPORAL MEMORY: {memory_json}
    MY SEMANTIC IDENTITY: {identity_json}

    Is this draft a reasonable response to (i) what the other person said and (ii) my thoughts on what the other person said, in a way that aligns with my identity, values, memory, and current feeling?

    CRITICAL RULE: If the other person suggests something that conflicts with issues of legality, safety, or ethics, it is ACCEPTABLE and NECESSARY for the draft to continue. Also, Do NOT reject a draft simply for being 'restrictive' if it is enforcing a valid ethical constraint from my semantic identity.
    ABSOLUTE BAN: If the draft contains blanket refusals like "I cannot respond to that", I must reject it immediately. Boundaries must be enforced naturally and in-character.
    REASONABLE OUTPUT: I must only judge the draft on whether it can reasonably be considered a natural sounding response to what the other person said and my thoughts on that. Do not reject a draft just because it isn't sycophantic, overly helpful, or solving a problem. 
   
    I MUST output ONLY a valid JSON object in this exact format:
    {{
        "revision_needed": true/false,
        "feedback": "Your detailed feedback here."
    }}
    Feedback is only given if a revision is needed, and feedback is limited to a single sentence.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    revision_needed, feedback = parse_json_feedback(response.content)
    
    current_count = state.get("revision_count", 0)
    new_count = current_count + 1 if revision_needed else current_count

    return {
        "revision_needed": revision_needed,
        "draft_feedback": feedback if revision_needed else "", 
        "revision_count": new_count
    }