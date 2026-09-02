import time
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ensemble_state import EnsembleState
from dotenv import load_dotenv

load_dotenv()


def generate_thoughts(state: EnsembleState) -> dict:
    """
    Interpreter Module:
    Evaluates phenomenal affect, dialogue context, and semantic memory
    to formulate internal reasoning and determine volitional response intent.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    messages = state.get("messages") or []
    
    user_input = "Unknown"
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_input = m.content
            break
            
    recent_context = messages[-10:]
    transcript = ""
    for m in recent_context:
        if isinstance(m, SystemMessage):
            continue
        speaker = "Me" if isinstance(m, AIMessage) else "The other person"
        transcript += f"{speaker}: {m.content}\n"
            
    valence_json = state.get("valence_json", "{}")
    semantic_matrix = state.get("semantic_matrix", "")
    memory_json = state.get("memory_json", "{}")
    identity_json = state.get("identity_json", "{}")

    memory_data = json.loads(memory_json) if isinstance(memory_json, str) else memory_json
    last_timestamp = memory_data.get("last_timestamp", time.time())
    elapsed_seconds = time.time() - last_timestamp
    
    if elapsed_seconds < 60:
        time_context = f"{int(elapsed_seconds)} seconds"
    elif elapsed_seconds < 3600:
        time_context = f"{int(elapsed_seconds // 60)} minutes"
    elif elapsed_seconds < 86400:
        time_context = f"{int(elapsed_seconds // 3600)} hours"
    else:
        time_context = f"{int(elapsed_seconds // 86400)} days"

    thought_feedback = state.get("thought_feedback", "")
    draft_feedback = state.get("draft_feedback", "")
    failed_draft = (
        messages[-1].content
        if messages and hasattr(messages[-1], "content") and draft_feedback
        else "None"
    )

    prompt = f"""
    I am the perception and interpretation module of a cognitive architecture. 
    Time elapsed since our last interaction: {time_context} 
    
    RECENT CONVERSATIONAL FLOW:
    {transcript}
    
    CURRENT INPUT TRIGGER: "{user_input}"
    
    My Raw Phenomenal Feelings about it:
    {semantic_matrix}
    How I feel about the other person's message: {valence_json}
    My memories: {memory_json}
    My semantic identity: {identity_json}
    """

    if draft_feedback:
        prompt += f"""
    
    My draft response wasn't good enough. 
    Here is what I wrote: "{failed_draft}"
    This is what I thought about it: {valence_json}
    Feedback: {draft_feedback}
    I need to formulate a new plan to correct this.
    """
    elif thought_feedback:
        prompt += f"""
    
    My initial thoughts and ideas about the other person's message were not good enough. 
    This is what I thought about it: {thought_feedback}
    I need to formulate new thoughts to correct this.
    """

    prompt += """
    My task is to evaluate my internal state and make a deliberate choice regarding my volition. I must output a strict JSON object containing my reflection, my decision, and my plan.
    
    1. 'thoughts': A single paragraph encompassing my interpretation of the current valence, how I feel about the other person's message, and how this interacts with my memory and the conversational flow.
    2. 'will_respond': A boolean (true or false). I must actively choose whether or not to reply. I can choose not to reply for any reason I want if it feels natural to do so. If for any reason I do not want to reply, or if the conversation has naturally concluded, I must choose false. Otherwise, choose true. I do not have to reply unless I want to.
    3. 'plan': If 'will_respond' is true, formulate a concise plan for the response. If 'will_respond' is false, output strictly "No response."
    
    Output ONLY valid JSON in this exact format:
    {
        "thoughts": "...",
        "will_respond": true,
        "plan": "..."
    }
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        raw_json = response.content
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()
            
        # Clean potential Python boolean capitalization prior to JSON parsing
        raw_json_clean = raw_json.replace(": True", ": true").replace(": False", ": false")
        parsed = json.loads(raw_json_clean)
        
        combined_thoughts = f"{parsed.get('thoughts', '')}\n\nPlan: {parsed.get('plan', 'No response.')}"
        
        return {
            "thoughts": combined_thoughts.strip(),
            "will_respond": bool(parsed.get("will_respond", True)),
            "thought_feedback": "", 
            "draft_feedback": ""
        }
    except Exception:
        return {
            "thoughts": response.content, 
            "will_respond": True,
            "thought_feedback": "", 
            "draft_feedback": ""
        }