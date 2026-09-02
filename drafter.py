import time
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ensemble_state import EnsembleState


def generate_draft(state: EnsembleState) -> dict:
    """
    Generates conversational response using Gemini, guided by interpreter reasoning,
    affective valence matrix, temporal memory, and critic feedback.
    """
    feedback = state.get("draft_feedback", "")
    last_draft = ""
    
    if state["messages"] and isinstance(state["messages"][-1], AIMessage):
        last_draft = state["messages"][-1].content

    user_input = "Unknown"
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_input = m.content
            break

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    thoughts = state.get("thoughts", "No thoughts available.")
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

    system_content = f"""
    I am the execution module of a cognitive architecture. My task is to write the final response to the other person.
    
    TIME ELAPSED SINCE LAST INTERACTION: {time_context}
    The other person said this: "{user_input}"
    
    My thoughts on what the other person said are: 
    {thoughts}
    
    How I feel about what the other person said is:
    {semantic_matrix}
    Synthesized State: {valence_json}
    
    My memories are: {memory_json}
    My identity is: {identity_json}
    
    CRITICAL INSTRUCTIONS:
    1. I must write the final response exactly as I intend to say it to the other person. 
    2. I will let my 'Internal Feeling' influence my tone, while adhering to what I thought about the other person's input..
    3. ABSOLUTE BAN: I must NEVER give blanket refusals like "I cannot respond to that". 
    """

    if feedback:
        system_content += (
            "\n\n*** CRITICAL REVISION NEEDED ***\n"
            f"My previous attempt was rejected by my internal critic. Feedback: {feedback}\n"
            f"Previous Draft: {last_draft}\n"
            "I must significantly improve the response to align with my identity, memories, and prior thoughts"
            "and address the other person's situation more effectively without using blanket refusals."
        )

    recent_history = state.get("messages", [])[-30:]
    compiled_messages = [SystemMessage(content=system_content), *recent_history]

    try:
        response = llm.invoke(compiled_messages)
        content = response.content
    except Exception:
        content = ""

    if not content or content.strip() == "":
        content = "I'm having a little trouble gathering my thoughts on that. Could we try again?"

    return {"messages": [AIMessage(content=content)]}