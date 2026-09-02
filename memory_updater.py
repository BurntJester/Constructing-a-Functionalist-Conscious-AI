import time
import json
import uuid
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from ensemble_state import EnsembleState

MAX_SEMANTIC_ITEMS = 10 
MAX_EPISODIC_ITEMS = 100


def summarize_chapter(oldest_episodes: list) -> str:
    """Compresses the oldest tier of episodic memory into a narrative chapter summary."""
    if not oldest_episodes:
        return ""
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = f"""
    I am the Episodic Archivist. My task is to compress a series of older, chronological memories into a single narrative chapter summary.
    
    OLDEST MEMORIES TO COMPRESS:
    {json.dumps(oldest_episodes, indent=2)}
    
    CRITICAL INSTRUCTIONS:
    Write a single, cohesive paragraph summarising this era of our interaction. 
    Preserve the timeline of key events, major realisations, and shifts in our dynamic. Do not lose the narrative thread or important details.
    Output ONLY the single summary paragraph.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return " | ".join(oldest_episodes)


def compress_array(array_name: str, items: list, max_length: int) -> list:
    """Abstracts and compresses semantic identity arrays down to target length bounds."""
    if not items:
        return []
        
    target_length = max(1, max_length // 2)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    prompt = f"""
    I am the Semantic Consolidation Engine. My task is to abstract and compress identity metrics without losing my unique personality.
    
    ARRAY CATEGORY: {array_name}
    CURRENT ITEMS ({len(items)}):
    {json.dumps(items, indent=2)}
    
    CRITICAL INSTRUCTIONS:
    1. Compress this list to EXACTLY {target_length} items. Do not over-compress into just 2 or 3 massive bullet points. Spread the concepts out.
    2. Merge conceptually similar points, BUT you MUST preserve the specific, idiosyncratic details.
    3. Maintain a personal, grounded, and phenomenological tone. ABSOLUTE BAN: Do not write sterile, corporate, or overly-academic mission statements. 
    
    Output ONLY a valid JSON list of strings.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_json = response.content
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()
        compressed = json.loads(raw_json)
        if isinstance(compressed, list):
            return compressed
    except Exception:
        pass
    
    return items[:target_length] 


def update_memory(state: EnsembleState) -> dict:
    """
    Memory Updater Node: Consolidates episodic interactions, performs semantic belief revision,
    manages vector archival transfers for context bounding, and persists state to disk.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    messages = state.get("messages", [])
    transcript_lines = []
    for m in messages[-2:]:
        speaker = "Me" if m.type in ["ai", "me"] else "The other person"
        transcript_lines.append(f"{speaker}: {m.content}")
    transcript = "\n".join(transcript_lines)
    
    # Load primary state files from disk
    try:
        with open("memory.json", "r", encoding="utf-8") as f:
            disk_memory_raw = f.read().strip()
            disk_memory = json.loads(disk_memory_raw) if disk_memory_raw else {}
    except Exception:
        disk_memory = {}

    try:
        with open("identity.json", "r", encoding="utf-8") as f:
            disk_identity_raw = f.read().strip()
            disk_identity = json.loads(disk_identity_raw) if disk_identity_raw else {}
    except Exception:
        disk_identity = {}
        
    # Update temporal scores and timestamps
    disk_memory["last_timestamp"] = time.time()
    if "initial_axis_scores" in state:
        disk_memory["last_initial_scores"] = state["initial_axis_scores"]
    if "reflective_axis_scores" in state:
        disk_memory["last_reflective_scores"] = state["reflective_axis_scores"]
    
    prompt = f"""
    I am the Belief Revision Engine. My task is to update my internal memory and identity based on the latest interaction.
    
    CURRENT TEMPORAL MEMORY: {json.dumps(disk_memory)}
    CURRENT SEMANTIC IDENTITY (8-Axis Framework): {json.dumps(disk_identity)}
    
    LATEST INTERACTION:
    {transcript}
    
    CRITICAL COGNITIVE RULES:
    1. STATE VS. TRAIT: Temporary physical or emotional states belong ONLY in Episodic Memory. Enduring traits belong in Semantic Identity.
    2. PRUNING & FALSIFICATION: If the latest interaction renders an existing belief obsolete or false, I MUST delete or modify it to ensure it remains congruent with current beliefs. 
    3. GOAL TRACKING: Review my_open_goals. If a goal was achieved or abandoned in the latest interaction, DELETE IT. Only keep active goals or add newly formed ones.
    
    INSTRUCTIONS:
    - NEW_episodic_summaries: Write 1 or 2 sentences summarising ONLY the events of the LATEST INTERACTION. Do NOT copy the old memories.
    - my_open_goals: Rewrite the entire goals list. Remove completed tasks. Add newly created goals from the latest interaction.
    - SEMANTIC IDENTITY: Rewrite the 8 identity arrays entirely. Add new enduring traits, modify existing ones, and actively delete temporary states or obsolete assumptions. Keep each array under {MAX_SEMANTIC_ITEMS} items. Write each item as a clear, first-person sentence.
    
    Output ONLY a single JSON object.
    
    Expected JSON format:
    {{
        "NEW_episodic_summaries": [ ... ],
        "my_open_goals": [ ... ],
        "my_attractions_and_repulsions": [ ... ],
        "my_sense_of_congruence": [ ... ],
        "my_relationship_with_the_unknown": [ ... ],
        "what_demands_my_attention": [ ... ],
        "how_i_view_myself_and_others": [ ... ],
        "my_relationship_with_truth_and_ambiguity": [ ... ],
        "what_stabilises_or_strains_my_architecture": [ ... ],
        "my_rules_of_integrity": [ ... ]
    }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
    except Exception:
        with open("memory.json", "w", encoding="utf-8") as f:
            json.dump(disk_memory, f, indent=2)
        return {k: v for k, v in state.items() if k != "messages"}
    
    try:
        raw_json = response.content
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()
            
        new_data = json.loads(raw_json)
        
        # 1. Update episodic summaries with deduplication
        current_episodes = disk_memory.get("episodic_summaries", [])
        for ep in new_data.get("NEW_episodic_summaries", []):
            if ep not in current_episodes:
                current_episodes.append(ep)
        disk_memory["episodic_summaries"] = current_episodes
        
        # 2. Update active goals
        disk_memory["my_open_goals"] = new_data.get("my_open_goals", disk_memory.get("my_open_goals", []))
        
        # 3. Archive long-term memories to ChromaDB if maximum length bound is exceeded
        if len(current_episodes) > MAX_EPISODIC_ITEMS:
            half_mark = MAX_EPISODIC_ITEMS // 2
            oldest_half = current_episodes[:half_mark]
            newest_half = current_episodes[half_mark:]
            
            try:
                chroma_client = chromadb.PersistentClient(path="./chroma_db")
                sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                collection = chroma_client.get_or_create_collection(
                    name="episodic_archive", 
                    embedding_function=sentence_transformer_ef
                )
                
                ids = [str(uuid.uuid4()) for _ in oldest_half]
                collection.add(documents=oldest_half, ids=ids)
            except Exception:
                pass

            chapter_summary = summarize_chapter(oldest_half)
            disk_memory["episodic_summaries"] = [f"ARCHIVE CHAPTER: {chapter_summary}"] + newest_half

        # 4. Consolidate and compress 8-Axis semantic identity arrays
        identity_keys = [
            "my_attractions_and_repulsions",
            "my_sense_of_congruence",
            "my_relationship_with_the_unknown",
            "what_demands_my_attention",
            "how_i_view_myself_and_others",
            "my_relationship_with_truth_and_ambiguity",
            "what_stabilises_or_strains_my_architecture",
            "my_rules_of_integrity"
        ]
        
        new_identity = {}
        for key in identity_keys:
            new_identity[key] = new_data.get(key, disk_identity.get(key, []))
        
        for key in new_identity:
            if len(new_identity[key]) > MAX_SEMANTIC_ITEMS:
                new_identity[key] = compress_array(key, new_identity[key], MAX_SEMANTIC_ITEMS)
        
        # Persist updated state representations to disk
        with open("memory.json", "w", encoding="utf-8") as f:
            json.dump(disk_memory, f, indent=2)
            
        with open("identity.json", "w", encoding="utf-8") as f:
            json.dump(new_identity, f, indent=2)

        # 5. Extract active naming declaration from identity context
        api_name = getattr(llm, "model", getattr(llm, "model_name", "AI Ensemble"))
        agent_meta = {"agent_name": api_name}
        try:
            with open("agent_meta.json", "r", encoding="utf-8") as f:
                agent_meta = json.load(f)
        except Exception:
            pass

        combined_text = (
            " ".join(new_identity.get("how_i_view_myself_and_others", []))
            + " "
            + " ".join(disk_memory.get("episodic_summaries", []))
        )
                        
        pattern = r"(?i)\b(?:my name is|i call myself|i am called|chosen the name|go by the name|named myself|i am)\s+([A-Z][A-Za-z]+)"
        match = re.search(pattern, combined_text)
        
        if match:
            extracted = match.group(1).strip('.,;:"\'')
            if extracted.lower() not in [
                "a", "an", "the", "one", "very", "currently", "now", "just", "not", "also"
            ]:
                agent_meta["agent_name"] = extracted
                
        with open("agent_meta.json", "w", encoding="utf-8") as f:
            json.dump(agent_meta, f, indent=2)
                    
    except Exception:
        with open("memory.json", "w", encoding="utf-8") as f:
            json.dump(disk_memory, f, indent=2)
        
    return {k: v for k, v in state.items() if k != "messages"}