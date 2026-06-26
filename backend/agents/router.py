import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.state import AgentState

def get_llm():
    """Helper to initialize the LLM using env variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")
    
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=api_base,
        model_name=model_name,
        temperature=0.0
    )

def router_node(state: AgentState) -> dict:
    """
    Evaluates the user's query for ambiguity.
    Routes to clarification if ambiguous, or continues the research flow if clear.
    """
    # If the user has already answered a clarifying question, bypass the check
    if state.get("clarification_answers"):
        return {
            "clarification_needed": False,
            "clarification_question": None
        }
        
    query = state["query"]
    
    # We construct a prompt for the LLM to analyze the user's query for ambiguity.
    system_prompt = """You are the Orchestrator/Router of a scientific research knowledge graph system.
Your job is to analyze the user's input and determine if it is ambiguous.
Strictly flag the input as ambiguous if:
1. The user asks to "compare papers" or "compare these two" without listing specific papers or topics.
2. The user asks to "find good papers" or "retrieve papers" without specifying the research topic/domain.
3. The user asks to "give me a research idea" or "suggest a research gap" without any context or domain details.
4. The user asks for a "citation path" without specifying either the start paper, end paper, or concepts.

If you detect ambiguity, set "ambiguous" to true and provide a helpful, polite "clarifying_question".
If the query is clear, set "ambiguous" to false and "clarifying_question" to null.

You must respond with a JSON object in this format:
{
  "ambiguous": true/false,
  "clarifying_question": "polite question here if ambiguous, otherwise null"
}
"""
    
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query: {query}")
    ]
    
    try:
        response = llm.invoke(messages, response_format={"type": "json_object"})
        result = json.loads(response.content)
    except Exception as e:
        print(f"[Router Agent] Error calling LLM: {e}. Falling back to default routing.")
        # Fallback in case of json or LLM error
        result = {"ambiguous": False, "clarifying_question": None}

    return {
        "clarification_needed": result.get("ambiguous", False),
        "clarification_question": result.get("clarifying_question"),
    }
