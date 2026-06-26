from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    # Core User Inputs
    query: str
    messages: List[AnyMessage]
    
    # Ambiguity / Clarification State
    clarification_needed: bool
    clarification_question: Optional[str]
    clarification_answers: List[Dict[str, str]] # [{'question': '...', 'answer': '...'}]
    
    # Agent Outputs
    research_plan: Optional[str]
    retrieved_papers: List[Dict[str, Any]]
    graph_context: Dict[str, Any]
    reading_path: Optional[str]
    analysis: Optional[str]
    research_ideas: Optional[str]
    verification_report: Dict[str, Any]
    final_response: Optional[str]
