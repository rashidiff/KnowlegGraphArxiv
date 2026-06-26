from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseDatabase(ABC):
    @abstractmethod
    def init_db(self) -> None:
        """Initialize database schemas, tables, and indices."""
        pass

    @abstractmethod
    def insert_papers(self, papers: List[Dict[str, Any]]) -> None:
        """Insert a batch of papers with full metadata, references, and topics."""
        pass

    @abstractmethod
    def search_papers(self, 
                      query_embedding: List[float], 
                      keywords: List[str], 
                      topic: Optional[str] = None, 
                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using weighted ranking formula:
        0.45 * similarity + 0.25 * citation_score + 0.15 * centrality + 0.10 * recency + 0.05 * venue_quality
        """
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full details of a single paper."""
        pass

    @abstractmethod
    def get_citation_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        """Find the shortest citation path between two papers."""
        pass

    @abstractmethod
    def get_graph_data(self, focus_paper_ids: Optional[List[str]] = None, max_nodes: int = 150) -> Dict[str, Any]:
        """Get graph nodes and edges for the D3 interactive visualization."""
        pass

    @abstractmethod
    def get_graph_metrics(self) -> Dict[str, Any]:
        """Compute global and local graph centrality, bridge nodes, and cluster metrics."""
        pass

    @abstractmethod
    def get_papers_by_topic(self, topic_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all papers associated with a specific topic."""
        pass

    @abstractmethod
    def get_all_topics(self) -> List[str]:
        """Get all available paper topics."""
        pass

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Look up a cached paper by arXiv ID. Default: returns None (subclasses override)."""
        return None
