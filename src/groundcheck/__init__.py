"""groundcheck — a RAG agent that verifies its own citations.

Pipeline: retrieve context, generate an answer with citations, verify
that each cited passage actually supports the claim attached to it,
flag low-confidence answers, and fall back to a corpus-wide search to
repair or drop unsupported claims.
"""

from .pipeline import GroundCheckAgent
from .verify import SupportScorer

__all__ = ["GroundCheckAgent", "SupportScorer"]
__version__ = "0.1.0"
