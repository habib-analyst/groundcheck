"""groundcheck — a RAG agent that verifies its own citations.

Pipeline: retrieve context, generate an answer with citations, verify
that each cited passage actually supports the claim attached to it,
flag low-confidence answers, and fall back to a corpus-wide search to
repair or drop unsupported claims.

v2 adds a local NLI contradiction veto (groundcheck.nli) on top of the
v1 support score: a cited passage that contradicts its claim fails
verification even when lexical/embedding similarity is high.
"""

from .nli import NLIClassifier
from .pipeline import GroundCheckAgent
from .verify import SupportScorer

__all__ = ["GroundCheckAgent", "SupportScorer", "NLIClassifier"]
__version__ = "0.2.0"
