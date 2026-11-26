"""Backward-compatible import path. The re-ranker lives in ``src.reranker.heuristic``.

``CrossEncoderReranker`` is a historical name kept as an alias; the class is a lexical
heuristic, not a trained cross-encoder.
"""

from src.reranker.heuristic import HeuristicReranker

CrossEncoderReranker = HeuristicReranker

__all__ = ["CrossEncoderReranker", "HeuristicReranker"]
