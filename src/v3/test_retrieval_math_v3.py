"""Offline checks for lexical ranking and token normalization."""

from .retrieval_v3 import _comparison_parts, bm25_scores, tokenize


documents = ["annual leave approval and carryover", "device encryption and security"]
scores = bm25_scores("leave carryover", documents)
assert scores[0] > scores[1]
assert tokenize("Employees' policies") == []
assert _comparison_parts("Compare benefits enrollment and IT access requests") == [
    "benefits enrollment", "IT access requests"
]
print("retrieval math assertions: ok")
