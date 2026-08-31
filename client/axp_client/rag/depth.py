"""Server-owned policy for the single progressive retrieval pass."""
from dataclasses import dataclass
from math import floor

DEPTH_MULTIPLIER = {0: 1.0, 1: 1.5}


def validate_search_depth(value):
    if type(value) is not int or value not in DEPTH_MULTIPLIER:
        raise ValueError("invalid_search_depth")
    return value


@dataclass(frozen=True)
class DepthPolicy:
    search_depth: int
    input_multiplier: float
    output_multiplier: float
    target_evidence_tokens: int | None
    target_answer_tokens: int
    retrieval_limit: int
    candidate_depth: int
    seed_limit: int
    neighbor_radius: int


def depth_policy(depth, *, evidence_tokens, answer_tokens):
    multiplier = DEPTH_MULTIPLIER[validate_search_depth(depth)]
    return DepthPolicy(depth, multiplier, multiplier,
        None if evidence_tokens is None else floor(evidence_tokens * multiplier),
        floor(answer_tokens * multiplier), floor(24 * multiplier), floor(100 * multiplier),
        2 if depth == 0 else 3, 1 if depth == 0 else 2)
