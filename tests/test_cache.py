"""Unit tests for semantic query cache including LRU eviction and RBAC role scoping."""

import numpy as np

from src.cache.semantic_cache import SemanticCache


def test_semantic_cache_hit_and_miss():
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=60, max_entries=10)

    vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec2_close = np.array([0.98, 0.05, 0.0], dtype=np.float32)
    vec2_close /= np.linalg.norm(vec2_close)
    vec3_different = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    miss = cache.lookup("What is the SLA?", vec1, user_clearance=1)
    assert miss is None
    assert cache.misses == 1

    cache.store("What is the SLA?", vec1, "99.95% uptime", [{"doc": "sla"}], clearance_level=1)

    hit = cache.lookup("Can you tell me the SLA?", vec2_close, user_clearance=1)
    assert hit is not None
    response, citations, score = hit
    assert response == "99.95% uptime"
    assert score >= 0.90
    assert cache.hits == 1

    miss2 = cache.lookup("How to cook pasta?", vec3_different, user_clearance=1)
    assert miss2 is None
    assert cache.misses == 2

def test_role_scoped_cache_clearance_isolation():
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=60, max_entries=10)
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # 1. Admin stores a RESTRICTED answer (clearance_level=3)
    cache.store(
        query="What is the CEO executive compensation?",
        query_embedding=vec,
        response="Restricted board compensation details.",
        citations=[],
        clearance_level=3  # RESTRICTED
    )

    # 2. Employee (clearance_level=1) queries with same vector -> MUST BE DENIED!
    employee_hit = cache.lookup(
        query="What is the CEO executive compensation?",
        query_embedding=vec,
        user_clearance=1  # INTERNAL
    )
    assert employee_hit is None
    assert cache.clearance_denials == 1

    # 3. Executive/Admin (clearance_level=3) queries -> GRANTED!
    admin_hit = cache.lookup(
        query="What is the CEO executive compensation?",
        query_embedding=vec,
        user_clearance=3  # RESTRICTED
    )
    assert admin_hit is not None
    assert admin_hit[0] == "Restricted board compensation details."

def test_lru_cache_eviction():
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=60, max_entries=2)
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    cache.store("q1", vec, "a1", [], clearance_level=1)
    cache.store("q2", vec, "a2", [], clearance_level=1)
    assert len(cache.cache) == 2
    assert cache.evictions == 0

    cache.store("q3", vec, "a3", [], clearance_level=1)
    assert len(cache.cache) == 2
    assert cache.evictions == 1
    assert (1, "q1") not in cache.cache
    assert (1, "q2") in cache.cache
    assert (1, "q3") in cache.cache


def test_same_query_at_different_clearances_does_not_overwrite():
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=60, max_entries=10)
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    q = "What is the incident budget?"

    cache.store(q, vec, "full restricted answer", [{"doc": "r"}], clearance_level=3)
    cache.store(q, vec, "partial internal answer", [{"doc": "i"}], clearance_level=1)
    assert len(cache.cache) == 2

    assert cache.lookup(q, vec, user_clearance=3)[0] == "full restricted answer"
    assert cache.lookup(q, vec, user_clearance=1)[0] == "partial internal answer"


def test_higher_clearance_never_served_lower_clearance_answer():
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=60, max_entries=10)
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cache.store("q", vec, "internal-only answer", [], clearance_level=1)

    # The INTERNAL answer was built from fewer documents: a RESTRICTED user must miss
    assert cache.lookup("q", vec, user_clearance=3) is None
    # ...and a PUBLIC user must miss as well (insufficient clearance)
    assert cache.lookup("q", vec, user_clearance=0) is None
    assert cache.clearance_denials == 1  # only the PUBLIC lookup hit a higher-level entry


def test_closer_entry_at_other_clearance_does_not_shadow_valid_entry():
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=60, max_entries=10)
    exact = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    near = np.array([0.99, 0.141, 0.0], dtype=np.float32)
    near /= np.linalg.norm(near)

    cache.store("q-restricted", exact, "restricted answer", [], clearance_level=3)
    cache.store("q-internal", near, "internal answer", [], clearance_level=1)

    hit = cache.lookup("q", exact, user_clearance=1)
    assert hit is not None and hit[0] == "internal answer"


def test_invalidate_drops_entries_but_keeps_counters():
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=60, max_entries=10)
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cache.store("q", vec, "a", [], clearance_level=1)
    assert cache.lookup("q", vec, user_clearance=1) is not None

    cache.invalidate()
    assert len(cache.cache) == 0
    assert cache.hits == 1
    assert cache.lookup("q", vec, user_clearance=1) is None
