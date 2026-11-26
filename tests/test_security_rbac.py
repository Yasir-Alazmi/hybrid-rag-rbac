"""Unit tests for Role-Based Access Control and multi-tenant document isolation."""

import pytest

from src.core.security import AccessControlManager, ClassificationLevel
from src.ingestion.chunker import TextChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_search import BM25Index
from src.retrieval.vector_store import DenseVectorStore


@pytest.fixture
def classified_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            chunk_id="pub_01", document_id="doc_pub",
            content="Welcome to our public company website and products overview.",
            chunk_index=0, token_count=10, classification="PUBLIC"
        ),
        TextChunk(
            chunk_id="int_01", document_id="doc_int",
            content="The internal corporate cafeteria hours are 8am to 4pm daily.",
            chunk_index=0, token_count=10, classification="INTERNAL"
        ),
        TextChunk(
            chunk_id="conf_01", document_id="doc_conf",
            content="Security encryption keys must rotate automatically every 90 days in the HSM.",
            chunk_index=0, token_count=12, classification="CONFIDENTIAL"
        ),
        TextChunk(
            chunk_id="rest_01", document_id="doc_rest",
            content="Executive payroll and executive bonus distribution records for Q4.",
            chunk_index=0, token_count=10, classification="RESTRICTED"
        )
    ]

def test_clearance_levels():
    assert AccessControlManager.can_access("guest", "PUBLIC") is True
    assert AccessControlManager.can_access("guest", "INTERNAL") is False
    assert AccessControlManager.can_access("guest", "RESTRICTED") is False

    assert AccessControlManager.can_access("employee", "PUBLIC") is True
    assert AccessControlManager.can_access("employee", "INTERNAL") is True
    assert AccessControlManager.can_access("employee", "CONFIDENTIAL") is False

    assert AccessControlManager.can_access("admin", "RESTRICTED") is True

def test_rbac_retrieval_isolation(classified_chunks: list[TextChunk]):
    bm25 = BM25Index()
    bm25.index_chunks(classified_chunks)
    vec = DenseVectorStore(dimension=128)
    vec.index_chunks(classified_chunks)
    retriever = HybridRetriever(bm25, vec)

    # Guest user searching for security keys
    guest_results = retriever.search("security encryption keys HSM", user_role="guest")
    # Guest should get ZERO confidential chunks
    assert not any(c.classification in ["CONFIDENTIAL", "RESTRICTED"] for c, _ in guest_results)

    # Admin user searching for security keys
    admin_results = retriever.search("security encryption keys HSM", user_role="admin")
    assert any(c.chunk_id == "conf_01" for c, _ in admin_results)

def test_unknown_role_fails_closed_to_public():
    # Unknown, empty or missing roles get the lowest clearance (PUBLIC)
    assert AccessControlManager.get_user_clearance("unknown_external_agent") == ClassificationLevel.PUBLIC
    assert AccessControlManager.get_user_clearance("") == ClassificationLevel.PUBLIC
    assert AccessControlManager.get_user_clearance(None) == ClassificationLevel.PUBLIC
    assert AccessControlManager.can_access("unknown_external_agent", "RESTRICTED") is False
    assert AccessControlManager.can_access("unknown_external_agent", "INTERNAL") is False
    assert AccessControlManager.can_access("unknown_external_agent", "PUBLIC") is True
    assert AccessControlManager.can_access(None, "INTERNAL") is False


def test_unknown_document_classification_is_denied():
    # An unknown or missing label must never be readable, not even by RESTRICTED
    assert AccessControlManager.can_access("admin", "TOP_SECRET_TYPO") is False
    assert AccessControlManager.can_access("admin", "") is False
    assert AccessControlManager.can_access_level(ClassificationLevel.RESTRICTED, None) is False


def test_search_without_role_or_clearance_fails_closed(classified_chunks: list[TextChunk]):
    bm25 = BM25Index()
    bm25.index_chunks(classified_chunks)
    vec = DenseVectorStore(dimension=128)
    vec.index_chunks(classified_chunks)
    retriever = HybridRetriever(bm25, vec)

    for kwargs in ({}, {"user_role": ""}, {"user_role": None}):
        results = retriever.search("internal cafeteria hours security keys payroll", **kwargs)
        assert {c.classification for c, _ in results} <= {"PUBLIC"}
        assert {cid for cid, _ in bm25.search("cafeteria hours", **kwargs)} <= {"pub_01"}
        assert {cid for cid, _ in vec.search("cafeteria hours", **kwargs)} <= {"pub_01"}
    # Sanity: the INTERNAL chunk is found once a clearance is given
    found = bm25.search("cafeteria hours", user_clearance=ClassificationLevel.INTERNAL)
    assert "int_01" in {cid for cid, _ in found}


def test_search_rejects_role_and_clearance_together(classified_chunks: list[TextChunk]):
    bm25 = BM25Index()
    bm25.index_chunks(classified_chunks)
    with pytest.raises(ValueError):
        bm25.search("payroll", user_role="guest", user_clearance=ClassificationLevel.RESTRICTED)


def test_chunk_with_unknown_classification_never_retrieved():
    chunk = TextChunk(
        chunk_id="odd_01", document_id="doc_odd",
        content="Mislabelled payroll ledger entries.",
        chunk_index=0, token_count=4, classification="SECRETISH"
    )
    bm25 = BM25Index()
    bm25.index_chunks([chunk])
    vec = DenseVectorStore(dimension=128)
    vec.index_chunks([chunk])
    top = ClassificationLevel.RESTRICTED
    assert bm25.search("payroll ledger", user_clearance=top) == []
    assert vec.search("payroll ledger", user_clearance=top) == []


def test_demo_tokens_disabled_in_production(monkeypatch):
    from fastapi import HTTPException

    from src.core import security

    monkeypatch.setattr(security.settings, "app_env", "production")
    monkeypatch.setattr(security.settings, "api_tokens_json", '{"prod-token-1": ["manager", 2]}')
    registry = AccessControlManager.get_token_registry()
    assert not set(security.DEFAULT_TOKEN_REGISTRY) & set(registry)
    assert registry["prod-token-1"] == ("manager", ClassificationLevel.CONFIDENTIAL)
    with pytest.raises(HTTPException) as exc:
        AccessControlManager.resolve_bearer_identity("Bearer token-ciso-root")
    assert exc.value.status_code == 401
    assert AccessControlManager.resolve_bearer_identity("Bearer prod-token-1")[1] == 2


def test_demo_tokens_available_outside_production(monkeypatch):
    from src.core import security

    monkeypatch.setattr(security.settings, "app_env", "development")
    monkeypatch.setattr(security.settings, "api_tokens_json", None)
    role, clearance = AccessControlManager.resolve_bearer_identity("Bearer token-ciso-root")
    assert (role, clearance) == ("ciso", ClassificationLevel.RESTRICTED)


def test_jwt_explicit_public_clearance_is_kept():
    from src.core.security import create_jwt_token, verify_jwt_token

    token = create_jwt_token("svc", role="admin", clearance=ClassificationLevel.PUBLIC)
    assert verify_jwt_token(token) == ("admin", ClassificationLevel.PUBLIC)


def test_jwt_invalid_clearance_claim_is_401():
    import jwt as pyjwt
    from fastapi import HTTPException

    from src.core.config import settings
    from src.core.security import verify_jwt_token

    for bad in (99, "3", True):
        token = pyjwt.encode(
            {"sub": "x", "role": "admin", "clearance": bad, "exp": 4102444800},
            settings.jwt_secret_key, algorithm=settings.jwt_algorithm,
        )
        with pytest.raises(HTTPException) as exc:
            verify_jwt_token(token)
        assert exc.value.status_code == 401

def test_hierarchical_clearance_inheritance():
    # RESTRICTED clearance inherits access to ALL lower levels
    assert AccessControlManager.can_access("executive", "PUBLIC") is True
    assert AccessControlManager.can_access("executive", "INTERNAL") is True
    assert AccessControlManager.can_access("executive", "CONFIDENTIAL") is True
    assert AccessControlManager.can_access("executive", "RESTRICTED") is True


def test_fail_fast_production_security_validation():
    from src.core.config import Settings

    # Insecure default secret in production environment must fail fast
    prod_insecure = Settings(app_env="production")
    with pytest.raises(ValueError) as excinfo:
        prod_insecure.validate_production_security()
    assert "FATAL SECURITY VIOLATION" in str(excinfo.value)

    # An empty secret is rejected too
    with pytest.raises(ValueError):
        Settings(app_env="production", jwt_secret_key="  ").validate_production_security()


def test_valid_production_security_validation():
    from src.core.config import Settings

    # Secure custom key in production environment succeeds
    prod_secure = Settings(
        app_env="production",
        jwt_secret_key="a-truly-cryptographic-and-secure-high-entropy-jwt-secret-key-12345"
    )
    # Should not raise
    prod_secure.validate_production_security()

