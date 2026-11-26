"""Role-Based Access Control (RBAC), JWT Verification, and Bearer Token Security."""

import json
import time
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import jwt
from fastapi import HTTPException, status

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class ClassificationLevel(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


ROLE_CLEARANCE_MAP: Dict[str, ClassificationLevel] = {
    "anonymous": ClassificationLevel.PUBLIC,
    "guest": ClassificationLevel.PUBLIC,
    "employee": ClassificationLevel.INTERNAL,
    "internal": ClassificationLevel.INTERNAL,
    "contractor": ClassificationLevel.INTERNAL,
    "compliance_officer": ClassificationLevel.CONFIDENTIAL,
    "manager": ClassificationLevel.CONFIDENTIAL,
    "admin": ClassificationLevel.RESTRICTED,
    "ciso": ClassificationLevel.RESTRICTED,
    "executive": ClassificationLevel.RESTRICTED,
}

# Demo tokens for local development and tests only. They are never loaded when
# APP_ENV is production (see AccessControlManager.get_token_registry).
DEFAULT_TOKEN_REGISTRY: Dict[str, Tuple[str, ClassificationLevel]] = {
    "token-admin-restricted": ("admin", ClassificationLevel.RESTRICTED),
    "token-ciso-root": ("ciso", ClassificationLevel.RESTRICTED),
    "token-manager-confidential": ("manager", ClassificationLevel.CONFIDENTIAL),
    "token-compliance-officer": ("compliance_officer", ClassificationLevel.CONFIDENTIAL),
    "token-employee-internal": ("employee", ClassificationLevel.INTERNAL),
    "token-guest-public": ("guest", ClassificationLevel.PUBLIC),
}


def create_jwt_token(
    subject: str,
    role: str,
    clearance: Optional[ClassificationLevel] = None,
    expires_in_seconds: int = 3600,
    secret_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> str:
    """Generate signed HMAC-SHA256 JWT for testing and machine-to-machine auth."""
    secret = secret_key or settings.jwt_secret_key
    algo = algorithm or settings.jwt_algorithm
    if clearance is not None:
        assigned_clearance = ClassificationLevel(clearance)
    else:
        assigned_clearance = AccessControlManager.get_user_clearance(role)

    payload: Dict[str, Any] = {
        "sub": subject,
        "role": role,
        "clearance": int(assigned_clearance),
        "iat": int(time.time()),
        "exp": int(time.time() + expires_in_seconds),
        "iss": "hybrid-rag-rbac",
    }
    return jwt.encode(payload, secret, algorithm=algo)


def verify_jwt_token(
    token: str,
    secret_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> Tuple[str, ClassificationLevel]:
    """Verify cryptographic signature, expiration, and claims of a JWT."""
    secret = secret_key or settings.jwt_secret_key
    algo = algorithm or settings.jwt_algorithm

    try:
        decoded = jwt.decode(
            token,
            secret,
            algorithms=[algo],
            options={"require": ["exp", "sub", "role"]},
        )
        role = str(decoded["role"]).lower().strip()
        clearance_val = decoded.get("clearance")
        if clearance_val is None:
            # No explicit claim: derive from the role (unknown roles get PUBLIC).
            return (role, AccessControlManager.get_user_clearance(role))
        if isinstance(clearance_val, bool) or not isinstance(clearance_val, int):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed: JWT clearance claim must be an integer.",
            )
        try:
            clearance = ClassificationLevel(clearance_val)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed: JWT clearance claim is out of range.",
            )
        return (role, clearance)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: JWT signature has expired.",
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: Invalid JWT token ({type(e).__name__}).",
        )


class AccessControlManager:
    """Enforces zero-trust identity resolution, cryptographic token verification, and RBAC isolation."""

    @classmethod
    def get_token_registry(cls) -> Dict[str, Tuple[str, ClassificationLevel]]:
        """Resolve the active pre-shared token registry.

        The demo DEFAULT_TOKEN_REGISTRY is included only outside production; in
        production only tokens configured through API_TOKENS_JSON are accepted.
        """
        registry: Dict[str, Tuple[str, ClassificationLevel]] = {}
        if not settings.is_production():
            registry.update(DEFAULT_TOKEN_REGISTRY)
        if settings.api_tokens_json:
            try:
                custom_tokens = json.loads(settings.api_tokens_json)
                for tok, (r, c_val) in custom_tokens.items():
                    registry[tok] = (r, ClassificationLevel(c_val))
            except Exception as e:
                logger.warning("Failed to parse custom api_tokens_json: %s", e)
        return registry

    @staticmethod
    def get_user_clearance(role: Optional[str]) -> ClassificationLevel:
        """Resolve a role string to its clearance level.

        Fail closed: an unknown, empty or missing role gets PUBLIC (the lowest level).
        """
        if not role:
            return ClassificationLevel.PUBLIC
        normalized = str(role).lower().strip()
        return ROLE_CLEARANCE_MAP.get(normalized, ClassificationLevel.PUBLIC)

    @staticmethod
    def parse_classification(document_classification: Optional[str]) -> Optional[ClassificationLevel]:
        """Parse a document classification label; returns None if it is not a known level."""
        if not document_classification:
            return None
        try:
            return ClassificationLevel[str(document_classification).upper().strip()]
        except KeyError:
            return None

    @classmethod
    def can_access_level(
        cls, user_clearance: ClassificationLevel, document_classification: Optional[str]
    ) -> bool:
        """Check a clearance level against a document classification.

        Fail closed: a missing or unknown document classification is never readable.
        """
        doc_level = cls.parse_classification(document_classification)
        if doc_level is None:
            return False
        return int(user_clearance) >= int(doc_level)

    @classmethod
    def resolve_search_clearance(
        cls,
        user_clearance: Optional[ClassificationLevel] = None,
        user_role: Optional[str] = None,
    ) -> ClassificationLevel:
        """Pick the single clearance a retrieval call filters by.

        Pass either an explicit clearance (what the API does) or a role. Passing both is
        rejected so the two can never disagree; passing neither fails closed to PUBLIC.
        """
        if user_clearance is not None and user_role is not None:
            raise ValueError("Pass user_clearance or user_role, not both.")
        if user_clearance is not None:
            return ClassificationLevel(user_clearance)
        return cls.get_user_clearance(user_role)

    @classmethod
    def can_access(cls, user_role: Optional[str], document_classification: Optional[str]) -> bool:
        """Verify if a user's role has sufficient clearance to read a classified document."""
        return cls.can_access_level(cls.get_user_clearance(user_role or ""), document_classification)

    @classmethod
    def resolve_bearer_identity(
        cls,
        auth_header: Optional[str] = None,
        require_auth: bool = False,
    ) -> Tuple[str, ClassificationLevel]:
        """Cryptographically resolve user identity and clearance strictly from HTTP Authorization header.

        Zero-Trust Policy:
        - If no Authorization header is provided:
            * On public endpoints: assigns ('guest', PUBLIC).
            * On protected endpoints (require_auth=True): raises HTTP 401 Unauthorized.
        - If Authorization header is provided with malformed or unrecognized token:
            * Strictly raises HTTP 401 Unauthorized (never downgrades invalid credentials to guest).
        - Supports both standard HMAC-SHA256 JWT tokens and pre-shared API bearer tokens.
        """
        if not auth_header:
            if require_auth:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required: Missing Authorization Bearer header.",
                )
            return ("guest", ClassificationLevel.PUBLIC)

        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Malformed authorization header. Expected 'Bearer <token>'.",
            )

        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed: Empty Bearer token provided.",
            )

        # Check if token is a JWT (contains three base64 segments separated by dots)
        if token.count(".") == 2:
            return verify_jwt_token(token)

        # Pre-shared API Bearer token lookup
        registry = cls.get_token_registry()
        if token in registry:
            return registry[token]

        # Explicitly supplied unrecognized token must be rejected with 401
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: Invalid or unrecognized Bearer token.",
        )

    @classmethod
    def enforce_clearance(
        cls,
        user_clearance: ClassificationLevel,
        min_clearance: ClassificationLevel,
        operation_name: str = "administrative operation",
    ) -> None:
        """Enforce minimum required clearance level, raising HTTP 403 Forbidden on violation."""
        if user_clearance < min_clearance:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access forbidden: Clearance level '{user_clearance.name}' is insufficient "
                    f"for {operation_name}. Minimum required: '{min_clearance.name}'."
                ),
            )
