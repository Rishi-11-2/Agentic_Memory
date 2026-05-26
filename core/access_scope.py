"""PII-safe access scoping for multi-tenant Agentic Memory records."""

from __future__ import annotations

import hashlib

from contextvars import ContextVar

from pydantic import BaseModel, Field

# Thread/Async-safe context variable for the active request's scope hash
current_scope_hash: ContextVar[str | None] = ContextVar("current_scope_hash", default=None)



def compute_scope_hash(application_id: str, tenant_id: str, user_id: str) -> str:
    """Compute sha256(application_id || tenant_id || user_id) without storing raw PII."""
    material = f"{application_id}{tenant_id}{user_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


class AccessScope(BaseModel):
    """Represent the raw API scope while exposing only a durable scope hash to storage."""

    application_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)

    @property
    def scope_hash(self) -> str:
        """Return the hashed storage scope for all memory tables and RLS policies."""
        return compute_scope_hash(self.application_id, self.tenant_id, self.user_id)
