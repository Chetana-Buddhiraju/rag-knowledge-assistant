"""Document-level access control (Step 5, Q4): department ACL enforced as a
pre-filter at the retrieval layer (see retrieval/retriever.py ->
vectorstore.search(department_filter=...)), not as a post-hoc check on the
generated answer. That distinction matters: filtering after generation would
mean an HR document could still leak into the prompt an Engineering user's
question is answered from, even if the final text is later redacted.
"""
from __future__ import annotations

from .config import ROLE_DEPARTMENT_ACCESS


def get_allowed_departments(role: str) -> list[str]:
    role = role.lower().strip()
    if role not in ROLE_DEPARTMENT_ACCESS:
        raise ValueError(f"Unknown role {role!r}. Known roles: {sorted(ROLE_DEPARTMENT_ACCESS)}")
    return ROLE_DEPARTMENT_ACCESS[role]
