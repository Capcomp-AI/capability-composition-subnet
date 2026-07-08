"""Independent verification of published evaluation records.

The evaluation engine is centralised, so its output is only as trustworthy as
the checks anyone can run against it. These are those checks: they need the
published reports, no hidden data, no GPU, and no cooperation from the operator
beyond a readable API.
"""

from capability_subnet.audit.verify import (
    AuditResult,
    Finding,
    audit_window,
    recompute_qualified_score,
    verify_report,
    verify_weight_vector,
)

__all__ = [
    "AuditResult",
    "Finding",
    "audit_window",
    "recompute_qualified_score",
    "verify_report",
    "verify_weight_vector",
]
