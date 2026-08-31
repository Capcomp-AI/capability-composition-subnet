"""Independent verification of published evaluation records.

Validators measure for themselves, so a published run is something to check
rather than believe. These are those checks: they need the published reports,
no hidden data, no GPU, and no cooperation from whoever produced them beyond a
readable API.
"""

from capability_subnet.audit.verify import (
    AuditResult,
    Finding,
    audit_run,
    recompute_qualified_score,
    verify_report,
    verify_weight_vector,
)

__all__ = [
    "AuditResult",
    "Finding",
    "audit_run",
    "recompute_qualified_score",
    "verify_report",
    "verify_weight_vector",
]
