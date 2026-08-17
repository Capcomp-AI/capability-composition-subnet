"""LoRA Merger — the Capability Composition Subnet.

Miners submit one declarative merge recipe over a frozen, certified pool of LoRA
adapters. A continuous champion-challenge engine reconstructs each recipe
deterministically, serves it, runs it against hidden instances of a real agentic
workflow, and compares it head-to-head with the reigning champion and a set of
permanent reference baselines. Every validator does this itself, on its own GPU,
and sets weights from what it measured.
"""

__version__ = "2.7.0"

__all__ = ["__version__"]
