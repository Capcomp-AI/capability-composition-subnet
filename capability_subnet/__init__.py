"""LoRA Merger — the Capability Composition Subnet.

Miners submit one declarative merge recipe over a frozen, certified pool of LoRA
adapters. A continuous champion-challenge engine reconstructs each recipe
deterministically, serves it, runs it against hidden instances of a real agentic
workflow, and compares it head-to-head with the reigning champion and a set of
permanent reference baselines. Validators are thin: they fetch the signed weight
vector the engine publishes and set it on-chain.
"""

__version__ = "2.2.0"

_version_split = __version__.split(".")
#: Integer spec version submitted alongside weights. Bump on any change that
#: alters how a recipe is reconstructed or scored.
__spec_version__ = (
    (1000 * int(_version_split[0])) + (10 * int(_version_split[1])) + (1 * int(_version_split[2]))
)

__all__ = ["__version__", "__spec_version__"]
