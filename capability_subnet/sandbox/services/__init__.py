"""Sandbox tool services.

The inventory simulator and the safety rule engine, packaged as standalone
containers for the isolated deployment. They wrap exactly the same logic the
in-process tools use, so a candidate scores identically either way — what changes
is where the state lives, not what the rules are.

The scorer is deliberately not among them. It holds the ground truth and runs in
the engine process, outside every network the agent container can see.
"""

from capability_subnet.sandbox.services import inventory_api, safety_api

__all__ = ["inventory_api", "safety_api"]
