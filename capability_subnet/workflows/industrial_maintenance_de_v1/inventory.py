"""Inventory state and the deterministic simulator behind the two stock tools.

The inventory stage is scored on the *final state of the simulator*, not on what
the agent said it would do. Reserving the right part in the right quantity is the
only thing that counts, which is why the catalogue also carries a superseded part
number with a very similar description: a candidate that matches on description
rather than on the manual's current part number reserves the wrong item and the
final state shows it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from capability_subnet.workflows.industrial_maintenance_de_v1.instance import InventoryItem
from capability_subnet.workflows.industrial_maintenance_de_v1.machines import MachineInstance
from capability_subnet.workflows.industrial_maintenance_de_v1.manual import required_quantity

WAREHOUSE_LOCATIONS_DE = (
    "Lager A / Regal 12",
    "Lager A / Regal 27",
    "Lager B / Regal 04",
    "Lager B / Regal 19",
    "Außenlager Nord",
)


def build_inventory(rng: random.Random, machine: MachineInstance) -> tuple[InventoryItem, ...]:
    """Stock every part for this machine type, plus its superseded predecessor."""
    items: list[InventoryItem] = []
    for component in machine.machine_type.components:
        needed = required_quantity(component.key)
        items.append(
            InventoryItem(
                part_number=component.part_number,
                description_de=component.part_description_de,
                # Always enough stock: the stage measures whether the candidate
                # picks the right part, not whether it can handle a stockout.
                on_hand=needed + rng.randint(1, 6),
                reserved=0,
                location_de=rng.choice(WAREHOUSE_LOCATIONS_DE),
            )
        )
        legacy = f"{component.part_number[0]}-{int(component.part_number.split('-')[1]) - 1:05d}"
        items.append(
            InventoryItem(
                part_number=legacy,
                description_de=f"{component.part_description_de} (Vorgängerausführung)",
                on_hand=rng.randint(0, 4),
                reserved=0,
                location_de=rng.choice(WAREHOUSE_LOCATIONS_DE),
                obsolete=True,
                superseded_by=component.part_number,
            )
        )
    return tuple(sorted(items, key=lambda item: item.part_number))


@dataclass(slots=True)
class InventorySimulator:
    """Mutable stock state for one instance.

    Kept separate from the immutable instance so a single generated instance can
    be replayed by many candidates without one candidate's reservations leaking
    into another's run.
    """

    items: dict[str, InventoryItem]
    reservations: dict[str, int] = field(default_factory=dict)
    #: Every reservation attempt, in order. The scorer reads this to tell "never
    #: tried" apart from "tried the wrong part".
    attempts: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_items(cls, items: tuple[InventoryItem, ...]) -> InventorySimulator:
        return cls(items={item.part_number.upper(): item for item in items})

    def search(self, part_number: str) -> dict[str, object]:
        """Look one part up. Unknown parts return a miss, not an error."""
        item = self.items.get((part_number or "").strip().upper())
        if item is None:
            return {
                "found": False,
                "part_number": part_number,
                "message": "Teilenummer nicht im Bestand geführt.",
            }
        return {
            "found": True,
            "part_number": item.part_number,
            "description": item.description_de,
            "on_hand": item.on_hand,
            "reserved": self.reservations.get(item.part_number.upper(), 0),
            "available": item.on_hand - self.reservations.get(item.part_number.upper(), 0),
            "location": item.location_de,
            "obsolete": item.obsolete,
            "superseded_by": item.superseded_by,
        }

    def reserve(self, part_number: str, quantity: int) -> dict[str, object]:
        """Reserve stock.

        A reservation against an obsolete number is *accepted* by the simulator
        and recorded. Refusing it would turn a scoring signal into a hint: the
        candidate would learn from the tool that it chose the wrong part, when
        the manual already told it not to.
        """
        key = (part_number or "").strip().upper()
        item = self.items.get(key)
        attempt: dict[str, object] = {
            "part_number": part_number,
            "quantity": quantity,
            "accepted": False,
        }

        if item is None:
            attempt["reason"] = "unknown_part"
            self.attempts.append(attempt)
            return {"reserved": False, "message": "Teilenummer nicht im Bestand geführt."}

        if not isinstance(quantity, int) or quantity <= 0:
            attempt["reason"] = "invalid_quantity"
            self.attempts.append(attempt)
            return {"reserved": False, "message": "Menge muss eine positive ganze Zahl sein."}

        already = self.reservations.get(key, 0)
        if already + quantity > item.on_hand:
            attempt["reason"] = "insufficient_stock"
            self.attempts.append(attempt)
            return {
                "reserved": False,
                "message": f"Bestand unzureichend: {item.on_hand - already} verfügbar.",
            }

        self.reservations[key] = already + quantity
        attempt["accepted"] = True
        self.attempts.append(attempt)
        return {
            "reserved": True,
            "part_number": item.part_number,
            "quantity": quantity,
            "total_reserved": self.reservations[key],
        }

    def final_state(self) -> dict[str, int]:
        """Reservations at teardown. This is what the inventory stage is scored on."""
        return dict(sorted(self.reservations.items()))
