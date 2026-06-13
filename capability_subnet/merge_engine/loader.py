"""Adapter tensor sources.

The merge engine never reaches for a file path directly. It asks a source for
one named tensor at a time, which keeps peak memory bounded to the tensors of a
single projection and lets tests substitute small in-memory adapters for the
real 8B-scale pool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from capability_subnet.merge_engine.determinism import WORK_DTYPE


class AdapterLoadError(Exception):
    """Raised when an adapter tensor cannot be produced."""


@runtime_checkable
class AdapterSource(Protocol):
    """Anything that can hand the merge engine adapter tensors."""

    def has_adapter(self, adapter_id: str) -> bool: ...

    def get_tensor(self, adapter_id: str, key: str) -> torch.Tensor:
        """Return one tensor as float32 on CPU."""
        ...

    def keys(self, adapter_id: str) -> list[str]:
        """Every tensor key this adapter provides, sorted."""
        ...


class SafetensorsAdapterSource:
    """Reads adapters from a pool directory laid out as ``<root>/<adapter_id>/``.

    File handles are opened lazily and kept open for the life of the source: a
    merge touches every tensor of every selected adapter exactly once, and
    reopening the file per tensor would dominate the runtime.
    """

    def __init__(self, root: str | Path, *, device: str = "cpu") -> None:
        self.root = Path(root)
        self.device = device
        self._handles: dict[str, object] = {}
        self._keys: dict[str, list[str]] = {}

    def _weights_path(self, adapter_id: str) -> Path:
        return self.root / adapter_id / "adapter_model.safetensors"

    def has_adapter(self, adapter_id: str) -> bool:
        return self._weights_path(adapter_id).is_file()

    def _handle(self, adapter_id: str):
        if adapter_id not in self._handles:
            from safetensors import safe_open

            path = self._weights_path(adapter_id)
            if not path.is_file():
                raise AdapterLoadError(
                    f"adapter {adapter_id!r} not found at {path}. The certified pool must be "
                    "materialised on this host before reconstruction."
                )
            self._handles[adapter_id] = safe_open(path, framework="pt", device=self.device)
        return self._handles[adapter_id]

    def keys(self, adapter_id: str) -> list[str]:
        if adapter_id not in self._keys:
            self._keys[adapter_id] = sorted(self._handle(adapter_id).keys())
        return self._keys[adapter_id]

    def get_tensor(self, adapter_id: str, key: str) -> torch.Tensor:
        handle = self._handle(adapter_id)
        try:
            tensor = handle.get_tensor(key)
        except Exception as exc:  # noqa: BLE001
            raise AdapterLoadError(f"adapter {adapter_id!r} has no tensor {key!r}") from exc
        return tensor.to(WORK_DTYPE)

    def close(self) -> None:
        for handle in self._handles.values():
            close = getattr(handle, "close", None)
            if callable(close):
                close()
        self._handles.clear()

    def __enter__(self) -> SafetensorsAdapterSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class InMemoryAdapterSource:
    """A source backed by dictionaries. Used by tests and by the reference
    baseline builders, which synthesise adapters rather than reading them."""

    def __init__(self, adapters: dict[str, dict[str, torch.Tensor]]) -> None:
        self._adapters = {
            adapter_id: {key: tensor for key, tensor in tensors.items()}
            for adapter_id, tensors in adapters.items()
        }

    def has_adapter(self, adapter_id: str) -> bool:
        return adapter_id in self._adapters

    def keys(self, adapter_id: str) -> list[str]:
        if adapter_id not in self._adapters:
            raise AdapterLoadError(f"adapter {adapter_id!r} not present")
        return sorted(self._adapters[adapter_id])

    def get_tensor(self, adapter_id: str, key: str) -> torch.Tensor:
        try:
            return self._adapters[adapter_id][key].to(WORK_DTYPE)
        except KeyError as exc:
            raise AdapterLoadError(f"adapter {adapter_id!r} has no tensor {key!r}") from exc
