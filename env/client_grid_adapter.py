from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import pandapower as pp

from env.grid_variants import ALL_CLIENTS, build_grid_variant


@dataclass(frozen=True)
class ClientGridContext:
    """Lightweight wrapper for a client-specific IEEE 33-bus grid."""

    client_name: str
    net: pp.pandapowerNet

    def clone_net(self) -> pp.pandapowerNet:
        """Return a deep-ish copy for isolated simulation steps."""
        return self.net.deepcopy() if hasattr(self.net, "deepcopy") else self.net.copy(deep=True)


class ClientGridAdapter:
    """Minimal adapter to expose the three client grids through a stable interface.

    The adapter is intentionally lightweight so it can later be plugged into the
    EMA / federated training pipeline without modifying immutable power modules.
    """

    def __init__(self, client_name: str, seed: int = 42):
        if client_name not in ALL_CLIENTS:
            raise ValueError(f"Unknown client_name={client_name!r}; expected one of {ALL_CLIENTS}")
        self.client_name = client_name
        self.seed = seed
        self.net = build_grid_variant(client_name, seed=seed)

    @property
    def context(self) -> ClientGridContext:
        return ClientGridContext(client_name=self.client_name, net=self.net)

    def get_net(self) -> pp.pandapowerNet:
        return self.net

    def refresh(self) -> pp.pandapowerNet:
        self.net = build_grid_variant(self.client_name, seed=self.seed)
        return self.net

    def as_context(self) -> ClientGridContext:
        return self.context


def make_client_grid(client_name: str, seed: int = 42) -> pp.pandapowerNet:
    """Convenience helper used by downstream FL / EMA code."""

    return build_grid_variant(client_name, seed=seed)


def iter_client_grids(seed: int = 42) -> Iterator[ClientGridContext]:
    """Yield the three client grids in a stable order."""

    for client_name in ALL_CLIENTS:
        net = build_grid_variant(client_name, seed=seed)
        yield ClientGridContext(client_name=client_name, net=net)


__all__ = [
    "ClientGridAdapter",
    "ClientGridContext",
    "ALL_CLIENTS",
    "build_grid_variant",
    "iter_client_grids",
    "make_client_grid",
]
