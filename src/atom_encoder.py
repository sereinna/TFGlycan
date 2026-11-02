"""Bond-gated molecular graph encoding with recurrent atom updates."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn

from .atom_graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, AtomBatch


class _BondGatedLayer(nn.Module):
    """Update nodes using directed, bond-conditioned messages and a GRU."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.gate = nn.Linear(3 * hidden_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: Tensor, edges: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index.unbind(0)
        message_input = torch.cat(
            (nodes.index_select(0, source), nodes.index_select(0, target), edges),
            dim=-1,
        )
        messages = self.message(message_input) * torch.sigmoid(self.gate(message_input))
        aggregate = nodes.new_zeros(nodes.shape).index_add(0, target, messages)
        degree = torch.bincount(target, minlength=nodes.shape[0]).to(nodes.dtype)
        aggregate = aggregate / degree.clamp_min(1).unsqueeze(-1)
        updated = self.update(aggregate, nodes)
        return self.norm(nodes + self.dropout(updated))


class BondGatedEncoder(nn.Module):
    """Encode each molecule in an :class:`AtomBatch` as one vector.

    Node updates average gated incoming messages, then apply a GRU and a
    normalized residual connection. A graph-specific attention distribution,
    an ordinary mean, and ``log(1 + atom_count)`` form the graph representation.
    Graphs may have interleaved nodes; each graph must contain at least one atom.
    All bonds must remain within one graph, and isolated atoms are supported.

    Args:
        hidden_dim: Positive width of node states and output vectors.
        layers: Positive number of message-passing layers.
        dropout: Dropout probability in the half-open interval [0, 1).

    Returns:
        ``forward`` returns a tensor of shape ``[batch.num_graphs, output_dim]``.
    """

    def __init__(self, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        for name, value in (("hidden_dim", hidden_dim), ("layers", layers)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be a finite number in [0, 1)")
        self.output_dim = hidden_dim
        self.node_projection = nn.Sequential(
            nn.Linear(ATOM_FEATURE_DIM, hidden_dim), nn.SiLU()
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(BOND_FEATURE_DIM, hidden_dim), nn.SiLU()
        )
        self.layers = nn.ModuleList(
            _BondGatedLayer(hidden_dim, float(dropout)) for _ in range(layers)
        )
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            # A shared scalar bias would cancel under softmax and cannot learn.
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    @staticmethod
    def _validate_batch(graphs: AtomBatch) -> None:
        for name in ("node_features", "edge_index", "edge_features", "batch"):
            if not isinstance(getattr(graphs, name, None), Tensor):
                raise TypeError(f"AtomBatch.{name} must be a torch.Tensor")
        nodes, edge_index, edges, membership = (
            graphs.node_features,
            graphs.edge_index,
            graphs.edge_features,
            graphs.batch,
        )
        if nodes.ndim != 2 or nodes.shape[1] != ATOM_FEATURE_DIM:
            raise ValueError(f"node_features must have shape [N, {ATOM_FEATURE_DIM}]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edges.ndim != 2 or edges.shape != (edge_index.shape[1], BOND_FEATURE_DIM):
            raise ValueError(f"edge_features must have shape [E, {BOND_FEATURE_DIM}]")
        if membership.ndim != 1 or membership.shape[0] != nodes.shape[0]:
            raise ValueError("batch must have shape [N]")
        if not nodes.is_floating_point() or not edges.is_floating_point():
            raise TypeError("node_features and edge_features must be floating point")
        if nodes.dtype != edges.dtype:
            raise TypeError("node_features and edge_features must have matching dtypes")
        if edge_index.dtype != torch.long or membership.dtype != torch.long:
            raise TypeError("edge_index and batch must have dtype torch.long")
        if any(t.device != nodes.device for t in (edge_index, edges, membership)):
            raise ValueError("all AtomBatch tensors must be on the same device")
        if (
            isinstance(graphs.num_graphs, bool)
            or not isinstance(graphs.num_graphs, int)
            or graphs.num_graphs <= 0
        ):
            raise ValueError("num_graphs must be a positive integer")
        if nodes.shape[0] == 0:
            raise ValueError("each graph must contain at least one atom")
        if not bool(torch.isfinite(nodes).all()) or not bool(torch.isfinite(edges).all()):
            raise ValueError("node_features and edge_features must contain finite values")
        if bool((membership < 0).any()) or bool((membership >= graphs.num_graphs).any()):
            raise ValueError("batch graph indices must be in [0, num_graphs)")
        if bool((torch.bincount(membership, minlength=graphs.num_graphs) == 0).any()):
            raise ValueError("each graph must contain at least one atom")
        if edge_index.numel():
            if bool((edge_index < 0).any()) or bool((edge_index >= nodes.shape[0]).any()):
                raise ValueError("edge_index contains an out-of-range atom index")
            source, target = edge_index.unbind(0)
            if bool((membership[source] != membership[target]).any()):
                raise ValueError("edges must not connect atoms from different graphs")

    def forward(self, graphs: AtomBatch) -> Tensor:
        self._validate_batch(graphs)
        nodes = self.node_projection(graphs.node_features)
        edges = self.edge_projection(graphs.edge_features)
        for layer in self.layers:
            nodes = layer(nodes, edges, graphs.edge_index)

        membership = graphs.batch
        count = torch.bincount(membership, minlength=graphs.num_graphs).to(nodes.dtype)
        score = self.attention_score(nodes).squeeze(-1)
        # Detaching the stabilizing shift preserves softmax derivatives and
        # avoids routing gradients through a discrete choice of maximal atom.
        maximum = score.new_full((graphs.num_graphs,), -torch.inf)
        maximum.scatter_reduce_(
            0, membership, score.detach(), reduce="amax", include_self=True
        )
        numerator = torch.exp(score - maximum.index_select(0, membership))
        denominator = score.new_zeros(graphs.num_graphs).index_add(0, membership, numerator)
        weights = numerator / denominator.index_select(0, membership)
        attention_pool = nodes.new_zeros((graphs.num_graphs, self.output_dim)).index_add(
            0, membership, nodes * weights.unsqueeze(-1)
        )
        mean_pool = nodes.new_zeros((graphs.num_graphs, self.output_dim)).index_add(
            0, membership, nodes
        ) / count.unsqueeze(-1)
        graph_features = torch.cat(
            (attention_pool, mean_pool, torch.log1p(count).unsqueeze(-1)), dim=-1
        )
        return self.readout(graph_features)
