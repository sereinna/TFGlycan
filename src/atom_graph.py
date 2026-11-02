"""Independent, explicit-hydrogen molecular graphs built with public RDKit APIs.

GlyLES is used only to convert IUPAC-condensed text to complete SMILES.  Unknown
stereocenters remain unknown; unspecified topology is not guessed.  Features are
ordinary tensors so cached graphs do not retain RDKit or GlyLES objects.

API references:
https://glyles.readthedocs.io/en/latest/source/convert.html
https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html
https://www.rdkit.org/docs/source/rdkit.Chem.rdCIPLabeler.html
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Iterable

import torch
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler


ELEMENTS = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other")
ATOM_STEREO = ("none", "R", "S", "r", "s", "unknown")
BOND_STEREO = ("none", "E", "Z", "unknown")
ATOM_FEATURE_NAMES = (
    *(f"element_{symbol}" for symbol in ELEMENTS),
    "atomic_number/118", "formal_charge/4", "degree/6", "mass/250",
    "isotope/250", "aromatic", "in_ring", "hydrogen_neighbors/4",
    "radical_electrons/4", "hybrid_SP", "hybrid_SP2", "hybrid_SP3", "hybrid_other",
    *(f"stereo_{label}" for label in ATOM_STEREO),
)
BOND_FEATURE_NAMES = (
    "single", "double", "triple", "aromatic", "conjugated", "in_ring",
    *(f"stereo_{label}" for label in BOND_STEREO),
)
ATOM_FEATURE_DIM = len(ATOM_FEATURE_NAMES)
BOND_FEATURE_DIM = len(BOND_FEATURE_NAMES)
FORMAT_ID = "tfglycan-independent-atoms-v2-explicit-h-cip"
CACHE_TAG = FORMAT_ID

_BOND_TYPES = (
    Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE, Chem.BondType.AROMATIC,
)
_HYBRID_TYPES = (Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                 Chem.HybridizationType.SP3)
_TETRAHEDRAL_TAGS = (
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)


def _one_hot(value, choices) -> list[float]:
    return [float(value == choice) for choice in choices]


def _nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _promote_leading_branch(text: str) -> str:
    """Write a leading side chain as the main chain without changing bonds."""
    while text.startswith("["):
        depth, closing = 0, None
        for index, character in enumerate(text):
            depth += int(character == "[") - int(character == "]")
            if depth == 0:
                closing = index
                break
        if closing is None or closing <= 1 or closing == len(text) - 1:
            raise ValueError("A leading branch must have an attachment residue")
        text = text[1:closing] + text[closing + 1:]
    return text


def _molecule_from_smiles(smiles: str):
    # Do not let RDKit silently accept a trailing molecule name or CXSMILES
    # annotations containing topology/stereo that these features do not encode.
    if any(character.isspace() for character in smiles) or "|" in smiles:
        raise ValueError("SMILES must contain one structure without names or CXSMILES annotations")
    parameters = Chem.SmilesParserParams()
    parameters.parseName = False
    parameters.allowCXSMILES = False
    parameters.removeHs = False
    try:
        molecule = Chem.MolFromSmiles(smiles, parameters)
    except Exception as exc:
        raise ValueError("RDKit could not parse the SMILES") from exc
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("Invalid or empty SMILES structure")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("A graph must describe one connected molecule")
    for atom in molecule.GetAtoms():
        if atom.HasQuery() or not 1 <= atom.GetAtomicNum() <= 118:
            raise ValueError("Wildcard or query atoms do not define a complete molecule")
        if atom.GetChiralTag() not in _TETRAHEDRAL_TAGS:
            raise ValueError("Only tetrahedral atom stereochemistry is supported")
    for bond in molecule.GetBonds():
        if bond.HasQuery() or bond.GetBondType() not in _BOND_TYPES:
            raise ValueError("Only defined single, double, triple and aromatic bonds are supported")
    try:
        molecule = Chem.AddHs(molecule, explicitOnly=False, addCoords=False)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        potential = list(Chem.FindPotentialStereo(molecule, cleanIt=False, flagPossible=True))
        # The accurate CIP implementation avoids encoding traversal-relative @/@@.
        rdCIPLabeler.AssignCIPLabels(molecule)
    except Exception as exc:
        raise ValueError("Hydrogen or stereochemistry assignment failed") from exc
    atom_stereo, bond_stereo = {}, {}
    for item in potential:
        if item.type == Chem.StereoType.Atom_Tetrahedral:
            atom_stereo[item.centeredOn] = item.specified
        elif item.type == Chem.StereoType.Bond_Double:
            bond_stereo[item.centeredOn] = item.specified
        elif item.type in (
            Chem.StereoType.Atom_SquarePlanar,
            Chem.StereoType.Atom_TrigonalBipyramidal,
            Chem.StereoType.Atom_Octahedral,
        ) and item.specified != Chem.StereoSpecified.Specified:
            # RDKit can flag unassigned non-tetrahedral possibilities at sulfate
            # or phosphate centers. Preserve them as unknown, without assigning
            # a geometry or rejecting an otherwise defined molecular structure.
            atom_stereo[item.centeredOn] = item.specified
        else:
            raise ValueError(f"Unsupported stereochemistry: {item.type}")
    return molecule, atom_stereo, bond_stereo


def _atom_features(atom, potential: dict) -> list[float]:
    symbol = atom.GetSymbol()
    symbol = symbol if symbol in ELEMENTS else "other"
    if atom.HasProp("_CIPCode"):
        stereo = atom.GetProp("_CIPCode")
        if stereo not in ATOM_STEREO:
            raise ValueError(f"Unsupported atom CIP label: {stereo}")
    elif atom.GetIdx() in potential:
        if potential[atom.GetIdx()] == Chem.StereoSpecified.Specified:
            raise ValueError("Specified atom stereochemistry has no supported CIP label")
        stereo = "unknown"
    else:
        stereo = "none"
    hybridization = atom.GetHybridization()
    return (
        _one_hot(symbol, ELEMENTS)
        + [atom.GetAtomicNum() / 118.0, atom.GetFormalCharge() / 4.0,
           atom.GetDegree() / 6.0, atom.GetMass() / 250.0,
           atom.GetIsotope() / 250.0, float(atom.GetIsAromatic()),
           float(atom.IsInRing()),
           sum(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()) / 4.0,
           atom.GetNumRadicalElectrons() / 4.0]
        + _one_hot(hybridization, _HYBRID_TYPES)
        + [float(hybridization not in _HYBRID_TYPES)]
        + _one_hot(stereo, ATOM_STEREO)
    )


def _bond_features(bond, potential: dict) -> list[float]:
    if bond.HasProp("_CIPCode"):
        stereo = bond.GetProp("_CIPCode")
        if stereo not in ("E", "Z"):
            raise ValueError(f"Unsupported bond CIP label: {stereo}")
    elif bond.GetIdx() in potential:
        if potential[bond.GetIdx()] == Chem.StereoSpecified.Specified:
            raise ValueError("Specified double-bond stereochemistry has no E/Z label")
        stereo = "unknown"
    elif bond.GetStereo() == Chem.BondStereo.STEREOANY:
        stereo = "unknown"
    elif bond.GetStereo() == Chem.BondStereo.STEREONONE:
        stereo = "none"
    else:
        raise ValueError("Bond stereochemistry could not be represented as E/Z")
    return (_one_hot(bond.GetBondType(), _BOND_TYPES)
            + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
            + _one_hot(stereo, BOND_STEREO))


@dataclass
class AtomGraph:
    """One molecular graph with explicitly represented hydrogen atoms.

    Feature column definitions are exported as ATOM_FEATURE_NAMES and
    BOND_FEATURE_NAMES.  num_edge counts directed edges (twice the bond count).
    Graphs must be imported through ``src.atom_graph`` for portable pickle caches.
    """

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    smiles: str = ""
    iupac: str = ""

    FORMAT_ID: ClassVar[str] = FORMAT_ID
    CACHE_TAG: ClassVar[str] = CACHE_TAG

    def __post_init__(self):
        if self.node_features.ndim != 2 or self.node_features.shape[1] != ATOM_FEATURE_DIM:
            raise ValueError(f"node_features must have shape [N, {ATOM_FEATURE_DIM}]")
        if self.node_features.dtype != torch.float32 or self.node_features.shape[0] == 0:
            raise ValueError("node_features must be nonempty float32")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.edge_index.dtype != torch.long:
            raise ValueError("edge_index must have dtype int64")
        if self.edge_features.shape != (self.num_edge, BOND_FEATURE_DIM):
            raise ValueError(f"edge_features must have shape [E, {BOND_FEATURE_DIM}]")
        if self.edge_features.dtype != torch.float32:
            raise ValueError("edge_features must have dtype float32")
        if len({self.node_features.device, self.edge_index.device, self.edge_features.device}) != 1:
            raise ValueError("All graph tensors must use the same device")
        if self.num_edge and (int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= self.num_node):
            raise ValueError("edge_index contains an out-of-range atom index")

    @property
    def num_node(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edge(self) -> int:
        return int(self.edge_index.shape[1])

    @classmethod
    def from_smiles(cls, smiles: str) -> "AtomGraph":
        smiles = _nonempty_text(smiles, "SMILES")
        molecule, atom_stereo, bond_stereo = _molecule_from_smiles(smiles)
        nodes = [_atom_features(atom, atom_stereo) for atom in molecule.GetAtoms()]
        edges, features = [], []
        for bond in molecule.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            vector = _bond_features(bond, bond_stereo)
            edges.extend(((start, end), (end, start)))
            features.extend((vector, vector))
        # Build from plain Python values, not NumPy-converted RDKit objects.
        edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                      if edges else torch.empty((2, 0), dtype=torch.long))
        edge_features = (torch.tensor(features, dtype=torch.float32)
                         if features else torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32))
        return cls(torch.tensor(nodes, dtype=torch.float32), edge_index, edge_features, smiles=smiles)

    @classmethod
    def from_iupac(cls, text: str) -> "AtomGraph":
        text = _nonempty_text(text, "IUPAC")
        if any(marker in text for marker in ("?", "*", "/", "\\", "{", "}")):
            raise ValueError("Ambiguous linkage, repeat or residue notation cannot define one molecular graph")
        conversion_text = _promote_leading_branch(text)
        # SMILES-only users and tensor cache readers do not need GlyLES installed.
        try:
            from glyles import convert
        except ImportError as exc:
            raise ImportError("IUPAC conversion requires the ordinary 'glyles' package") from exc
        try:
            converted = convert(glycan=conversion_text, returning=True, cpu_count=1, full=True)
        except Exception as exc:
            raise ValueError(f"GlyLES could not fully convert IUPAC: {text!r}") from exc
        if (not isinstance(converted, (list, tuple)) or len(converted) != 1
                or not isinstance(converted[0], (list, tuple)) or len(converted[0]) != 2
                or converted[0][0] != conversion_text or not isinstance(converted[0][1], str)
                or not converted[0][1].strip()):
            raise ValueError(f"GlyLES returned no complete unambiguous structure for: {text!r}")
        graph = cls.from_smiles(converted[0][1])
        graph.iupac = text
        return graph

    @staticmethod
    def pack(graphs: Iterable["AtomGraph"]) -> "AtomBatch":
        graphs = list(graphs)
        if not graphs:
            raise ValueError("Cannot pack an empty graph list")
        if any(not isinstance(graph, AtomGraph) or isinstance(graph, AtomBatch) for graph in graphs):
            raise TypeError("pack expects individual AtomGraph instances")
        device = graphs[0].node_features.device
        if any(graph.node_features.device != device for graph in graphs):
            raise ValueError("All graphs must be on the same device before packing")
        offset, edges, assignments = 0, [], []
        for index, graph in enumerate(graphs):
            edges.append(graph.edge_index + offset)
            assignments.append(torch.full((graph.num_node,), index, dtype=torch.long, device=device))
            offset += graph.num_node
        return AtomBatch(
            node_features=torch.cat([graph.node_features for graph in graphs], dim=0),
            edge_index=torch.cat(edges, dim=1),
            edge_features=torch.cat([graph.edge_features for graph in graphs], dim=0),
            batch=torch.cat(assignments), num_graphs=len(graphs),
        )

    def to(self, device, non_blocking: bool = False) -> "AtomGraph":
        changes = {
            name: getattr(self, name).to(device=device, non_blocking=non_blocking)
            for name in ("node_features", "edge_index", "edge_features")
        }
        if isinstance(self, AtomBatch):
            changes["batch"] = self.batch.to(device=device, non_blocking=non_blocking)
        return replace(self, **changes)

    def cpu(self) -> "AtomGraph":
        return self.to("cpu")

    def cuda(self, device=None, non_blocking: bool = False) -> "AtomGraph":
        if device is None:
            device = torch.device("cuda")
        elif isinstance(device, int):
            device = torch.device("cuda", device)
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("cuda() expects a CUDA device")
        return self.to(device, non_blocking=non_blocking)


@dataclass
class AtomBatch(AtomGraph):
    """Disjoint union of graphs, with a node-to-graph assignment tensor."""

    batch: torch.Tensor | None = None
    num_graphs: int = 0

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.num_graphs, int) or not 1 <= self.num_graphs <= self.num_node:
            raise ValueError("num_graphs must count one or more nonempty graphs")
        if self.batch is None or self.batch.shape != (self.num_node,) or self.batch.dtype != torch.long:
            raise ValueError("batch must be an int64 node-to-graph vector")
        if self.batch.device != self.node_features.device:
            raise ValueError("batch and graph tensors must use the same device")
        if int(self.batch.min()) < 0 or int(self.batch.max()) >= self.num_graphs:
            raise ValueError("batch contains an out-of-range graph index")
        if torch.unique(self.batch).numel() != self.num_graphs:
            raise ValueError("Every graph in a batch must contain at least one node")
        if self.num_edge and not torch.equal(self.batch[self.edge_index[0]], self.batch[self.edge_index[1]]):
            raise ValueError("Edges cannot connect different graphs in a batch")
