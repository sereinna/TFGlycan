"""Load the separately installed GlycanAA graph backend."""

from __future__ import annotations

import collections
import collections.abc
import importlib
import importlib.util
import os
import sys
import threading
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.RLock()
_loaded_root = None
_loaded_modules = None


def _glycanaa_root() -> Path:
    configured = os.environ.get("GLYCANAA_ROOT")
    root = Path(configured).expanduser() if configured else ROOT / "dependencies" / "GlycanAA"
    root = root.resolve()
    required = (
        "module/custom_data/__init__.py",
        "module/custom_data/glycan.py",
        "module/custom_data/all_atom_glycan.py",
        "module/custom_data/glycoword_vocab.pkl",
        "module/custom_models/glycanaa.py",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "GlycanAA is missing or incomplete. Clone "
            "https://github.com/kasawa1234/GlycanAA into dependencies/GlycanAA "
            "or set GLYCANAA_ROOT to an existing checkout. "
            f"Missing files: {', '.join(missing)}"
        )
    return root


def _import_torchdrug():
    """Let TorchDrug discover modern schedulers without editing its installation."""
    from torch.optim import lr_scheduler

    previous = getattr(lr_scheduler, "_LRScheduler", None)
    current = getattr(lr_scheduler, "LRScheduler", None)
    replace = previous is not None and current is not None and previous is not current
    if replace:
        lr_scheduler._LRScheduler = current
    try:
        return importlib.import_module("torchdrug")
    finally:
        if replace:
            lr_scheduler._LRScheduler = previous


def load_glycanaa_modules():
    """Return GlycanAA's glycan module, all-atom graph module, and encoder class."""
    global _loaded_root, _loaded_modules

    root = _glycanaa_root()
    with _LOCK:
        if _loaded_modules is not None:
            if root != _loaded_root:
                raise RuntimeError("GLYCANAA_ROOT changed after loading; start a new Python process.")
            return _loaded_modules

        existing = sys.modules.get("module")
        if existing is not None:
            locations = [Path(value).resolve() for value in getattr(existing, "__path__", ())]
            if root / "module" not in locations:
                raise RuntimeError(
                    "Another package named 'module' is already imported. "
                    "Load the GlycanAA backend in a new Python process."
                )

        executable_dir = str(Path(sys.executable).resolve().parent)
        search_path = os.environ.get("PATH", "").split(os.pathsep)
        if executable_dir not in search_path:
            os.environ["PATH"] = os.pathsep.join([executable_dir, *search_path])

        if not hasattr(collections, "Sequence"):
            collections.Sequence = collections.abc.Sequence

        import rdkit.Chem.Draw as draw

        if not hasattr(draw, "mplCanvas") or not hasattr(draw.mplCanvas, "Canvas"):
            class BaseCanvas:
                def rescalePt(self, point):
                    return point

            draw.mplCanvas = types.SimpleNamespace(Canvas=BaseCanvas)

        _import_torchdrug()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from module.custom_data import all_atom_glycan, glycan

        for graph_class in (
            all_atom_glycan.HeterogeneousAllAtomGlycan,
            all_atom_glycan.PackedHeterogeneousAllAtomGlycan,
        ):
            if not hasattr(graph_class, "cross_link_id"):
                graph_class.cross_link_id = (
                    graph_class.offset_bond_id_domain + graph_class.length_bond_id_domain
                )

        model_path = root / "module" / "custom_models" / "glycanaa.py"
        spec = importlib.util.spec_from_file_location("glycanaa_direct", model_path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the GlycanAA encoder from its checkout.")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        _loaded_root = root
        _loaded_modules = glycan, all_atom_glycan, model_module.GlycanAA
        return _loaded_modules
