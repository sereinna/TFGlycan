"""Feature caching and prediction for the task-family branches."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import Tensor
from .light import TAXONOMY_TASKS, DataStore, TensorBank, UnifiedGlycanModel
from .atom import AllAtomGraphCache, AllAtomTensorBank, GlycanAARankModel

def zscore(values: np.ndarray, stats: Tuple[float, float]) -> np.ndarray:
    (mean, std) = stats
    return (values - mean) / max(std, 1e-06)

@torch.no_grad()
def cache_light_features(model: UnifiedGlycanModel, bank: TensorBank, num_glycans: int, batch_size: int) -> Tensor:
    model.eval()
    chunks: List[Tensor] = []
    all_ids = np.arange(num_glycans, dtype=np.int64)
    for start in range(0, num_glycans, batch_size):
        batch_ids = all_ids[start:start + batch_size]
        (token_ids, mask, extra) = bank.glycan_batch(batch_ids)
        chunks.append(model.encode_glycan(token_ids, mask, extra).detach().cpu())
    return torch.cat(chunks, dim=0)

@torch.no_grad()
def cache_atom_features(model: GlycanAARankModel, graph_cache: AllAtomGraphCache, bank: AllAtomTensorBank, num_glycans: int, batch_size: int) -> Tensor:
    model.eval()
    chunks: List[Tensor] = []
    all_ids = np.arange(num_glycans, dtype=np.int64)
    for start in range(0, num_glycans, batch_size):
        batch_ids = all_ids[start:start + batch_size]
        chunks.append(model.encode_glycan_ids(batch_ids, graph_cache, bank).detach().cpu())
    return torch.cat(chunks, dim=0)

@torch.no_grad()
def predict_model_split(model, features: Tensor, protein_features: Tensor, data: DataStore, split: str, batch_size: int, device: torch.device) -> Dict[str, object]:
    model.eval()
    out: Dict[str, object] = {}
    tax = data.taxonomy[split]
    tax_logits = {task: np.zeros((len(tax['glycan']), data.num_classes[f'taxonomy/{task}']), dtype=np.float32) for task in TAXONOMY_TASKS}
    for start in range(0, len(tax['glycan']), batch_size * 4):
        end = start + batch_size * 4
        idx = torch.as_tensor(tax['glycan'][start:end], dtype=torch.long)
        feature = features[idx].to(device)
        logits = model.taxonomy_logits(feature)
        for task in TAXONOMY_TASKS:
            tax_logits[task][start:end] = logits[task].detach().cpu().numpy()
    out['taxonomy_logits'] = tax_logits
    out['taxonomy_labels'] = tax['labels']
    immuno = data.immuno[split]
    immuno_logits = []
    for start in range(0, len(immuno['glycan']), batch_size * 4):
        end = start + batch_size * 4
        idx = torch.as_tensor(immuno['glycan'][start:end], dtype=torch.long)
        feature = features[idx].to(device)
        immuno_logits.append(model.immuno_logits(feature).detach().cpu().numpy())
    out['immuno_logits'] = np.concatenate(immuno_logits) if immuno_logits else np.empty(0, dtype=np.float32)
    out['immuno_labels'] = immuno['labels']
    link = data.link[split]
    link_logits = np.zeros((len(link['glycan']), data.num_classes['glycosylation']), dtype=np.float32)
    for start in range(0, len(link['glycan']), batch_size * 4):
        end = start + batch_size * 4
        idx = torch.as_tensor(link['glycan'][start:end], dtype=torch.long)
        feature = features[idx].to(device)
        link_logits[start:end] = model.link_logits(feature).detach().cpu().numpy()
    out['link_logits'] = link_logits
    out['link_labels'] = link['labels']
    inter = data.interaction[split]
    preds = np.zeros(len(inter['labels']), dtype=np.float32)
    proteins = protein_features.to(device)
    for start in range(0, len(inter['labels']), batch_size * 8):
        end = start + batch_size * 8
        glycan_idx = torch.as_tensor(inter['glycan'][start:end], dtype=torch.long)
        protein_idx = torch.as_tensor(inter['protein'][start:end], dtype=torch.long, device=device)
        glycan_feature = features[glycan_idx].to(device)
        protein_feature = proteins[protein_idx]
        preds[start:end] = model.interaction_score(glycan_feature, protein_feature).detach().cpu().numpy()
    out['interaction_pred'] = preds
    out['interaction_labels'] = inter['labels']
    return out
