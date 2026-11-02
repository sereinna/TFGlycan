"""All-atom branch training with motif fusion and PCGrad."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .light import TAXONOMY_TASKS, DataStore, VariantConfig, json_ready, macro_f1, pairwise_rank_loss, safe_auprc, safe_spearman, sample_indices, sample_interaction_batch, scalar_mean, stable_hash, taxonomy_loss, to_device_weights
from .backend import load_glycanaa_modules

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
@dataclass
class GraphCacheStats:
    requested: int = 0
    built: int = 0
    failed: int = 0
    cached_ok: int = 0
    cached_fail: int = 0
    max_nodes: int = 0
    max_edges: int = 0

class AllAtomGraphCache:

    def __init__(self, data: DataStore, all_atom_glycan_module):
        self.all_atom_glycan = all_atom_glycan_module
        self.texts_by_id: List[str] = [''] * len(data.glycan_to_id)
        for (text, idx) in data.glycan_to_id.items():
            self.texts_by_id[idx] = text
        self.graphs: Dict[int, object] = {}
        self.failures: Dict[int, str] = {}
        self.stats = GraphCacheStats()

    def get(self, glycan_id: int):
        glycan_id = int(glycan_id)
        self.stats.requested += 1
        if glycan_id in self.graphs:
            self.stats.cached_ok += 1
            return self.graphs[glycan_id]
        if glycan_id in self.failures:
            self.stats.cached_fail += 1
            return None
        text = self.texts_by_id[glycan_id]
        try:
            graph = self.all_atom_glycan.HeterogeneousAllAtomGlycan.from_iupac(text, unit_feature='default', link_feature='default', atom_feature='default', bond_feature='default')
            if graph is None or int(graph.num_node) == 0:
                raise ValueError('empty all-atom graph')
            self.graphs[glycan_id] = graph
            self.stats.built += 1
            self.stats.max_nodes = max(self.stats.max_nodes, int(graph.num_node))
            self.stats.max_edges = max(self.stats.max_edges, int(graph.num_edge))
            return graph
        except Exception as exc:
            self.failures[glycan_id] = repr(exc)
            self.stats.failed += 1
            return None

    def pack(self, glycan_ids: Sequence[int], device: torch.device):
        graphs = []
        ok_pos = []
        failed_pos = []
        failed_ids = []
        for (pos, glycan_id) in enumerate(glycan_ids):
            graph = self.get(int(glycan_id))
            if graph is None:
                failed_pos.append(pos)
                failed_ids.append(int(glycan_id))
            else:
                ok_pos.append(pos)
                graphs.append(graph)
        packed = None
        if graphs:
            packed = self.all_atom_glycan.HeterogeneousAllAtomGlycan.pack(graphs)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    packed = packed.cuda()
            elif device.type == 'cpu':
                packed = packed.cpu()
        return (packed, np.asarray(ok_pos, dtype=np.int64), np.asarray(failed_pos, dtype=np.int64), np.asarray(failed_ids, dtype=np.int64))

    def warmup(self, ids: Sequence[int], report_every: int=1000) -> None:
        start = time.time()
        unique_ids = np.unique(np.asarray(ids, dtype=np.int64))
        for (i, glycan_id) in enumerate(unique_ids, start=1):
            self.get(int(glycan_id))
            if report_every > 0 and i % report_every == 0:
                elapsed = time.time() - start
                print(f'[graph-cache] warmed {i}/{len(unique_ids)} ok={len(self.graphs)} fail={len(self.failures)} elapsed={elapsed:.1f}s', flush=True)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        obj = torch.load(path, map_location='cpu', weights_only=False)
        self.graphs = {int(k): v for (k, v) in obj.get('graphs', {}).items()}
        self.failures = {int(k): str(v) for (k, v) in obj.get('failures', {}).items()}
        self.stats.built = len(self.graphs)
        self.stats.failed = len(self.failures)
        for graph in self.graphs.values():
            self.stats.max_nodes = max(self.stats.max_nodes, int(graph.num_node))
            self.stats.max_edges = max(self.stats.max_edges, int(graph.num_edge))
        print(f'[graph-cache] loaded {len(self.graphs)} graphs and {len(self.failures)} failures from {path}', flush=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'graphs': self.graphs, 'failures': self.failures}, path)
        print(f'[graph-cache] saved {len(self.graphs)} graphs and {len(self.failures)} failures to {path}', flush=True)

    def summary(self) -> Dict[str, object]:
        examples = []
        for (glycan_id, error) in list(self.failures.items())[:20]:
            examples.append([self.texts_by_id[glycan_id], error])
        return {**asdict(self.stats), 'num_cached_graphs': len(self.graphs), 'num_failures': len(self.failures), 'failure_examples': examples}

class AllAtomTensorBank:

    def __init__(self, data: DataStore, device: torch.device):
        self.device = device
        self.motif_features = data.glycan_features.to(device)
        self.protein_features = data.protein_features.to(device)

    def motif_batch(self, indices: Sequence[int]) -> Tensor:
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self.motif_features[idx]

    def protein_batch(self, indices: Sequence[int]) -> Tensor:
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self.protein_features[idx]

class GlycanAARankModel(nn.Module):

    def __init__(self, glycanaa_cls, num_glycans: int, motif_dim: int, protein_dim: int, taxonomy_dims: Mapping[str, int], link_dim: int, num_unit: int, num_relation: int, gnn_hidden_dim: int, gnn_layers: int, head_dim: int, dropout: float, use_motif_fusion: bool, interaction_head: str, readout: str, readout_type: str):
        super().__init__()
        self.use_motif_fusion = use_motif_fusion
        self.interaction_head = interaction_head
        self.glycanaa = glycanaa_cls(input_dim=gnn_hidden_dim, hidden_dims=[gnn_hidden_dim] * gnn_layers, num_unit=num_unit, num_relation=num_relation, edge_input_dim=None, short_cut=True, batch_norm=False, concat_hidden=True, readout=readout, readout_type=readout_type)
        graph_dim = int(self.glycanaa.output_dim)
        self.graph_proj = nn.Sequential(nn.Linear(graph_dim, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, head_dim), nn.LayerNorm(head_dim))
        self.fallback = nn.Embedding(num_glycans, head_dim)
        if use_motif_fusion:
            self.motif_proj = nn.Sequential(nn.Linear(motif_dim, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, head_dim), nn.LayerNorm(head_dim))
            self.fusion = nn.Sequential(nn.Linear(head_dim * 4, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, head_dim), nn.LayerNorm(head_dim))
        self.output_dim = head_dim
        self.taxonomy_heads = nn.ModuleDict({task: self._head(head_dim, num_class, dropout) for (task, num_class) in taxonomy_dims.items()})
        self.immuno_head = self._head(head_dim, 1, dropout)
        self.link_head = self._head(head_dim, link_dim, dropout)
        self.protein_proj = nn.Sequential(nn.LayerNorm(protein_dim), nn.Linear(protein_dim, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, head_dim), nn.LayerNorm(head_dim))
        self.pair_head = nn.Sequential(nn.Linear(head_dim * 4, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, 1))
        self.dot_scale = nn.Parameter(torch.tensor(1.0))
        if interaction_head == 'advanced':
            self.advanced_pair = nn.Sequential(nn.Linear(head_dim * 6, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, head_dim), nn.GELU(), nn.LayerNorm(head_dim), nn.Linear(head_dim, 1))
            self.advanced_gate = nn.Sequential(nn.Linear(head_dim * 4, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, head_dim), nn.Sigmoid())
            self.bilinear_glycan = nn.Linear(head_dim, head_dim, bias=False)
            self.bilinear_protein = nn.Linear(head_dim, head_dim, bias=False)
            self.advanced_scale = nn.Parameter(torch.tensor(0.0))
        elif interaction_head != 'base':
            raise ValueError(f'Unknown interaction head `{interaction_head}`')

    @staticmethod
    def _head(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, input_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(input_dim, output_dim))

    def encode_glycan_ids(self, glycan_ids: Sequence[int], graph_cache: AllAtomGraphCache, bank: AllAtomTensorBank) -> Tensor:
        (packed, ok_pos, failed_pos, failed_ids) = graph_cache.pack(glycan_ids, bank.device)
        features = torch.empty((len(glycan_ids), self.output_dim), device=bank.device)
        if packed is not None:
            raw = self.glycanaa(packed, None)['graph_feature']
            graph_feature = self.graph_proj(raw)
            pos = torch.as_tensor(ok_pos, dtype=torch.long, device=bank.device)
            features[pos] = graph_feature
        if len(failed_pos):
            pos = torch.as_tensor(failed_pos, dtype=torch.long, device=bank.device)
            ids = torch.as_tensor(failed_ids, dtype=torch.long, device=bank.device)
            features[pos] = self.fallback(ids)
        if self.use_motif_fusion:
            motif = self.motif_proj(bank.motif_batch(glycan_ids))
            features = self.fusion(torch.cat([features, motif, features * motif, features - motif], dim=-1))
        return features

    def taxonomy_logits(self, glycan_feature: Tensor) -> Dict[str, Tensor]:
        return {task: head(glycan_feature) for (task, head) in self.taxonomy_heads.items()}

    def immuno_logits(self, glycan_feature: Tensor) -> Tensor:
        return self.immuno_head(glycan_feature).squeeze(-1)

    def link_logits(self, glycan_feature: Tensor) -> Tensor:
        return self.link_head(glycan_feature)

    def interaction_score(self, glycan_feature: Tensor, protein_feature: Tensor) -> Tensor:
        protein = self.protein_proj(protein_feature)
        pair = torch.cat([glycan_feature, protein, glycan_feature * protein, glycan_feature - protein], dim=-1)
        score = self.pair_head(pair).squeeze(-1)
        score = score + self.dot_scale * (glycan_feature * protein).sum(dim=-1) / math.sqrt(glycan_feature.shape[-1])
        if self.interaction_head == 'advanced':
            gate = self.advanced_gate(pair)
            bilinear = self.bilinear_glycan(glycan_feature) * self.bilinear_protein(protein)
            advanced_pair = torch.cat([glycan_feature, protein, glycan_feature * protein, glycan_feature - protein, gate * glycan_feature, bilinear], dim=-1)
            advanced_score = self.advanced_pair(advanced_pair).squeeze(-1)
            advanced_score = advanced_score + bilinear.sum(dim=-1) / math.sqrt(glycan_feature.shape[-1])
            score = score + self.advanced_scale * advanced_score
        return score

def infer_graph_constants(args: argparse.Namespace, glycan_module, all_atom_glycan_module) -> Tuple[int, int]:
    graph = all_atom_glycan_module.HeterogeneousAllAtomGlycan.from_iupac('Man(a1-4)Man', unit_feature='default', link_feature='default', atom_feature='default', bond_feature='default')
    num_relation = int(graph.num_relation)
    min_num_unit = int(graph.unit_type.max().item()) + 1
    num_unit = max(args.num_unit, min_num_unit, len(glycan_module.Glycan.units) + 512)
    return (num_unit, num_relation)

def make_model(args: argparse.Namespace, data: DataStore, glycan_module, glycanaa_cls, all_atom_glycan_module, device: torch.device) -> GlycanAARankModel:
    (num_unit, num_relation) = infer_graph_constants(args, glycan_module, all_atom_glycan_module)
    taxonomy_dims = {task: data.num_classes[f'taxonomy/{task}'] for task in TAXONOMY_TASKS}
    model = GlycanAARankModel(glycanaa_cls=glycanaa_cls, num_glycans=len(data.glycan_to_id), motif_dim=data.feature_dim, protein_dim=int(data.protein_features.shape[-1]), taxonomy_dims=taxonomy_dims, link_dim=data.num_classes['glycosylation'], num_unit=num_unit, num_relation=num_relation, gnn_hidden_dim=args.gnn_hidden_dim, gnn_layers=args.gnn_layers, head_dim=args.head_dim, dropout=args.dropout, use_motif_fusion=args.use_motif_fusion, interaction_head=args.interaction_head, readout=args.readout, readout_type=args.readout_type)
    print(f'[model] num_unit={num_unit} num_relation={num_relation} glycanaa_dim={model.glycanaa.output_dim} head_dim={model.output_dim}', flush=True)
    return model.to(device)

def compute_task_loss(model: GlycanAARankModel, task: str, data: DataStore, bank: AllAtomTensorBank, rng: np.random.Generator, graph_cache: AllAtomGraphCache, loss_config: VariantConfig, weights: Mapping[str, Tensor], hierarchy_maps: Mapping[Tuple[str, str], Tensor], args: argparse.Namespace) -> Tuple[Tensor, str]:
    if task == 'taxonomy':
        split = data.taxonomy['train']
        local_idx = sample_indices(rng, len(split['glycan']), args.batch_size)
        glycan_idx = split['glycan'][local_idx]
        labels = torch.as_tensor(split['labels'][local_idx], dtype=torch.long, device=bank.device)
        glycan_feature = model.encode_glycan_ids(glycan_idx, graph_cache, bank)
        logits = model.taxonomy_logits(glycan_feature)
        loss = taxonomy_loss(logits, labels, loss_config, weights, hierarchy_maps, args.hierarchy_weight)
    elif task == 'immuno':
        split = data.immuno['train']
        local_idx = sample_indices(rng, len(split['glycan']), args.batch_size)
        glycan_idx = split['glycan'][local_idx]
        labels = torch.as_tensor(split['labels'][local_idx], dtype=torch.float32, device=bank.device)
        glycan_feature = model.encode_glycan_ids(glycan_idx, graph_cache, bank)
        logits = model.immuno_logits(glycan_feature)
        pos_weight = torch.tensor(data.binary_weights['immunogenicity'], dtype=torch.float32, device=bank.device) if loss_config.use_balanced_losses else None
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    elif task == 'link':
        split = data.link['train']
        local_idx = sample_indices(rng, len(split['glycan']), args.batch_size)
        glycan_idx = split['glycan'][local_idx]
        labels = torch.as_tensor(split['labels'][local_idx], dtype=torch.long, device=bank.device)
        glycan_feature = model.encode_glycan_ids(glycan_idx, graph_cache, bank)
        logits = model.link_logits(glycan_feature)
        weight = weights.get('glycosylation') if loss_config.use_balanced_losses else None
        loss = F.cross_entropy(logits, labels, weight=weight)
    elif task == 'interaction':
        split = data.interaction['train']
        local_idx = sample_interaction_batch(rng, split, data.interaction_groups['train'], args.interaction_proteins_per_batch, args.interaction_pairs_per_protein)
        glycan_idx = split['glycan'][local_idx]
        protein_idx = split['protein'][local_idx]
        labels = torch.as_tensor(split['labels'][local_idx], dtype=torch.float32, device=bank.device)
        glycan_feature = model.encode_glycan_ids(glycan_idx, graph_cache, bank)
        protein_feature = bank.protein_batch(protein_idx)
        pred = model.interaction_score(glycan_feature, protein_feature)
        loss = F.smooth_l1_loss(pred, labels)
        if args.rank_weight > 0:
            group = torch.as_tensor(protein_idx, dtype=torch.long, device=bank.device)
            loss = loss + args.rank_weight * pairwise_rank_loss(pred, labels, group)
    else:
        raise ValueError(f'Unknown task `{task}`')
    return (loss, task)

def flatten_grads(grads: Sequence[Optional[Tensor]], params: Sequence[Tensor]) -> Tensor:
    pieces = []
    for (grad, param) in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(grad.reshape(-1))
    return torch.cat(pieces)

def assign_flat_grad(flat_grad: Tensor, params: Sequence[Tensor]) -> None:
    offset = 0
    for param in params:
        numel = param.numel()
        param.grad = flat_grad[offset:offset + numel].view_as(param).clone()
        offset += numel

def pcgrad_backward(losses: Sequence[Tensor], params: Sequence[Tensor]) -> None:
    flat_grads = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        flat_grads.append(flatten_grads(grads, params))
    stacked = torch.stack(flat_grads)
    projected = stacked.clone()
    for i in range(projected.shape[0]):
        order = torch.randperm(projected.shape[0], device=projected.device)
        for j in order:
            if int(j) == i:
                continue
            dot = torch.dot(projected[i], stacked[j])
            if dot < 0:
                denom = torch.dot(stacked[j], stacked[j]).clamp_min(1e-12)
                projected[i] = projected[i] - dot / denom * stacked[j]
    assign_flat_grad(projected.mean(dim=0), params)

def train_one_epoch(model: GlycanAARankModel, data: DataStore, bank: AllAtomTensorBank, graph_cache: AllAtomGraphCache, optimizer: torch.optim.Optimizer, rng: np.random.Generator, loss_config: VariantConfig, weights: Mapping[str, Tensor], hierarchy_maps: Mapping[Tuple[str, str], Tensor], args: argparse.Namespace) -> Dict[str, float]:
    model.train()
    task_names = ['taxonomy', 'immuno', 'link', 'interaction']
    task_probs = np.asarray(args.task_probs, dtype=np.float64)
    task_probs = task_probs / task_probs.sum()
    losses: Dict[str, List[float]] = {task: [] for task in task_names}
    for step in range(args.steps_per_epoch):
        optimizer.zero_grad(set_to_none=True)
        if args.pcgrad:
            task_losses = []
            for pc_task in task_names:
                (task_loss, _) = compute_task_loss(model, pc_task, data, bank, rng, graph_cache, loss_config, weights, hierarchy_maps, args)
                task_losses.append(task_loss)
                losses[pc_task].append(float(task_loss.detach().cpu()))
            trainable_params = [param for group in optimizer.param_groups for param in group['params'] if param.requires_grad]
            pcgrad_backward(task_losses, trainable_params)
            last_task = 'pcgrad'
            last_loss = torch.stack([loss.detach() for loss in task_losses]).mean()
        else:
            task = str(rng.choice(task_names, p=task_probs))
            (loss, last_task) = compute_task_loss(model, task, data, bank, rng, graph_cache, loss_config, weights, hierarchy_maps, args)
            loss.backward()
            last_loss = loss.detach()
            losses[task].append(float(loss.detach().cpu()))
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params'] if p.requires_grad], args.grad_clip)
        optimizer.step()
        if args.train_log_every > 0 and (step + 1) % args.train_log_every == 0:
            print(f'[train] step {step + 1}/{args.steps_per_epoch} last_task={last_task} loss={float(last_loss.cpu()):.4f} cache_ok={len(graph_cache.graphs)} cache_fail={len(graph_cache.failures)}', flush=True)
    return {f'train_loss/{task}': scalar_mean(values) for (task, values) in losses.items()}

@torch.no_grad()
def encode_unique_features(model: GlycanAARankModel, graph_cache: AllAtomGraphCache, bank: AllAtomTensorBank, glycan_idx: np.ndarray, batch_size: int) -> Tuple[np.ndarray, Tensor, np.ndarray]:
    model.eval()
    (unique_ids, inverse) = np.unique(glycan_idx.astype(np.int64), return_inverse=True)
    chunks = []
    for start in range(0, len(unique_ids), batch_size):
        batch_ids = unique_ids[start:start + batch_size]
        chunks.append(model.encode_glycan_ids(batch_ids, graph_cache, bank).detach().cpu())
    if chunks:
        features = torch.cat(chunks, dim=0)
    else:
        features = torch.empty((0, model.output_dim), dtype=torch.float32)
    return (unique_ids, features, inverse.astype(np.int64))

@torch.no_grad()
def evaluate_split(model: GlycanAARankModel, data: DataStore, bank: AllAtomTensorBank, graph_cache: AllAtomGraphCache, split: str, batch_size: int, row_limit: int=0) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}

    def limited(split_data: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if row_limit <= 0:
            return dict(split_data)
        size = min(row_limit, len(split_data['glycan']))
        return {key: value[:size] for (key, value) in split_data.items()}
    tax = limited(data.taxonomy[split])
    (_, tax_unique_features, tax_inverse) = encode_unique_features(model, graph_cache, bank, tax['glycan'], batch_size)
    tax_preds: Dict[str, List[np.ndarray]] = {task: [] for task in TAXONOMY_TASKS}
    for start in range(0, len(tax_inverse), batch_size * 4):
        end = start + batch_size * 4
        inv = torch.as_tensor(tax_inverse[start:end], dtype=torch.long)
        feature = tax_unique_features[inv].to(bank.device)
        logits = model.taxonomy_logits(feature)
        for task in TAXONOMY_TASKS:
            tax_preds[task].append(logits[task].argmax(dim=-1).detach().cpu().numpy())
    tax_scores = []
    for (i, task) in enumerate(TAXONOMY_TASKS):
        pred = np.concatenate(tax_preds[task]) if tax_preds[task] else np.empty(0, dtype=np.int64)
        score = macro_f1(tax['labels'][:, i], pred)
        metrics[f'taxonomy/{task}_macro_f1'] = score
        tax_scores.append(score)
    metrics['taxonomy/mean_macro_f1'] = scalar_mean(tax_scores)
    immuno = limited(data.immuno[split])
    (_, immuno_unique_features, immuno_inverse) = encode_unique_features(model, graph_cache, bank, immuno['glycan'], batch_size)
    immuno_logits = []
    for start in range(0, len(immuno_inverse), batch_size * 4):
        end = start + batch_size * 4
        inv = torch.as_tensor(immuno_inverse[start:end], dtype=torch.long)
        feature = immuno_unique_features[inv].to(bank.device)
        immuno_logits.append(model.immuno_logits(feature).detach().cpu().numpy())
    immuno_logits_np = np.concatenate(immuno_logits) if immuno_logits else np.empty(0, dtype=np.float32)
    immuno_score = 1.0 / (1.0 + np.exp(-immuno_logits_np))
    metrics['immunogenicity/auprc'] = safe_auprc(immuno['labels'], immuno_score)
    link = limited(data.link[split])
    (_, link_unique_features, link_inverse) = encode_unique_features(model, graph_cache, bank, link['glycan'], batch_size)
    link_preds = []
    for start in range(0, len(link_inverse), batch_size * 4):
        end = start + batch_size * 4
        inv = torch.as_tensor(link_inverse[start:end], dtype=torch.long)
        feature = link_unique_features[inv].to(bank.device)
        link_preds.append(model.link_logits(feature).argmax(dim=-1).detach().cpu().numpy())
    link_pred = np.concatenate(link_preds) if link_preds else np.empty(0, dtype=np.int64)
    metrics['glycosylation/macro_f1'] = macro_f1(link['labels'], link_pred)
    inter = limited(data.interaction[split])
    (_, inter_unique_features_cpu, inter_inverse) = encode_unique_features(model, graph_cache, bank, inter['glycan'], batch_size)
    inter_unique_features = inter_unique_features_cpu.to(bank.device)
    preds = []
    labels = []
    for start in range(0, len(inter['labels']), batch_size * 8):
        end = start + batch_size * 8
        inv = torch.as_tensor(inter_inverse[start:end], dtype=torch.long, device=bank.device)
        glycan_feature = inter_unique_features[inv]
        protein_feature = bank.protein_batch(inter['protein'][start:end])
        pred = model.interaction_score(glycan_feature, protein_feature)
        preds.append(pred.detach().cpu().numpy())
        labels.append(inter['labels'][start:end])
    if preds:
        pred_all = np.concatenate(preds)
        label_all = np.concatenate(labels)
        metrics['interaction/spearman'] = safe_spearman(label_all, pred_all)
        diff = pred_all - label_all
        metrics['interaction/smooth_l1'] = float(np.mean(np.where(np.abs(diff) < 1, 0.5 * diff ** 2, np.abs(diff) - 0.5)))
    else:
        metrics['interaction/spearman'] = float('nan')
        metrics['interaction/smooth_l1'] = float('nan')
    metrics['selection/composite'] = scalar_mean([metrics['taxonomy/mean_macro_f1'], metrics['immunogenicity/auprc'], metrics['glycosylation/macro_f1'], metrics['interaction/spearman']])
    return metrics

def flatten_result(result: Mapping[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {'variant': result['variant'], 'best_epoch': result['best_epoch'], 'elapsed_sec': result['elapsed_sec']}
    for prefix in ('best_valid', 'test'):
        metrics = result[prefix]
        assert isinstance(metrics, Mapping)
        for (key, value) in metrics.items():
            row[f'{prefix}/{key}'] = value
    return row

def write_summary(results: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    rows = [flatten_result(result) for result in results]
    keys = sorted({key for row in rows for key in row})
    with (output_dir / 'summary.csv').open('w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'summary.json').open('w') as fout:
        json.dump(json_ready(rows), fout, indent=2, sort_keys=True)

def variant_name(args: argparse.Namespace) -> str:
    return args.variant_name or 'functional'

def train_run(args: argparse.Namespace, data: DataStore, glycan_module, all_atom_glycan_module, glycanaa_cls, device: torch.device) -> Dict[str, object]:
    name = variant_name(args)
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_cache = AllAtomGraphCache(data, all_atom_glycan_module)
    if args.graph_cache_file:
        graph_cache.load(args.graph_cache_file)
    if args.warm_graph_cache:
        all_ids = np.arange(len(data.glycan_to_id), dtype=np.int64)
        graph_cache.warmup(all_ids, report_every=args.graph_report_every)
        if args.graph_cache_file:
            graph_cache.save(args.graph_cache_file)
    bank = AllAtomTensorBank(data, device)
    model = make_model(args, data, glycan_module, glycanaa_cls, all_atom_glycan_module, device)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if args.init_non_strict:
            (missing, unexpected) = model.load_state_dict(state, strict=False)
            print(f'[model] non-strict init missing={len(missing)} unexpected={len(unexpected)}', flush=True)
            if missing:
                print(f'[model] missing keys sample={missing[:8]}', flush=True)
            if unexpected:
                print(f'[model] unexpected keys sample={unexpected[:8]}', flush=True)
        else:
            model.load_state_dict(state)
        print(f'[model] loaded init checkpoint from {args.init_checkpoint}', flush=True)
    weights = to_device_weights(data.class_weight_arrays, device)
    hierarchy_maps = {key: torch.as_tensor(value, dtype=torch.long, device=device) for (key, value) in data.hierarchy_maps.items()}
    loss_config = VariantConfig(name=name, use_pair_bilinear=True, use_pair_rank_loss=True, use_balanced_losses=args.balanced_losses, use_taxonomy_hierarchy=args.taxonomy_hierarchy, use_multiview_features=args.use_motif_fusion)
    encoder_names = ('glycanaa.', 'graph_proj.', 'motif_proj.', 'fusion.')
    encoder_params = []
    head_params = []
    for (param_name, param) in model.named_parameters():
        if param_name.startswith(encoder_names):
            encoder_params.append(param)
        else:
            head_params.append(param)
    optimizer = torch.optim.AdamW([{'params': encoder_params, 'lr': args.lr * args.encoder_lr_mult, 'name': 'encoder'}, {'params': head_params, 'lr': args.lr, 'name': 'heads'}], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    rng = np.random.default_rng(args.seed + stable_hash(name) % 100000)

    def set_encoder_trainable(trainable: bool) -> None:
        for param in encoder_params:
            param.requires_grad = trainable
    best_state = None
    best_valid: Optional[Dict[str, float]] = None
    best_epoch = -1
    history: List[Dict[str, object]] = []
    history_path = run_dir / 'history.jsonl'
    if history_path.exists():
        history_path.unlink()
    start_time = time.time()
    if args.eval_initial:
        valid_metrics = evaluate_split(model, data, bank, graph_cache, 'valid', args.eval_batch_size, row_limit=args.eval_row_limit)
        best_valid = valid_metrics
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, run_dir / 'best_model.pt')
        row = {'epoch': 0, 'lr': optimizer.param_groups[0]['lr'], **{f'valid/{k}': v for (k, v) in valid_metrics.items()}, 'graph_cache': graph_cache.summary()}
        history.append(row)
        with history_path.open('a') as fout:
            fout.write(json.dumps(json_ready(row), sort_keys=True) + '\n')
        print(f"[{name}] epoch 000 valid composite={valid_metrics['selection/composite']:.4f} tax={valid_metrics['taxonomy/mean_macro_f1']:.4f} immuno={valid_metrics['immunogenicity/auprc']:.4f} glycos={valid_metrics['glycosylation/macro_f1']:.4f} inter={valid_metrics['interaction/spearman']:.4f} cache_ok={len(graph_cache.graphs)} fail={len(graph_cache.failures)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        encoder_trainable = epoch > args.freeze_encoder_epochs
        set_encoder_trainable(encoder_trainable)
        if args.freeze_encoder_epochs > 0:
            print(f'[{name}] epoch {epoch:03d} encoder_trainable={encoder_trainable}', flush=True)
        train_metrics = train_one_epoch(model, data, bank, graph_cache, optimizer, rng, loss_config, weights, hierarchy_maps, args)
        scheduler.step()
        valid_metrics = evaluate_split(model, data, bank, graph_cache, 'valid', args.eval_batch_size, row_limit=args.eval_row_limit)
        row = {'epoch': epoch, 'lr': scheduler.get_last_lr()[0], **train_metrics, **{f'valid/{k}': v for (k, v) in valid_metrics.items()}, 'graph_cache': graph_cache.summary()}
        history.append(row)
        with history_path.open('a') as fout:
            fout.write(json.dumps(json_ready(row), sort_keys=True) + '\n')
        valid_score = valid_metrics['selection/composite']
        if best_valid is None or valid_score > best_valid['selection/composite']:
            best_valid = valid_metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, run_dir / 'best_model.pt')
        print(f"[{name}] epoch {epoch:03d} valid composite={valid_metrics['selection/composite']:.4f} tax={valid_metrics['taxonomy/mean_macro_f1']:.4f} immuno={valid_metrics['immunogenicity/auprc']:.4f} glycos={valid_metrics['glycosylation/macro_f1']:.4f} inter={valid_metrics['interaction/spearman']:.4f} cache_ok={len(graph_cache.graphs)} fail={len(graph_cache.failures)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_split(model, data, bank, graph_cache, 'test', args.eval_batch_size, row_limit=args.eval_row_limit)
    elapsed = time.time() - start_time
    result = {'variant': name, 'loss_config': asdict(loss_config), 'best_epoch': best_epoch, 'elapsed_sec': elapsed, 'best_valid': best_valid, 'test': test_metrics, 'graph_cache': graph_cache.summary(), 'args': vars(args)}
    with (run_dir / 'result.json').open('w') as fout:
        json.dump(json_ready(result), fout, indent=2, sort_keys=True)
    with (run_dir / 'graph_cache_summary.json').open('w') as fout:
        json.dump(json_ready(graph_cache.summary()), fout, indent=2, sort_keys=True)
    if args.graph_cache_file:
        graph_cache.save(args.graph_cache_file)
    if best_state is not None:
        torch.save(best_state, run_dir / 'best_model.pt')
    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data/glycanml'))
    parser.add_argument('--protein-features', type=Path, default=Path('data/protein_features.pt'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/functional'))
    parser.add_argument('--variant-name', default='')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init-checkpoint', type=Path, default=None)
    parser.add_argument('--init-non-strict', action='store_true')
    parser.add_argument('--eval-initial', action='store_true')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--steps-per-epoch', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--eval-batch-size', type=int, required=True)
    parser.add_argument('--interaction-proteins-per-batch', type=int, required=True)
    parser.add_argument('--interaction-pairs-per-protein', type=int, required=True)
    parser.add_argument('--task-probs', type=float, nargs=4, metavar=('TAXONOMY', 'IMMUNO', 'LINK', 'INTERACTION'), required=True)
    parser.add_argument('--max-len', type=int, required=True)
    parser.add_argument('--hash-dim', type=int, required=True)
    parser.add_argument('--gnn-hidden-dim', type=int, required=True)
    parser.add_argument('--gnn-layers', type=int, required=True)
    parser.add_argument('--head-dim', type=int, required=True)
    parser.add_argument('--num-unit', type=int, default=0)
    parser.add_argument('--readout', choices=['mean', 'sum', 'dual'], default='dual')
    parser.add_argument('--readout-type', choices=['mono', 'all'], default='mono')
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--interaction-head', choices=['base', 'advanced'], default='base')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--encoder-lr-mult', type=float, required=True)
    parser.add_argument('--freeze-encoder-epochs', type=int, required=True)
    parser.add_argument('--weight-decay', type=float, required=True)
    parser.add_argument('--grad-clip', type=float, required=True)
    parser.add_argument('--rank-weight', type=float, required=True)
    parser.add_argument('--hierarchy-weight', type=float, required=True)
    parser.add_argument('--taxonomy-hierarchy', action='store_true')
    parser.add_argument('--balanced-losses', action='store_true')
    parser.add_argument('--pcgrad', action='store_true')
    parser.add_argument('--use-motif-fusion', action='store_true')
    parser.add_argument('--warm-graph-cache', action='store_true')
    parser.add_argument('--graph-cache-file', type=Path, default=None)
    parser.add_argument('--graph-report-every', type=int, default=1000)
    parser.add_argument('--train-log-every', type=int, default=0)
    parser.add_argument('--eval-row-limit', type=int, default=0, help='debug-only cap per task/split; 0 evaluates full splits')
    parser.add_argument('--max-interaction-train-pairs', type=int, default=0, help='optional cap for quick smoke runs; 0 uses all training pairs')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.max_interaction_train_pairs <= 0:
        args.max_interaction_train_pairs = None
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith('cuda') else 'cpu')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (glycan_module, all_atom_glycan_module, glycanaa_cls) = load_glycanaa_modules()
    data = DataStore(args.data_dir, max_len=args.max_len, hash_dim=args.hash_dim, protein_features=args.protein_features, max_interaction_train_pairs=args.max_interaction_train_pairs)
    print('[data]', json.dumps(data.summary(), indent=2, sort_keys=True), flush=True)
    with (args.output_dir / 'data_summary.json').open('w') as fout:
        json.dump(json_ready(data.summary()), fout, indent=2, sort_keys=True)
    print(f'[run] starting variant={variant_name(args)} on device={device}', flush=True)
    result = train_run(args, data, glycan_module, all_atom_glycan_module, glycanaa_cls, device)
    write_summary([result], args.output_dir)
    print(f"[done] wrote {args.output_dir / 'summary.csv'}", flush=True)
if __name__ == '__main__':
    main()
