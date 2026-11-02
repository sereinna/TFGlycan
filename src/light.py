"""Sequence and motif encoding, task heads, and joint training."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import scipy.stats
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import Tensor, nn
from torch.nn import functional as F
TAXONOMY_TASKS = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
SPLITS = ('train', 'valid', 'test')
TOKEN_RE = re.compile('[ab]\\d-\\d|[ab]\\d-\\?|bond|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|\\[|\\]|\\(|\\)')
LINK_RE = re.compile('^[ab]\\d-\\d|^[ab]\\d-\\?')

@dataclass
class VariantConfig:
    name: str
    use_pair_bilinear: bool = False
    use_pair_rank_loss: bool = False
    use_balanced_losses: bool = False
    use_taxonomy_hierarchy: bool = False
    use_multiview_features: bool = False
VARIANTS: Dict[str, VariantConfig] = {'interaction_rank': VariantConfig('interaction_rank', use_pair_bilinear=True, use_pair_rank_loss=True)}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def tokenize_glycan(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(str(text))
    return tokens if tokens else ['<empty>']

def stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, byteorder='little', signed=False)

def max_branch_depth(text: str) -> int:
    depth = 0
    max_depth = 0
    for char in str(text):
        if char == '[':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ']':
            depth = max(0, depth - 1)
    return max_depth

def motif_feature(tokens: Sequence[str], raw: str, hash_dim: int) -> np.ndarray:
    numeric = np.array([len(tokens), len(raw), raw.count('['), raw.count(']'), max_branch_depth(raw), sum((1 for token in tokens if LINK_RE.match(token))), sum((1 for token in tokens if token.startswith('a'))), sum((1 for token in tokens if token.startswith('b'))), sum((1 for token in tokens if 'Neu' in token)), sum((1 for token in tokens if 'Fuc' in token)), sum((1 for token in tokens if 'NAc' in token)), len({token for token in tokens if token not in {'[', ']', '(', ')'}})], dtype=np.float32)
    numeric = np.log1p(numeric)
    hashed = np.zeros(hash_dim, dtype=np.float32)
    for ngram in (1, 2, 3):
        if len(tokens) < ngram:
            continue
        scale = 1.0 / math.sqrt(ngram)
        for i in range(len(tokens) - ngram + 1):
            key = '\x1f'.join(tokens[i:i + ngram])
            hashed[stable_hash(key) % hash_dim] += scale
    hashed = np.log1p(hashed)
    return np.concatenate([numeric, hashed])

def first_seen_mapping(values: Iterable[object]) -> Dict[object, int]:
    mapping: Dict[object, int] = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping

def split_mask(df: pd.DataFrame, split: str) -> np.ndarray:
    if split == 'valid':
        col = 'valid' if 'valid' in df.columns else 'validation'
    else:
        col = split
    return df[col].astype(bool).to_numpy()

def class_weights(labels: np.ndarray, num_class: int, max_weight: float=10.0) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=num_class).astype(np.float32)
    seen = counts > 0
    weights = np.zeros(num_class, dtype=np.float32)
    weights[seen] = counts[seen].sum() / (seen.sum() * counts[seen])
    if seen.any():
        weights[seen] /= weights[seen].mean()
    return np.clip(weights, 0.0, max_weight)

def binary_pos_weight(labels: np.ndarray, max_weight: float=20.0) -> float:
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())
    if positives == 0:
        return 1.0
    return min(max_weight, negatives / positives)

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))

def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(average_precision_score(y_true, y_score))

def safe_spearman(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_score) == 0:
        return float('nan')
    value = scipy.stats.spearmanr(y_true, y_score).correlation
    return float(value) if value is not None and (not np.isnan(value)) else float('nan')

def scalar_mean(values: Iterable[float]) -> float:
    valid = [float(v) for v in values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float('nan')

def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_ready(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value

class DataStore:

    def __init__(self, root: Path, max_len: int, hash_dim: int, protein_features: Path, max_interaction_train_pairs: Optional[int]=None):
        self.root = root
        self.max_len = max_len
        self.hash_dim = hash_dim
        self.protein_feature_path = protein_features
        self.max_interaction_train_pairs = max_interaction_train_pairs
        self.glycan_to_id: Dict[str, int] = {}
        self.tokenized: Dict[str, List[str]] = {}
        self.vocab: Dict[str, int] = {'<pad>': 0, '<unk>': 1}
        self.classification = pd.read_csv(root / 'glycan_classification.csv')
        self.immunogenicity = pd.read_csv(root / 'glycan_immunogenicity.csv')
        self.properties = pd.read_csv(root / 'glycan_properties.csv')
        self.interaction_wide = pd.read_csv(root / 'glycan_interaction.csv')
        self.taxonomy_maps: Dict[str, Dict[object, int]] = {}
        self.link_map: Dict[object, int] = {}
        self.train_glycan_ids: set[int] = set()
        self.taxonomy: Dict[str, Dict[str, np.ndarray]] = {}
        self.immuno: Dict[str, Dict[str, np.ndarray]] = {}
        self.link: Dict[str, Dict[str, np.ndarray]] = {}
        self.interaction: Dict[str, Dict[str, np.ndarray]] = {}
        self.interaction_groups: Dict[str, Dict[int, np.ndarray]] = {}
        self.hierarchy_maps: Dict[Tuple[str, str], np.ndarray] = {}
        self.class_weight_arrays: Dict[str, np.ndarray] = {}
        self.binary_weights: Dict[str, float] = {}
        self.num_classes: Dict[str, int] = {}
        self._prepare()

    def add_glycan(self, glycan: object) -> int:
        text = str(glycan)
        if text not in self.glycan_to_id:
            idx = len(self.glycan_to_id)
            self.glycan_to_id[text] = idx
            tokens = tokenize_glycan(text)
            self.tokenized[text] = tokens
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        return self.glycan_to_id[text]

    def _prepare(self) -> None:
        self._prepare_taxonomy()
        self._prepare_immunogenicity()
        self._prepare_link()
        self._prepare_interaction()
        self._prepare_hierarchy()
        self._prepare_tensors()

    def _prepare_taxonomy(self) -> None:
        df = self.classification
        for task in TAXONOMY_TASKS:
            self.taxonomy_maps[task] = first_seen_mapping(df[task])
            self.num_classes[f'taxonomy/{task}'] = len(self.taxonomy_maps[task])
        glycan_ids = np.array([self.add_glycan(x) for x in df['target']], dtype=np.int64)
        labels = np.stack([df[task].map(self.taxonomy_maps[task]).to_numpy(dtype=np.int64) for task in TAXONOMY_TASKS], axis=1)
        for split in SPLITS:
            mask = split_mask(df, split)
            self.taxonomy[split] = {'glycan': glycan_ids[mask], 'labels': labels[mask]}
            if split == 'train':
                self.train_glycan_ids.update(glycan_ids[mask].tolist())
        train_labels = self.taxonomy['train']['labels']
        for (i, task) in enumerate(TAXONOMY_TASKS):
            self.class_weight_arrays[f'taxonomy/{task}'] = class_weights(train_labels[:, i], self.num_classes[f'taxonomy/{task}'])

    def _prepare_immunogenicity(self) -> None:
        df = self.immunogenicity
        glycan_ids = np.array([self.add_glycan(x) for x in df['glycan']], dtype=np.int64)
        labels = df['immunogenicity'].to_numpy(dtype=np.float32)
        for split in SPLITS:
            mask = split_mask(df, split)
            self.immuno[split] = {'glycan': glycan_ids[mask], 'labels': labels[mask]}
            if split == 'train':
                self.train_glycan_ids.update(glycan_ids[mask].tolist())
        self.binary_weights['immunogenicity'] = binary_pos_weight(self.immuno['train']['labels'])

    def _prepare_link(self) -> None:
        df = self.properties[self.properties['link'].notna()].copy()
        self.link_map = first_seen_mapping(df['link'])
        self.num_classes['glycosylation'] = len(self.link_map)
        glycan_ids = np.array([self.add_glycan(x) for x in df['glycan']], dtype=np.int64)
        labels = df['link'].map(self.link_map).to_numpy(dtype=np.int64)
        for split in SPLITS:
            mask = split_mask(df, split)
            self.link[split] = {'glycan': glycan_ids[mask], 'labels': labels[mask]}
            if split == 'train':
                self.train_glycan_ids.update(glycan_ids[mask].tolist())
        self.class_weight_arrays['glycosylation'] = class_weights(self.link['train']['labels'], self.num_classes['glycosylation'])

    def _prepare_interaction(self) -> None:
        obj = torch.load(self.protein_feature_path, map_location='cpu')
        sequences = list(obj['sequences'])
        features = obj['features'].float()
        seq_to_protein = {seq: i for (i, seq) in enumerate(sequences)}
        self.protein_features = features
        df = self.interaction_wide
        glycan_columns = [c for c in df.columns if c not in {'target', 'train', 'valid', 'test'}]
        glycan_ids = np.array([self.add_glycan(c) for c in glycan_columns], dtype=np.int64)
        split_records: Dict[str, List[Tuple[int, int, float]]] = {split: [] for split in SPLITS}
        missing_feature_rows = 0
        for (_, row) in df.iterrows():
            seq = row['target']
            protein_idx = seq_to_protein.get(seq)
            if protein_idx is None:
                if row[glycan_columns].notna().any():
                    missing_feature_rows += 1
                continue
            split = 'train' if row['train'] else 'valid' if row['valid'] else 'test'
            values = row[glycan_columns].to_numpy(dtype=np.float32)
            observed = np.flatnonzero(~np.isnan(values))
            for col_idx in observed:
                split_records[split].append((protein_idx, int(glycan_ids[col_idx]), float(values[col_idx])))
        if missing_feature_rows:
            print(f'[data] skipped {missing_feature_rows} interaction rows without protein features')
        rng = np.random.default_rng(0)
        for (split, records) in split_records.items():
            arr = np.array(records, dtype=np.float64)
            if arr.size == 0:
                self.interaction[split] = {'protein': np.empty(0, dtype=np.int64), 'glycan': np.empty(0, dtype=np.int64), 'labels': np.empty(0, dtype=np.float32)}
                self.interaction_groups[split] = {}
                continue
            if split == 'train' and self.max_interaction_train_pairs:
                if len(arr) > self.max_interaction_train_pairs:
                    keep = rng.choice(len(arr), size=self.max_interaction_train_pairs, replace=False)
                    arr = arr[keep]
            protein = arr[:, 0].astype(np.int64)
            glycan = arr[:, 1].astype(np.int64)
            labels = arr[:, 2].astype(np.float32)
            self.interaction[split] = {'protein': protein, 'glycan': glycan, 'labels': labels}
            if split == 'train':
                self.train_glycan_ids.update(glycan.tolist())
            groups: Dict[int, List[int]] = {}
            for (idx, protein_idx) in enumerate(protein):
                groups.setdefault(int(protein_idx), []).append(idx)
            self.interaction_groups[split] = {protein_idx: np.array(indices, dtype=np.int64) for (protein_idx, indices) in groups.items() if len(indices) > 0}

    def _prepare_hierarchy(self) -> None:
        df = self.classification
        for (parent, child) in zip(TAXONOMY_TASKS[:-1], TAXONOMY_TASKS[1:]):
            parent_map = self.taxonomy_maps[parent]
            child_map = self.taxonomy_maps[child]
            child_to_parent = np.zeros(len(child_map), dtype=np.int64)
            for (child_value, group) in df.groupby(child):
                parent_value = group[parent].mode(dropna=False).iloc[0]
                child_to_parent[child_map[child_value]] = parent_map[parent_value]
            self.hierarchy_maps[parent, child] = child_to_parent

    def _prepare_tensors(self) -> None:
        texts_by_id = [None] * len(self.glycan_to_id)
        for (text, idx) in self.glycan_to_id.items():
            texts_by_id[idx] = text
        token_ids = np.zeros((len(texts_by_id), self.max_len), dtype=np.int64)
        mask = np.zeros((len(texts_by_id), self.max_len), dtype=np.bool_)
        raw_features = []
        for (idx, text) in enumerate(texts_by_id):
            assert text is not None
            tokens = self.tokenized[text]
            encoded = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens[:self.max_len]]
            token_ids[idx, :len(encoded)] = encoded
            mask[idx, :len(encoded)] = True
            raw_features.append(motif_feature(tokens, text, self.hash_dim))
        features = np.stack(raw_features).astype(np.float32)
        train_ids = sorted(self.train_glycan_ids)
        train_features = features[train_ids] if train_ids else features
        mean = train_features.mean(axis=0, keepdims=True)
        std = train_features.std(axis=0, keepdims=True)
        std[std < 1e-06] = 1.0
        features = (features - mean) / std
        self.glycan_token_ids = torch.from_numpy(token_ids)
        self.glycan_mask = torch.from_numpy(mask)
        self.glycan_features = torch.from_numpy(features)
        self.feature_dim = features.shape[1]

    def summary(self) -> Dict[str, object]:
        return {'num_unique_glycans': len(self.glycan_to_id), 'vocab_size': len(self.vocab), 'feature_dim': self.feature_dim, 'taxonomy_train': len(self.taxonomy['train']['glycan']), 'immuno_train': len(self.immuno['train']['glycan']), 'link_train': len(self.link['train']['glycan']), 'interaction_train_pairs': len(self.interaction['train']['labels']), 'interaction_valid_pairs': len(self.interaction['valid']['labels']), 'interaction_test_pairs': len(self.interaction['test']['labels']), 'link_map': {str(k): v for (k, v) in self.link_map.items()}}

class TensorBank:

    def __init__(self, data: DataStore, device: torch.device):
        self.device = device
        self.token_ids = data.glycan_token_ids.to(device)
        self.mask = data.glycan_mask.to(device)
        self.features = data.glycan_features.to(device)
        self.protein_features = data.protein_features.to(device)

    def glycan_batch(self, indices: np.ndarray) -> Tuple[Tensor, Tensor, Tensor]:
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return (self.token_ids[idx], self.mask[idx], self.features[idx])

    def protein_batch(self, indices: np.ndarray) -> Tensor:
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self.protein_features[idx]

class GlycanEncoder(nn.Module):

    def __init__(self, vocab_size: int, feature_dim: int, hidden_dim: int, token_dim: int, num_layers: int, dropout: float, use_multiview_features: bool):
        super().__init__()
        self.use_multiview_features = use_multiview_features
        self.embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.convs = nn.ModuleList([nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim, 1)
        self.pool_proj = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        if use_multiview_features:
            self.feature_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            self.fusion = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.output_dim = hidden_dim

    def forward(self, token_ids: Tensor, mask: Tensor, features: Tensor) -> Tensor:
        x = self.embedding(token_ids)
        x = self.input_proj(x)
        mask_f = mask.unsqueeze(-1).float()
        x = x * mask_f
        for (conv, norm) in zip(self.convs, self.norms):
            residual = x
            y = conv(x.transpose(1, 2)).transpose(1, 2)
            y = F.gelu(y)
            y = self.dropout(y)
            x = norm((residual + y) * mask_f)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (x * mask_f).sum(dim=1) / denom
        max_pool = x.masked_fill(~mask.unsqueeze(-1), -10000.0).max(dim=1).values
        attn_logits = self.attn(x).squeeze(-1).masked_fill(~mask, -10000.0)
        attn_weight = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
        attn_pool = (x * attn_weight).sum(dim=1)
        seq_feature = self.pool_proj(torch.cat([mean_pool, max_pool, attn_pool], dim=-1))
        if not self.use_multiview_features:
            return seq_feature
        extra = self.feature_proj(features)
        return self.fusion(torch.cat([seq_feature, extra, seq_feature * extra, seq_feature - extra], dim=-1))

class UnifiedGlycanModel(nn.Module):

    def __init__(self, vocab_size: int, glycan_feature_dim: int, protein_dim: int, taxonomy_dims: Mapping[str, int], link_dim: int, hidden_dim: int, token_dim: int, num_layers: int, dropout: float, variant: VariantConfig):
        super().__init__()
        self.variant = variant
        self.encoder = GlycanEncoder(vocab_size=vocab_size, feature_dim=glycan_feature_dim, hidden_dim=hidden_dim, token_dim=token_dim, num_layers=num_layers, dropout=dropout, use_multiview_features=variant.use_multiview_features)
        self.taxonomy_heads = nn.ModuleDict({task: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_class)) for (task, num_class) in taxonomy_dims.items()})
        self.immuno_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.link_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, link_dim))
        self.protein_proj = nn.Sequential(nn.LayerNorm(protein_dim), nn.Linear(protein_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        if variant.use_pair_bilinear:
            self.pair_head = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            self.dot_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.pair_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            self.dot_scale = None

    def encode_glycan(self, token_ids: Tensor, mask: Tensor, features: Tensor) -> Tensor:
        return self.encoder(token_ids, mask, features)

    def taxonomy_logits(self, glycan_feature: Tensor) -> Dict[str, Tensor]:
        return {task: head(glycan_feature) for (task, head) in self.taxonomy_heads.items()}

    def immuno_logits(self, glycan_feature: Tensor) -> Tensor:
        return self.immuno_head(glycan_feature).squeeze(-1)

    def link_logits(self, glycan_feature: Tensor) -> Tensor:
        return self.link_head(glycan_feature)

    def interaction_score(self, glycan_feature: Tensor, protein_feature: Tensor) -> Tensor:
        protein = self.protein_proj(protein_feature)
        if self.variant.use_pair_bilinear:
            pair = torch.cat([glycan_feature, protein, glycan_feature * protein, glycan_feature - protein], dim=-1)
            score = self.pair_head(pair).squeeze(-1)
            score = score + self.dot_scale * (glycan_feature * protein).sum(dim=-1) / math.sqrt(glycan_feature.shape[-1])
            return score
        pair = torch.cat([glycan_feature, protein], dim=-1)
        return self.pair_head(pair).squeeze(-1)

def to_device_weights(weights: Mapping[str, np.ndarray], device: torch.device) -> Dict[str, Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for (k, v) in weights.items()}

def sample_indices(rng: np.random.Generator, size: int, batch_size: int) -> np.ndarray:
    replace = size < batch_size
    return rng.choice(size, size=batch_size, replace=replace)

def sample_interaction_batch(rng: np.random.Generator, split_data: Mapping[str, np.ndarray], groups: Mapping[int, np.ndarray], proteins_per_batch: int, pairs_per_protein: int) -> np.ndarray:
    protein_ids = np.array(list(groups.keys()), dtype=np.int64)
    chosen_proteins = rng.choice(protein_ids, size=min(proteins_per_batch, len(protein_ids)), replace=len(protein_ids) < proteins_per_batch)
    chosen_indices = []
    for protein_id in chosen_proteins:
        options = groups[int(protein_id)]
        replace = len(options) < pairs_per_protein
        chosen_indices.append(rng.choice(options, size=pairs_per_protein, replace=replace))
    if not chosen_indices:
        return sample_indices(rng, len(split_data['labels']), proteins_per_batch * pairs_per_protein)
    return np.concatenate(chosen_indices)

def pairwise_rank_loss(pred: Tensor, target: Tensor, group: Tensor, min_delta: float=0.0001) -> Tensor:
    losses = []
    for group_id in torch.unique(group):
        idx = torch.nonzero(group == group_id, as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        p = pred[idx]
        y = target[idx]
        diff_y = y.unsqueeze(1) - y.unsqueeze(0)
        diff_p = p.unsqueeze(1) - p.unsqueeze(0)
        mask = diff_y.abs() > min_delta
        if not mask.any():
            continue
        sign = diff_y.sign()
        losses.append(F.softplus(-sign[mask] * diff_p[mask]).mean())
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()

def taxonomy_loss(logits: Mapping[str, Tensor], labels: Tensor, variant: VariantConfig, weights: Mapping[str, Tensor], hierarchy_maps: Mapping[Tuple[str, str], Tensor], hierarchy_weight: float) -> Tensor:
    losses = []
    for (i, task) in enumerate(TAXONOMY_TASKS):
        weight = weights.get(f'taxonomy/{task}') if variant.use_balanced_losses else None
        losses.append(F.cross_entropy(logits[task], labels[:, i], weight=weight))
    loss = torch.stack(losses).mean()
    if variant.use_taxonomy_hierarchy and hierarchy_weight > 0:
        hierarchy_losses = []
        for (parent, child) in zip(TAXONOMY_TASKS[:-1], TAXONOMY_TASKS[1:]):
            child_probs = torch.softmax(logits[child], dim=-1)
            child_to_parent = hierarchy_maps[parent, child]
            parent_dim = logits[parent].shape[-1]
            parent_from_child = child_probs.new_zeros((child_probs.shape[0], parent_dim))
            parent_from_child.scatter_add_(1, child_to_parent.unsqueeze(0).expand(child_probs.shape[0], -1), child_probs)
            parent_log_prob = torch.log(parent_from_child.clamp_min(1e-08))
            parent_labels = labels[:, TAXONOMY_TASKS.index(parent)]
            hierarchy_losses.append(F.nll_loss(parent_log_prob, parent_labels))
        loss = loss + hierarchy_weight * torch.stack(hierarchy_losses).mean()
    return loss

def train_one_epoch(model: UnifiedGlycanModel, data: DataStore, bank: TensorBank, optimizer: torch.optim.Optimizer, rng: np.random.Generator, variant: VariantConfig, weights: Mapping[str, Tensor], hierarchy_maps: Mapping[Tuple[str, str], Tensor], args: argparse.Namespace) -> Dict[str, float]:
    model.train()
    task_names = ['taxonomy', 'immuno', 'link', 'interaction']
    task_probs = np.array(args.task_probs, dtype=np.float64)
    task_probs = task_probs / task_probs.sum()
    losses: Dict[str, List[float]] = {task: [] for task in task_names}
    for _ in range(args.steps_per_epoch):
        task = str(rng.choice(task_names, p=task_probs))
        optimizer.zero_grad(set_to_none=True)
        if task == 'taxonomy':
            split = data.taxonomy['train']
            local_idx = sample_indices(rng, len(split['glycan']), args.batch_size)
            glycan_idx = split['glycan'][local_idx]
            labels = torch.as_tensor(split['labels'][local_idx], dtype=torch.long, device=bank.device)
            (token_ids, mask, features) = bank.glycan_batch(glycan_idx)
            glycan_feature = model.encode_glycan(token_ids, mask, features)
            logits = model.taxonomy_logits(glycan_feature)
            loss = taxonomy_loss(logits, labels, variant, weights, hierarchy_maps, args.hierarchy_weight)
        elif task == 'immuno':
            split = data.immuno['train']
            local_idx = sample_indices(rng, len(split['glycan']), args.batch_size)
            glycan_idx = split['glycan'][local_idx]
            labels = torch.as_tensor(split['labels'][local_idx], dtype=torch.float32, device=bank.device)
            (token_ids, mask, features) = bank.glycan_batch(glycan_idx)
            glycan_feature = model.encode_glycan(token_ids, mask, features)
            logits = model.immuno_logits(glycan_feature)
            pos_weight = torch.tensor(data.binary_weights['immunogenicity'], dtype=torch.float32, device=bank.device) if variant.use_balanced_losses else None
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        elif task == 'link':
            split = data.link['train']
            local_idx = sample_indices(rng, len(split['glycan']), args.batch_size)
            glycan_idx = split['glycan'][local_idx]
            labels = torch.as_tensor(split['labels'][local_idx], dtype=torch.long, device=bank.device)
            (token_ids, mask, features) = bank.glycan_batch(glycan_idx)
            glycan_feature = model.encode_glycan(token_ids, mask, features)
            logits = model.link_logits(glycan_feature)
            weight = weights.get('glycosylation') if variant.use_balanced_losses else None
            loss = F.cross_entropy(logits, labels, weight=weight)
        else:
            split = data.interaction['train']
            local_idx = sample_interaction_batch(rng, split, data.interaction_groups['train'], args.interaction_proteins_per_batch, args.interaction_pairs_per_protein)
            glycan_idx = split['glycan'][local_idx]
            protein_idx = split['protein'][local_idx]
            labels = torch.as_tensor(split['labels'][local_idx], dtype=torch.float32, device=bank.device)
            (token_ids, mask, features) = bank.glycan_batch(glycan_idx)
            protein_feature = bank.protein_batch(protein_idx)
            glycan_feature = model.encode_glycan(token_ids, mask, features)
            pred = model.interaction_score(glycan_feature, protein_feature)
            loss = F.smooth_l1_loss(pred, labels)
            if variant.use_pair_rank_loss and args.rank_weight > 0:
                group = torch.as_tensor(protein_idx, dtype=torch.long, device=bank.device)
                loss = loss + args.rank_weight * pairwise_rank_loss(pred, labels, group)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses[task].append(float(loss.detach().cpu()))
    return {f'train_loss/{task}': scalar_mean(values) for (task, values) in losses.items()}

@torch.no_grad()
def predict_glycan_features(model: UnifiedGlycanModel, bank: TensorBank, glycan_idx: np.ndarray, batch_size: int) -> Tensor:
    features = []
    model.eval()
    for start in range(0, len(glycan_idx), batch_size):
        batch_idx = glycan_idx[start:start + batch_size]
        (token_ids, mask, extra) = bank.glycan_batch(batch_idx)
        features.append(model.encode_glycan(token_ids, mask, extra).detach().cpu())
    return torch.cat(features, dim=0)

@torch.no_grad()
def evaluate_split(model: UnifiedGlycanModel, data: DataStore, bank: TensorBank, split: str, batch_size: int) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}
    tax = data.taxonomy[split]
    tax_features = predict_glycan_features(model, bank, tax['glycan'], batch_size).to(bank.device)
    logits = model.taxonomy_logits(tax_features)
    tax_scores = []
    for (i, task) in enumerate(TAXONOMY_TASKS):
        pred = logits[task].argmax(dim=-1).cpu().numpy()
        score = macro_f1(tax['labels'][:, i], pred)
        metrics[f'taxonomy/{task}_macro_f1'] = score
        tax_scores.append(score)
    metrics['taxonomy/mean_macro_f1'] = scalar_mean(tax_scores)
    immuno = data.immuno[split]
    immuno_features = predict_glycan_features(model, bank, immuno['glycan'], batch_size).to(bank.device)
    immuno_logits = model.immuno_logits(immuno_features).cpu().numpy()
    immuno_score = 1.0 / (1.0 + np.exp(-immuno_logits))
    metrics['immunogenicity/auprc'] = safe_auprc(immuno['labels'], immuno_score)
    link = data.link[split]
    link_features = predict_glycan_features(model, bank, link['glycan'], batch_size).to(bank.device)
    link_pred = model.link_logits(link_features).argmax(dim=-1).cpu().numpy()
    metrics['glycosylation/macro_f1'] = macro_f1(link['labels'], link_pred)
    inter = data.interaction[split]
    preds = []
    labels = []
    for start in range(0, len(inter['labels']), batch_size * 4):
        end = start + batch_size * 4
        glycan_idx = inter['glycan'][start:end]
        protein_idx = inter['protein'][start:end]
        (token_ids, mask, extra) = bank.glycan_batch(glycan_idx)
        protein_feature = bank.protein_batch(protein_idx)
        glycan_feature = model.encode_glycan(token_ids, mask, extra)
        pred = model.interaction_score(glycan_feature, protein_feature)
        preds.append(pred.detach().cpu().numpy())
        labels.append(inter['labels'][start:end])
    if preds:
        pred_all = np.concatenate(preds)
        label_all = np.concatenate(labels)
        metrics['interaction/spearman'] = safe_spearman(label_all, pred_all)
        metrics['interaction/smooth_l1'] = float(np.mean(np.where(np.abs(pred_all - label_all) < 1, 0.5 * (pred_all - label_all) ** 2, np.abs(pred_all - label_all) - 0.5)))
    else:
        metrics['interaction/spearman'] = float('nan')
        metrics['interaction/smooth_l1'] = float('nan')
    metrics['selection/composite'] = scalar_mean([metrics['taxonomy/mean_macro_f1'], metrics['immunogenicity/auprc'], metrics['glycosylation/macro_f1'], metrics['interaction/spearman']])
    return metrics

def make_model(args: argparse.Namespace, data: DataStore, variant: VariantConfig, device: torch.device) -> UnifiedGlycanModel:
    taxonomy_dims = {task: data.num_classes[f'taxonomy/{task}'] for task in TAXONOMY_TASKS}
    model = UnifiedGlycanModel(vocab_size=len(data.vocab), glycan_feature_dim=data.feature_dim, protein_dim=int(data.protein_features.shape[-1]), taxonomy_dims=taxonomy_dims, link_dim=data.num_classes['glycosylation'], hidden_dim=args.hidden_dim, token_dim=args.token_dim, num_layers=args.num_layers, dropout=args.dropout, variant=variant)
    return model.to(device)

def train_variant(args: argparse.Namespace, data: DataStore, variant: VariantConfig, device: torch.device) -> Dict[str, object]:
    variant_dir = Path(args.output_dir) / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    bank = TensorBank(data, device)
    model = make_model(args, data, variant, device)
    weights = to_device_weights(data.class_weight_arrays, device)
    hierarchy_maps = {key: torch.as_tensor(value, dtype=torch.long, device=device) for (key, value) in data.hierarchy_maps.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    rng = np.random.default_rng(args.seed + stable_hash(variant.name) % 100000)
    best_state = None
    best_valid: Optional[Dict[str, float]] = None
    best_epoch = -1
    history = []
    history_path = variant_dir / 'history.jsonl'
    if history_path.exists():
        history_path.unlink()
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, data, bank, optimizer, rng, variant, weights, hierarchy_maps, args)
        scheduler.step()
        valid_metrics = evaluate_split(model, data, bank, 'valid', args.eval_batch_size)
        row = {'epoch': epoch, 'lr': scheduler.get_last_lr()[0], **train_metrics, **{f'valid/{k}': v for (k, v) in valid_metrics.items()}}
        history.append(row)
        with history_path.open('a') as fout:
            fout.write(json.dumps(json_ready(row), sort_keys=True) + '\n')
        valid_score = valid_metrics['selection/composite']
        if best_valid is None or valid_score > best_valid['selection/composite']:
            best_valid = valid_metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, variant_dir / 'best_model.pt')
        print(f"[{variant.name}] epoch {epoch:03d} valid composite={valid_metrics['selection/composite']:.4f} tax={valid_metrics['taxonomy/mean_macro_f1']:.4f} immuno={valid_metrics['immunogenicity/auprc']:.4f} glycos={valid_metrics['glycosylation/macro_f1']:.4f} inter={valid_metrics['interaction/spearman']:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_split(model, data, bank, 'test', args.eval_batch_size)
    elapsed = time.time() - start_time
    with history_path.open('w') as fout:
        for row in history:
            fout.write(json.dumps(json_ready(row), sort_keys=True) + '\n')
    if best_state is not None:
        torch.save(best_state, variant_dir / 'best_model.pt')
    result = {'variant': asdict(variant), 'best_epoch': best_epoch, 'elapsed_sec': elapsed, 'best_valid': best_valid, 'test': test_metrics, 'args': vars(args)}
    with (variant_dir / 'result.json').open('w') as fout:
        json.dump(json_ready(result), fout, indent=2, sort_keys=True)
    return result

def flatten_result(result: Mapping[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {'variant': result['variant']['name'], 'best_epoch': result['best_epoch'], 'elapsed_sec': result['elapsed_sec']}
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--protein-features', type=Path, default=Path('data/protein_features.pt'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/unified'))
    parser.add_argument('--variant', choices=sorted(VARIANTS), default='interaction_rank')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--steps-per-epoch', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--eval-batch-size', type=int, required=True)
    parser.add_argument('--interaction-proteins-per-batch', type=int, required=True)
    parser.add_argument('--interaction-pairs-per-protein', type=int, required=True)
    parser.add_argument('--task-probs', type=float, nargs=4, metavar=('TAXONOMY', 'IMMUNO', 'LINK', 'INTERACTION'), required=True)
    parser.add_argument('--max-len', type=int, required=True)
    parser.add_argument('--hash-dim', type=int, required=True)
    parser.add_argument('--hidden-dim', type=int, required=True)
    parser.add_argument('--token-dim', type=int, required=True)
    parser.add_argument('--num-layers', type=int, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, required=True)
    parser.add_argument('--grad-clip', type=float, required=True)
    parser.add_argument('--rank-weight', type=float, required=True)
    parser.add_argument('--hierarchy-weight', type=float, required=True)
    parser.add_argument('--max-interaction-train-pairs', type=int, default=0, help='optional cap for quick smoke runs; 0 uses all training pairs')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.max_interaction_train_pairs <= 0:
        args.max_interaction_train_pairs = None
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith('cuda') else 'cpu')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = DataStore(args.data_dir, max_len=args.max_len, hash_dim=args.hash_dim, protein_features=args.protein_features, max_interaction_train_pairs=args.max_interaction_train_pairs)
    print('[data]', json.dumps(data.summary(), indent=2, sort_keys=True), flush=True)
    with (args.output_dir / 'data_summary.json').open('w') as fout:
        json.dump(json_ready(data.summary()), fout, indent=2, sort_keys=True)
    variant_names = [args.variant]
    results = []
    for name in variant_names:
        print(f'[run] starting variant={name} on device={device}', flush=True)
        results.append(train_variant(args, data, VARIANTS[name], device))
        write_summary(results, args.output_dir)
    print(f"[done] wrote {args.output_dir / 'summary.csv'}", flush=True)
if __name__ == '__main__':
    main()
