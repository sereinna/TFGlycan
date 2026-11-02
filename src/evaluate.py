"""Task-family routing and evaluation."""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Mapping, Optional, Tuple
import numpy as np
import torch
from .light import TAXONOMY_TASKS, VARIANTS, DataStore, TensorBank, UnifiedGlycanModel, json_ready, macro_f1, safe_auprc, safe_spearman, scalar_mean, make_model as make_light_model, set_seed
from .atom import AllAtomGraphCache, AllAtomTensorBank, GlycanAARankModel, load_glycanaa_modules, make_model as make_atom_model
from .prediction import cache_atom_features, cache_light_features, predict_model_split, zscore

@dataclass
class BranchManifest:
    taxonomy_branch: str
    functional_glycan_branch: str
    interaction_light_branch: str
    interaction_atom_branch: str
    taxonomy_coarse_tau: float
    taxonomy_fine_tau: float
    interaction_alpha: float
    interaction_zscore: bool

def make_light_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(hidden_dim=args.hidden_dim, token_dim=args.token_dim, num_layers=args.num_layers, dropout=args.dropout)

def make_atom_args(args: argparse.Namespace, interaction_head: str) -> argparse.Namespace:
    return SimpleNamespace(num_unit=args.num_unit, gnn_hidden_dim=args.gnn_hidden_dim, gnn_layers=args.gnn_layers, head_dim=args.head_dim, dropout=args.dropout, use_motif_fusion=args.atom_use_motif_fusion, interaction_head=interaction_head, readout=args.readout, readout_type=args.readout_type)

def load_light_branch(args: argparse.Namespace, data: DataStore, checkpoint: Path, variant_name: str, device: torch.device) -> UnifiedGlycanModel:
    model = make_light_model(make_light_args(args), data, VARIANTS[variant_name], device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f'[light-branch] loaded {checkpoint} variant={variant_name}', flush=True)
    return model

def load_atom_branch(args: argparse.Namespace, data: DataStore, checkpoint: Path, glycan_module, all_atom_glycan_module, glycanaa_cls, interaction_head: str, device: torch.device) -> GlycanAARankModel:
    model = make_atom_model(make_atom_args(args, interaction_head), data, glycan_module, glycanaa_cls, all_atom_glycan_module, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    (missing, unexpected) = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'[atom-branch] non-strict load missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    model.eval()
    print(f'[atom-branch] loaded {checkpoint} interaction_head={interaction_head}', flush=True)
    return model

def class_log_prior(data: DataStore, smoothing: float) -> Dict[str, np.ndarray]:
    labels = data.taxonomy['train']['labels']
    priors: Dict[str, np.ndarray] = {}
    for (i, task) in enumerate(TAXONOMY_TASKS):
        num_class = data.num_classes[f'taxonomy/{task}']
        counts = np.bincount(labels[:, i].astype(np.int64), minlength=num_class).astype(np.float64)
        probs = (counts + smoothing) / (counts.sum() + smoothing * num_class)
        priors[task] = np.log(np.clip(probs, 1e-12, None)).astype(np.float32)
    return priors

def taxonomy_tau_by_task(coarse_tau: float, fine_tau: float) -> Dict[str, float]:
    result = {}
    for task in TAXONOMY_TASKS[:4]:
        result[task] = coarse_tau
    for task in TAXONOMY_TASKS[4:]:
        result[task] = fine_tau
    return result

def calibrated_taxonomy_metrics(taxonomy_outputs: Mapping[str, object], data: DataStore, priors: Mapping[str, np.ndarray], coarse_tau: float, fine_tau: float) -> Dict[str, float]:
    logits = taxonomy_outputs['taxonomy_logits']
    labels = taxonomy_outputs['taxonomy_labels']
    assert isinstance(logits, Mapping)
    assert isinstance(labels, np.ndarray)
    tau = taxonomy_tau_by_task(coarse_tau, fine_tau)
    scores = []
    metrics: Dict[str, float] = {}
    for (i, task) in enumerate(TAXONOMY_TASKS):
        task_logits = logits[task]
        assert isinstance(task_logits, np.ndarray)
        adjusted = task_logits - tau[task] * priors[task][None, :]
        pred = adjusted.argmax(axis=-1)
        score = macro_f1(labels[:, i], pred)
        metrics[f'taxonomy/{task}_macro_f1'] = score
        scores.append(score)
    metrics['taxonomy/mean_macro_f1'] = scalar_mean(scores)
    return metrics

def interaction_metrics(light_outputs: Mapping[str, object], atom_outputs: Mapping[str, object], alpha: float, interaction_stats: Optional[Mapping[str, Tuple[float, float]]]) -> Dict[str, float]:
    light_pred = light_outputs['interaction_pred']
    atom_pred = atom_outputs['interaction_pred']
    labels = light_outputs['interaction_labels']
    assert isinstance(light_pred, np.ndarray)
    assert isinstance(atom_pred, np.ndarray)
    assert isinstance(labels, np.ndarray)
    if interaction_stats is not None:
        light_pred = zscore(light_pred, interaction_stats['light'])
        atom_pred = zscore(atom_pred, interaction_stats['atom'])
    pred = (1.0 - alpha) * light_pred + alpha * atom_pred
    diff = pred - labels
    return {'interaction/spearman': safe_spearman(labels, pred), 'interaction/smooth_l1': float(np.mean(np.where(np.abs(diff) < 1, 0.5 * diff ** 2, np.abs(diff) - 0.5)))}

def full_metrics(taxonomy_metrics: Mapping[str, float], functional_outputs: Mapping[str, object], interaction_metrics_value: Mapping[str, float]) -> Dict[str, float]:
    metrics = dict(taxonomy_metrics)
    immuno_logits = functional_outputs['immuno_logits']
    immuno_labels = functional_outputs['immuno_labels']
    link_logits = functional_outputs['link_logits']
    link_labels = functional_outputs['link_labels']
    assert isinstance(immuno_logits, np.ndarray)
    assert isinstance(immuno_labels, np.ndarray)
    assert isinstance(link_logits, np.ndarray)
    assert isinstance(link_labels, np.ndarray)
    immuno_score = 1.0 / (1.0 + np.exp(-immuno_logits))
    metrics['immunogenicity/auprc'] = safe_auprc(immuno_labels, immuno_score)
    metrics['glycosylation/macro_f1'] = macro_f1(link_labels, link_logits.argmax(axis=-1))
    metrics.update(interaction_metrics_value)
    metrics['selection/composite'] = scalar_mean([metrics['taxonomy/mean_macro_f1'], metrics['immunogenicity/auprc'], metrics['glycosylation/macro_f1'], metrics['interaction/spearman']])
    metrics['selection/glycan_only_composite'] = scalar_mean([metrics['taxonomy/mean_macro_f1'], metrics['immunogenicity/auprc'], metrics['glycosylation/macro_f1']])
    return metrics

def flatten_result(result: Mapping[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {'framework_name': result['framework_name']}
    manifest = result.get('manifest', {})
    if isinstance(manifest, Mapping):
        for (key, value) in manifest.items():
            row[f'manifest/{key}'] = value
    for split in ('valid', 'test'):
        metrics = result.get(split, {})
        if not isinstance(metrics, Mapping):
            continue
        for (key, value) in metrics.items():
            row[f'{split}/{key}'] = value
    return row

def metric_line(metrics: Mapping[str, float]) -> str:
    return f"composite={metrics['selection/composite']:.4f} tax={metrics['taxonomy/mean_macro_f1']:.4f} immuno={metrics['immunogenicity/auprc']:.4f} glycos={metrics['glycosylation/macro_f1']:.4f} inter={metrics['interaction/spearman']:.4f}"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--protein-features', type=Path, default=Path('data/protein_features.pt'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--framework-name', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--taxonomy-light-checkpoint', type=Path, required=True)
    parser.add_argument('--taxonomy-light-variant', choices=sorted(VARIANTS), default='interaction_rank')
    parser.add_argument('--functional-atom-checkpoint', type=Path, required=True)
    parser.add_argument('--interaction-light-checkpoint', type=Path, required=True)
    parser.add_argument('--interaction-light-variant', choices=sorted(VARIANTS), default='interaction_rank')
    parser.add_argument('--interaction-atom-checkpoint', type=Path)
    parser.add_argument('--graph-cache-file', type=Path, required=True)
    parser.add_argument('--taxonomy-coarse-tau', type=float, required=True)
    parser.add_argument('--taxonomy-fine-tau', type=float, required=True)
    parser.add_argument('--taxonomy-prior-smoothing', type=float, required=True)
    parser.add_argument('--interaction-alpha', type=float, required=True)
    parser.add_argument('--no-interaction-zscore', action='store_true')
    parser.add_argument('--functional-atom-interaction-head', choices=['base', 'advanced'], default='base')
    parser.add_argument('--interaction-atom-interaction-head', choices=['base', 'advanced'], default='base')
    parser.add_argument('--atom-use-motif-fusion', action='store_true')
    parser.add_argument('--eval-batch-size', type=int, required=True)
    parser.add_argument('--cache-batch-size', type=int, required=True)
    parser.add_argument('--max-len', type=int, required=True)
    parser.add_argument('--hash-dim', type=int, required=True)
    parser.add_argument('--hidden-dim', type=int, required=True)
    parser.add_argument('--token-dim', type=int, required=True)
    parser.add_argument('--num-layers', type=int, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--gnn-hidden-dim', type=int, required=True)
    parser.add_argument('--gnn-layers', type=int, required=True)
    parser.add_argument('--head-dim', type=int, required=True)
    parser.add_argument('--num-unit', type=int, default=0)
    parser.add_argument('--readout', choices=['mean', 'sum', 'dual'], default='dual')
    parser.add_argument('--readout-type', choices=['mono', 'all'], default='mono')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith('cuda') else 'cpu')
    interaction_atom_checkpoint = args.interaction_atom_checkpoint or args.functional_atom_checkpoint
    data = DataStore(args.data_dir, max_len=args.max_len, hash_dim=args.hash_dim, protein_features=args.protein_features)
    print('[data]', json.dumps(data.summary(), indent=2, sort_keys=True), flush=True)
    light_bank = TensorBank(data, device)
    taxonomy_model = load_light_branch(args, data, args.taxonomy_light_checkpoint, args.taxonomy_light_variant, device)
    taxonomy_features = cache_light_features(taxonomy_model, light_bank, len(data.glycan_to_id), args.cache_batch_size)
    interaction_light_model = load_light_branch(args, data, args.interaction_light_checkpoint, args.interaction_light_variant, device)
    interaction_light_features = cache_light_features(interaction_light_model, light_bank, len(data.glycan_to_id), args.cache_batch_size)
    (glycan_module, all_atom_glycan_module, glycanaa_cls) = load_glycanaa_modules()
    graph_cache = AllAtomGraphCache(data, all_atom_glycan_module)
    graph_cache.load(args.graph_cache_file)
    atom_bank = AllAtomTensorBank(data, device)
    functional_atom_model = load_atom_branch(args, data, args.functional_atom_checkpoint, glycan_module, all_atom_glycan_module, glycanaa_cls, args.functional_atom_interaction_head, device)
    functional_atom_features = cache_atom_features(functional_atom_model, graph_cache, atom_bank, len(data.glycan_to_id), args.cache_batch_size)
    if interaction_atom_checkpoint == args.functional_atom_checkpoint:
        interaction_atom_model = functional_atom_model
        interaction_atom_features = functional_atom_features
    else:
        interaction_atom_model = load_atom_branch(args, data, interaction_atom_checkpoint, glycan_module, all_atom_glycan_module, glycanaa_cls, args.interaction_atom_interaction_head, device)
        interaction_atom_features = cache_atom_features(interaction_atom_model, graph_cache, atom_bank, len(data.glycan_to_id), args.cache_batch_size)
    graph_cache.save(args.graph_cache_file)
    protein_features = data.protein_features.detach().cpu()
    priors = class_log_prior(data, args.taxonomy_prior_smoothing)
    taxonomy_valid = predict_model_split(taxonomy_model, taxonomy_features, protein_features, data, 'valid', args.eval_batch_size, device)
    taxonomy_test = predict_model_split(taxonomy_model, taxonomy_features, protein_features, data, 'test', args.eval_batch_size, device)
    functional_valid = predict_model_split(functional_atom_model, functional_atom_features, protein_features, data, 'valid', args.eval_batch_size, device)
    functional_test = predict_model_split(functional_atom_model, functional_atom_features, protein_features, data, 'test', args.eval_batch_size, device)
    inter_light_valid = predict_model_split(interaction_light_model, interaction_light_features, protein_features, data, 'valid', args.eval_batch_size, device)
    inter_light_test = predict_model_split(interaction_light_model, interaction_light_features, protein_features, data, 'test', args.eval_batch_size, device)
    inter_atom_valid = predict_model_split(interaction_atom_model, interaction_atom_features, protein_features, data, 'valid', args.eval_batch_size, device)
    inter_atom_test = predict_model_split(interaction_atom_model, interaction_atom_features, protein_features, data, 'test', args.eval_batch_size, device)
    interaction_stats = None
    if not args.no_interaction_zscore:
        interaction_stats = {'light': (float(np.mean(inter_light_valid['interaction_pred'])), float(np.std(inter_light_valid['interaction_pred']))), 'atom': (float(np.mean(inter_atom_valid['interaction_pred'])), float(np.std(inter_atom_valid['interaction_pred'])))}
    valid_tax = calibrated_taxonomy_metrics(taxonomy_valid, data, priors, args.taxonomy_coarse_tau, args.taxonomy_fine_tau)
    test_tax = calibrated_taxonomy_metrics(taxonomy_test, data, priors, args.taxonomy_coarse_tau, args.taxonomy_fine_tau)
    interaction_alpha = args.interaction_alpha
    valid_inter = interaction_metrics(inter_light_valid, inter_atom_valid, interaction_alpha, interaction_stats)
    test_inter = interaction_metrics(inter_light_test, inter_atom_test, interaction_alpha, interaction_stats)
    valid_metrics = full_metrics(valid_tax, functional_valid, valid_inter)
    test_metrics = full_metrics(test_tax, functional_test, test_inter)
    manifest = BranchManifest(taxonomy_branch=str(args.taxonomy_light_checkpoint), functional_glycan_branch=str(args.functional_atom_checkpoint), interaction_light_branch=str(args.interaction_light_checkpoint), interaction_atom_branch=str(interaction_atom_checkpoint), taxonomy_coarse_tau=args.taxonomy_coarse_tau, taxonomy_fine_tau=args.taxonomy_fine_tau, interaction_alpha=interaction_alpha, interaction_zscore=not args.no_interaction_zscore)
    result: Dict[str, object] = {'framework_name': args.framework_name, 'manifest': asdict(manifest), 'valid': valid_metrics, 'test': test_metrics, 'args': vars(args), 'graph_cache': graph_cache.summary()}
    with (args.output_dir / 'result.json').open('w') as fout:
        json.dump(json_ready(result), fout, indent=2, sort_keys=True)
    row = flatten_result(result)
    with (args.output_dir / 'summary.csv').open('w', newline='') as fout:
        keys = sorted(row)
        writer = csv.DictWriter(fout, fieldnames=keys)
        writer.writeheader()
        writer.writerow(row)
    print(f'[{args.framework_name}] valid {metric_line(valid_metrics)}', flush=True)
    print(f'[{args.framework_name}] test  {metric_line(test_metrics)}', flush=True)
    print(f"[done] wrote {args.output_dir / 'summary.csv'}", flush=True)
if __name__ == '__main__':
    main()
