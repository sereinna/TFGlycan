"""Taxonomy branch fine-tuning."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .light import TAXONOMY_TASKS, VARIANTS, DataStore, TensorBank, VariantConfig, json_ready, macro_f1, sample_indices, scalar_mean, stable_hash, make_model, set_seed

def make_light_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(hidden_dim=args.hidden_dim, token_dim=args.token_dim, num_layers=args.num_layers, dropout=args.dropout)

def taxonomy_training_split(data: DataStore, include_valid: bool) -> Dict[str, np.ndarray]:
    if not include_valid:
        return data.taxonomy['train']
    return {'glycan': np.concatenate([data.taxonomy['train']['glycan'], data.taxonomy['valid']['glycan']]), 'labels': np.concatenate([data.taxonomy['train']['labels'], data.taxonomy['valid']['labels']], axis=0)}

def class_log_prior(labels: np.ndarray, data: DataStore, smoothing: float, device: torch.device) -> Dict[str, Tensor]:
    priors = {}
    for (i, task) in enumerate(TAXONOMY_TASKS):
        num_class = data.num_classes[f'taxonomy/{task}']
        counts = np.bincount(labels[:, i].astype(np.int64), minlength=num_class).astype(np.float64)
        probs = (counts + smoothing) / (counts.sum() + smoothing * num_class)
        priors[task] = torch.as_tensor(np.log(np.clip(probs, 1e-12, None)), dtype=torch.float32, device=device)
    return priors

def build_class_indices(labels: np.ndarray, data: DataStore) -> Dict[str, List[np.ndarray]]:
    result: Dict[str, List[np.ndarray]] = {}
    for (i, task) in enumerate(TAXONOMY_TASKS):
        classes = []
        for cls in range(data.num_classes[f'taxonomy/{task}']):
            idx = np.flatnonzero(labels[:, i] == cls).astype(np.int64)
            if len(idx):
                classes.append(idx)
            else:
                classes.append(np.empty(0, dtype=np.int64))
        result[task] = classes
    return result

def sample_taxonomy_class_aware(rng: np.random.Generator, data: DataStore, class_indices: Mapping[str, Sequence[np.ndarray]], batch_size: int, level_mode: str) -> np.ndarray:
    if level_mode == 'all':
        levels = TAXONOMY_TASKS
    elif level_mode == 'fine':
        levels = TAXONOMY_TASKS[4:]
    elif level_mode == 'mixed':
        levels = TAXONOMY_TASKS if rng.random() < 0.5 else TAXONOMY_TASKS[4:]
    else:
        raise ValueError(f'Unknown level_mode `{level_mode}`')
    chosen = []
    for _ in range(batch_size):
        task = str(rng.choice(levels))
        non_empty = [i for (i, idx) in enumerate(class_indices[task]) if len(idx)]
        cls = int(rng.choice(non_empty))
        options = class_indices[task][cls]
        chosen.append(int(rng.choice(options)))
    return np.asarray(chosen, dtype=np.int64)

def taxonomy_ce_loss(logits: Mapping[str, Tensor], labels: Tensor, log_priors: Mapping[str, Tensor], logit_adjust_tau: float) -> Tensor:
    losses = []
    for (i, task) in enumerate(TAXONOMY_TASKS):
        task_logits = logits[task]
        if logit_adjust_tau > 0:
            task_logits = task_logits + logit_adjust_tau * log_priors[task].unsqueeze(0)
        losses.append(F.cross_entropy(task_logits, labels[:, i]))
    return torch.stack(losses).mean()

def train_one_epoch(model, data: DataStore, train_split: Mapping[str, np.ndarray], bank: TensorBank, optimizer: torch.optim.Optimizer, rng: np.random.Generator, class_indices: Mapping[str, Sequence[np.ndarray]], log_priors: Mapping[str, Tensor], args: argparse.Namespace) -> Dict[str, float]:
    model.train()
    split = train_split
    losses = []
    for step in range(args.steps_per_epoch):
        optimizer.zero_grad(set_to_none=True)
        if args.taxonomy_sampler == 'random':
            local_idx = sample_indices(rng, len(split['glycan']), args.batch_size)
        else:
            local_idx = sample_taxonomy_class_aware(rng, data, class_indices, args.batch_size, args.class_aware_levels)
        glycan_idx = split['glycan'][local_idx]
        labels = torch.as_tensor(split['labels'][local_idx], dtype=torch.long, device=bank.device)
        (token_ids, mask, features) = bank.glycan_batch(glycan_idx)
        glycan_feature = model.encode_glycan(token_ids, mask, features)
        logits = model.taxonomy_logits(glycan_feature)
        loss = taxonomy_ce_loss(logits, labels, log_priors, args.logit_adjust_tau)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.train_log_every > 0 and (step + 1) % args.train_log_every == 0:
            print(f'[train] step {step + 1}/{args.steps_per_epoch} loss={losses[-1]:.4f}', flush=True)
    return {'train_loss/taxonomy': scalar_mean(losses)}

@torch.no_grad()
def evaluate_taxonomy_split(model, data: DataStore, bank: TensorBank, split: str, batch_size: int) -> Dict[str, float]:
    model.eval()
    split_data = data.taxonomy[split]
    preds: Dict[str, List[np.ndarray]] = {task: [] for task in TAXONOMY_TASKS}
    for start in range(0, len(split_data['glycan']), batch_size):
        end = start + batch_size
        glycan_idx = split_data['glycan'][start:end]
        (token_ids, mask, features) = bank.glycan_batch(glycan_idx)
        glycan_feature = model.encode_glycan(token_ids, mask, features)
        logits = model.taxonomy_logits(glycan_feature)
        for task in TAXONOMY_TASKS:
            preds[task].append(logits[task].argmax(dim=-1).detach().cpu().numpy())
    metrics: Dict[str, float] = {}
    scores = []
    labels = split_data['labels']
    for (i, task) in enumerate(TAXONOMY_TASKS):
        pred = np.concatenate(preds[task]) if preds[task] else np.empty(0, dtype=np.int64)
        score = macro_f1(labels[:, i], pred)
        metrics[f'taxonomy/{task}_macro_f1'] = score
        scores.append(score)
    metrics['taxonomy/mean_macro_f1'] = scalar_mean(scores)
    return metrics

def flatten_result(result: Mapping[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {'variant': result['variant'], 'best_epoch': result['best_epoch'], 'elapsed_sec': result['elapsed_sec']}
    for prefix in ('best_valid', 'test'):
        metrics = result[prefix]
        assert isinstance(metrics, Mapping)
        for (key, value) in metrics.items():
            row[f'{prefix}/{key}'] = value
    return row

def write_summary(result: Mapping[str, object], output_dir: Path) -> None:
    row = flatten_result(result)
    with (output_dir / 'summary.csv').open('w', newline='') as fout:
        keys = sorted(row)
        writer = csv.DictWriter(fout, fieldnames=keys)
        writer.writeheader()
        writer.writerow(row)

def run(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith('cuda') else 'cpu')
    data = DataStore(args.data_dir, max_len=args.max_len, hash_dim=args.hash_dim, protein_features=args.protein_features)
    bank = TensorBank(data, device)
    model = make_model(make_light_args(args), data, VARIANTS[args.variant], device)
    state = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    print(f'[model] loaded init {args.init_checkpoint} variant={args.variant}', flush=True)
    output_dir = args.output_dir / args.variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_split = taxonomy_training_split(data, args.include_valid_in_train)
    class_indices = build_class_indices(train_split['labels'], data)
    log_priors = class_log_prior(train_split['labels'], data, args.prior_smoothing, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    rng = np.random.default_rng(args.seed + stable_hash(args.variant_name) % 100000)
    best_state = None
    best_valid = None
    best_epoch = -1
    history = []
    history_path = output_dir / 'history.jsonl'
    if history_path.exists():
        history_path.unlink()
    start_time = time.time()
    if args.eval_initial:
        valid_metrics = evaluate_taxonomy_split(model, data, bank, 'valid', args.eval_batch_size)
        best_state = copy.deepcopy(model.state_dict())
        best_valid = valid_metrics
        best_epoch = 0
        torch.save(best_state, output_dir / 'best_model.pt')
        row = {'epoch': 0, 'lr': scheduler.get_last_lr()[0], **{f'valid/{k}': v for (k, v) in valid_metrics.items()}}
        history.append(row)
        with history_path.open('a') as fout:
            fout.write(json.dumps(json_ready(row), sort_keys=True) + '\n')
        print(f"[{args.variant_name}] epoch 000 valid_tax={valid_metrics['taxonomy/mean_macro_f1']:.4f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, data, train_split, bank, optimizer, rng, class_indices, log_priors, args)
        scheduler.step()
        valid_metrics = evaluate_taxonomy_split(model, data, bank, 'valid', args.eval_batch_size)
        row = {'epoch': epoch, 'lr': scheduler.get_last_lr()[0], **train_metrics, **{f'valid/{k}': v for (k, v) in valid_metrics.items()}}
        history.append(row)
        with history_path.open('a') as fout:
            fout.write(json.dumps(json_ready(row), sort_keys=True) + '\n')
        if best_valid is None or valid_metrics['taxonomy/mean_macro_f1'] > best_valid['taxonomy/mean_macro_f1']:
            best_valid = valid_metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / 'best_model.pt')
        if args.select_last:
            best_valid = valid_metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / 'best_model.pt')
        print(f"[{args.variant_name}] epoch {epoch:03d} valid_tax={valid_metrics['taxonomy/mean_macro_f1']:.4f} loss={train_metrics['train_loss/taxonomy']:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_taxonomy_split(model, data, bank, 'test', args.eval_batch_size)
    elapsed = time.time() - start_time
    result = {'variant': args.variant_name, 'best_epoch': best_epoch, 'elapsed_sec': elapsed, 'best_valid': best_valid, 'test': test_metrics, 'args': vars(args)}
    with (output_dir / 'result.json').open('w') as fout:
        json.dump(json_ready(result), fout, indent=2, sort_keys=True)
    with history_path.open('w') as fout:
        for row in history:
            fout.write(json.dumps(json_ready(row), sort_keys=True) + '\n')
    if best_state is not None:
        torch.save(best_state, output_dir / 'best_model.pt')
    write_summary(result, args.output_dir)
    print(f"[done] best_epoch={best_epoch} test_tax={test_metrics['taxonomy/mean_macro_f1']:.4f} wrote {output_dir / 'result.json'}", flush=True)
    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--protein-features', type=Path, default=Path('data/protein_features.pt'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variant-name', required=True)
    parser.add_argument('--init-checkpoint', type=Path, required=True)
    parser.add_argument('--variant', choices=sorted(VARIANTS), default='interaction_rank')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-initial', action='store_true')
    parser.add_argument('--include-valid-in-train', action='store_true')
    parser.add_argument('--select-last', action='store_true')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--steps-per-epoch', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--eval-batch-size', type=int, required=True)
    parser.add_argument('--taxonomy-sampler', choices=['random', 'class_aware'], default='random')
    parser.add_argument('--class-aware-levels', choices=['all', 'fine', 'mixed'], default='mixed')
    parser.add_argument('--logit-adjust-tau', type=float, required=True)
    parser.add_argument('--prior-smoothing', type=float, required=True)
    parser.add_argument('--max-len', type=int, required=True)
    parser.add_argument('--hash-dim', type=int, required=True)
    parser.add_argument('--hidden-dim', type=int, required=True)
    parser.add_argument('--token-dim', type=int, required=True)
    parser.add_argument('--num-layers', type=int, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, required=True)
    parser.add_argument('--grad-clip', type=float, required=True)
    parser.add_argument('--train-log-every', type=int, default=0)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args)
if __name__ == '__main__':
    main()
