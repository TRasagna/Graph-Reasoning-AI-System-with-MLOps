"""Repair model checkpoint shapes to match relation_embeddings rows.

This script will:
- load a checkpoint (.pt)
- detect the number of relation rows from 'relation_embeddings.weight'
- for any 'rgcn_layers.*.comp' tensor with fewer rows, expand it to match
  relation count by copying existing rows and Xavier-initializing new rows.
- update 'model_config.num_relations' to the relation_embeddings count
- save a backup and the repaired checkpoint to --out (or overwrite with --inplace)

Usage (PowerShell):
python .\scripts\repair_checkpoint.py --path .\models\r-gcn_1760253457\final_model.pt --inplace

Requires a Python env with torch installed (use the project venv).
"""
import argparse
from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def repair_checkpoint(path: Path, out: Path = None, inplace: bool = False):
    import torch
    import torch.nn as nn

    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('model_state_dict')
    if state is None:
        raise RuntimeError('checkpoint has no model_state_dict')

    # Determine desired relation count from relation_embeddings if present
    rel_key = next((k for k in state.keys() if 'relation_embeddings' in k and 'weight' in k), None)
    if rel_key is None:
        raise RuntimeError('Could not find relation_embeddings.weight in checkpoint')

    rel_count = state[rel_key].size(0)
    logger.info(f'Inferred relation count from {rel_key}: {rel_count}')

    # Find rgcn comp keys
    comp_keys = [k for k in state.keys() if 'rgcn_layers' in k and k.endswith('.comp')]
    logger.info(f'Found rgcn comp keys: {comp_keys}')

    modified = False
    for k in comp_keys:
        tensor = state[k]
        try:
            rows, cols = tensor.size()
        except Exception:
            logger.warning(f'Could not interpret shape for {k}, skipping')
            continue

        if rows == rel_count:
            logger.info(f'{k} already has {rows} rows, OK')
            continue

        if rows < rel_count:
            logger.info(f'Expanding {k} from {rows} -> {rel_count}')
            new_tensor = torch.empty((rel_count, cols), dtype=tensor.dtype)
            new_tensor[:rows] = tensor
            nn.init.xavier_uniform_(new_tensor[rows:])
            state[k] = new_tensor
            modified = True
        else:
            # rows > rel_count: we will truncate (warn)
            logger.warning(f'{k} has {rows} rows which is > inferred rel_count {rel_count}; truncating')
            state[k] = tensor[:rel_count].clone()
            modified = True

    if modified:
        # Update model_config
        if 'model_config' in ckpt:
            ckpt['model_config']['num_relations'] = rel_count

        # Backup original
        backup = path.with_suffix(path.suffix + '.bak')
        shutil.copy2(path, backup)
        logger.info(f'Created backup: {backup}')

        target = out if out else path
        torch.save(ckpt, target)
        logger.info(f'Saved repaired checkpoint to {target}')
    else:
        logger.info('No modifications required')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', required=True, type=str)
    parser.add_argument('--out', type=str, help='Output path for repaired checkpoint')
    parser.add_argument('--inplace', action='store_true', help='Overwrite original checkpoint (backup will be created)')

    args = parser.parse_args()
    p = Path(args.path)
    if not p.exists():
        print(f'Checkpoint not found: {p}')
        return

    out = Path(args.out) if args.out else None
    if args.inplace:
        out = p

    repair_checkpoint(p, out=out, inplace=args.inplace)


if __name__ == '__main__':
    main()
