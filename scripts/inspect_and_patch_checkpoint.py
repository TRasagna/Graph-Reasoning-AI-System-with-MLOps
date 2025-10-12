"""Utility to inspect and (optionally) patch a PyTorch model checkpoint.

Usage examples (PowerShell):

# Inspect checkpoint metadata and model_config
python .\scripts\inspect_and_patch_checkpoint.py -p .\models\r-gcn_1760253457\final_model.pt --inspect

# Create a backup and expand relation embeddings to 237 relations
python .\scripts\inspect_and_patch_checkpoint.py -p .\models\r-gcn_1760253457\final_model.pt --expand 237 --out .\models\r-gcn_1760253457\final_model_expanded.pt

Note: this script requires PyTorch installed in the Python environment you run it in.
"""
import argparse
from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def inspect_checkpoint(path: Path):
    import torch
    ckpt = torch.load(path, map_location='cpu')

    logger.info(f"Loaded checkpoint keys: {list(ckpt.keys())}")
    model_config = ckpt.get('model_config')
    metadata = ckpt.get('metadata')

    logger.info(f"model_config: {model_config}")
    logger.info(f"metadata present: {metadata is not None}")
    if metadata is not None:
        logger.info(f"metadata keys: {list(metadata.keys())}")

    # Print state_dict shapes for embeddings if available
    state = ckpt.get('model_state_dict')
    if state is not None:
        for k, v in state.items():
            try:
                shape = tuple(v.size())
            except Exception:
                shape = None
            if 'embedding' in k or 'emb' in k or 'weight' in k:
                logger.info(f"state key: {k}, shape: {shape}")

    return ckpt


def expand_relation_embeddings(ckpt: dict, new_num_relations: int):
    import torch
    import torch.nn as nn
    state = ckpt.get('model_state_dict')
    if state is None:
        raise ValueError('checkpoint has no model_state_dict')

    # Common key for relation embedding weight
    candidate_keys = [k for k in state.keys() if 'relation_embeddings.weight' in k or 'relation_embeddings' in k]
    if not candidate_keys:
        raise ValueError('Could not find relation_embeddings.weight key in state_dict')

    key = candidate_keys[0]
    old_weight = state[key]
    old_num_rel, emb_dim = old_weight.size()

    if new_num_relations <= old_num_rel:
        raise ValueError(f'New num_relations ({new_num_relations}) must be greater than existing ({old_num_rel})')

    logger.info(f'Expanding relation embeddings from {old_num_rel} to {new_num_relations} (dim={emb_dim})')

    # Create new weight tensor
    new_weight = torch.empty((new_num_relations, emb_dim), dtype=old_weight.dtype)
    # Copy old rows
    new_weight[:old_num_rel] = old_weight
    # Initialize new rows
    nn.init.xavier_uniform_(new_weight[old_num_rel:])

    # Replace in state_dict
    state[key] = new_weight

    # Update model_config if present
    if 'model_config' in ckpt:
        ckpt['model_config']['num_relations'] = new_num_relations

    return ckpt


def main():
    parser = argparse.ArgumentParser(description='Inspect and optionally patch model checkpoint')
    parser.add_argument('-p', '--path', required=True, type=str, help='Path to checkpoint .pt file')
    parser.add_argument('--inspect', action='store_true', help='Inspect checkpoint and print model_config/metadata')
    parser.add_argument('--expand', type=int, help='Expand relation embeddings to this number and save to --out')
    parser.add_argument('--out', type=str, help='Output path for modified checkpoint (if --expand used)')

    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        logger.error(f'Checkpoint not found: {path}')
        return

    # Load checkpoint
    ckpt = inspect_checkpoint(path)

    if args.expand:
        if not args.out:
            logger.error('--out must be provided when using --expand')
            return
        out_path = Path(args.out)
        # Make backup
        backup = path.with_suffix(path.suffix + '.bak')
        shutil.copy2(path, backup)
        logger.info(f'Created backup: {backup}')

        try:
            new_ckpt = expand_relation_embeddings(ckpt, args.expand)
            # Save
            torch = __import__('torch')
            torch.save(new_ckpt, out_path)
            logger.info(f'Saved expanded checkpoint to {out_path}')
        except Exception as e:
            logger.error(f'Failed to expand checkpoint: {e}')
            # Restore backup if something went wrong
            if backup.exists():
                shutil.copy2(backup, path)
                logger.info('Restored backup to original checkpoint')


if __name__ == '__main__':
    main()
