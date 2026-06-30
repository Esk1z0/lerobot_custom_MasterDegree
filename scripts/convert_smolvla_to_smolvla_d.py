#!/usr/bin/env python
"""
Converts a pretrained SmolVLA checkpoint (e.g. lerobot/smolvla_base) into a
SmolVLA-D checkpoint ready for fine-tuning with stereo depth injection.

What this script does:
  1. Downloads (or copies) the vanilla SmolVLA checkpoint.
  2. Patches config.json:
       - type: smolvla → smolvla_d
       - Adds all SmolVLA-D specific fields (depth injection, stereo camera)
         with their default values so the config is explicit and easy to edit.
  3. Leaves the weight file untouched.
     When SmolVLA-D loads from the converted checkpoint, PreTrainedPolicy uses
     strict=False, so the vanilla weights are loaded into matching modules and
     the new depth modules (PointCloudEncoder + DepthInjector) keep their
     zero-init values — ready to be trained.

Usage:
    python scripts/convert_smolvla_to_smolvla_d.py \\
        --src lerobot/smolvla_base \\
        --dst outputs/smolvla_d_base

Then train:
    lerobot-train \\
        --policy.path=outputs/smolvla_d_base \\
        --policy.stereo_camera_keys='["observation.images.camera2"]' \\
        ...
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

# New fields added by SmolVLA-D that are not in the vanilla SmolVLA config.
# These are written with their defaults so the converted config is self-documenting.
SMOLVLA_D_NEW_FIELDS = {
    # -----------------------------------------------------------------------
    # Stereo camera handling
    # The key(s) here are split at the horizontal midpoint:
    #   left half  → VLM RGB pathway
    #   full frame → depth processing pipeline
    # Override at training time with --policy.stereo_camera_keys='[...]'
    # -----------------------------------------------------------------------
    "stereo_camera_keys": [],

    # -----------------------------------------------------------------------
    # PointVLA-style depth injection (action expert layers 6, 7, 8)
    # -----------------------------------------------------------------------
    "depth_injection_layers": [6, 7, 8],
    "depth_point_encoder_channels": [64, 128, 256, 512],
    "depth_embed_dim": 128,
    "depth_num_sample_points": 1024,

    # During training, zero-out depth features with this probability so the
    # model stays robust when depth is missing at inference time.
    "depth_dropout_prob": 0.2,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src",
        default="lerobot/smolvla_base",
        help="HF hub repo-id OR local path of the source SmolVLA checkpoint.",
    )
    parser.add_argument(
        "--dst",
        default="outputs/smolvla_d_base",
        help="Output directory for the converted SmolVLA-D checkpoint.",
    )
    args = parser.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    src = args.src
    if not Path(src).exists():
        print(f"Downloading {src} from HuggingFace Hub...")
        src = snapshot_download(repo_id=src)
    src = Path(src)

    print(f"Copying files from {src} to {dst} ...")
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)

    config_path = dst / "config.json"
    if not config_path.exists():
        print("WARNING: config.json not found — patch it manually.")
        return

    with open(config_path) as f:
        cfg = json.load(f)

    original_type = cfg.get("type", "unknown")

    # Patch type
    cfg["type"] = "smolvla_d"
    cfg.pop("policy_type", None)   # remove stale field if present

    # Inject SmolVLA-D specific fields (only if not already present so
    # re-running the script on an already-converted checkpoint is idempotent)
    for key, default_value in SMOLVLA_D_NEW_FIELDS.items():
        if key not in cfg:
            cfg[key] = default_value

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Patched config.json: type {original_type!r} → 'smolvla_d'")
    print(f"Added SmolVLA-D fields: {list(SMOLVLA_D_NEW_FIELDS.keys())}")
    print()
    print("Done.  Use this checkpoint with:")
    print(f"  --policy.path={dst.resolve()}")
    print()
    print("Remember to set --policy.stereo_camera_keys at training time, e.g.:")
    print("  --policy.stereo_camera_keys='[\"observation.images.camera2\"]'")


if __name__ == "__main__":
    main()
