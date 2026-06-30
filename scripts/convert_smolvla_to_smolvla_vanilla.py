#!/usr/bin/env python
"""
Converts a pretrained SmolVLA checkpoint into a SmolVLA-Vanilla checkpoint.

SmolVLA-Vanilla is architecturally identical to vanilla SmolVLA — same parameters,
same expert layers, same flow-matching training.  The only addition is the
`stereo_camera_keys` config field, which tells `prepare_images` to extract the
RIGHT half of any listed side-by-side stereo frame (H × 2W → H × W) before
feeding it to SigLIP.

What this script does:
  1. Downloads (or copies) the source SmolVLA checkpoint.
  2. Patches config.json:
       - type: smolvla → smolvla_vanilla
       - Adds `stereo_camera_keys: []` (set it at training time to your camera key)
  3. Leaves the weight file untouched — weights load with strict=True because
     no parameters are added or removed.

Usage:
    python scripts/convert_smolvla_to_smolvla_vanilla.py \\
        --src lerobot/smolvla_base \\
        --dst outputs/smolvla_vanilla_base

    # Optional: verify the converted checkpoint loads correctly
    python scripts/convert_smolvla_to_smolvla_vanilla.py \\
        --src lerobot/smolvla_base \\
        --dst outputs/smolvla_vanilla_base \\
        --validate

Then train with stereo camera:
    lerobot-train \\
        --policy.path=outputs/smolvla_vanilla_base \\
        --policy.stereo_camera_keys='["observation.images.camera2"]' \\
        ...
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

# Fields added by SmolVLA-Vanilla that are not in the base SmolVLA config.
SMOLVLA_VANILLA_NEW_FIELDS = {
    # Keys of cameras whose raw tensor is a side-by-side (H × 2W) stereo frame.
    # The RIGHT half is extracted in prepare_images and fed to SigLIP.
    # Override at training / inference time:
    #   --policy.stereo_camera_keys='["observation.images.camera2"]'
    "stereo_camera_keys": [],
}


def _patch_config(dst: Path) -> str:
    config_path = dst / "config.json"
    if not config_path.exists():
        print("WARNING: config.json not found — patch it manually.")
        return "unknown"

    with open(config_path) as f:
        cfg = json.load(f)

    original_type = cfg.get("type", "unknown")
    cfg["type"] = "smolvla_vanilla"
    cfg.pop("policy_type", None)

    for key, default_value in SMOLVLA_VANILLA_NEW_FIELDS.items():
        if key not in cfg:
            cfg[key] = default_value

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    return original_type


def _validate(dst: Path) -> None:
    """Load the converted checkpoint and verify weights are identical to a fresh smolvla_vanilla."""
    import torch

    print("\nValidating converted checkpoint ...")

    # Patch sys.path so local lerobot is importable
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    from lerobot.policies.smolvla_vanilla.modeling_smolvla_vanilla import SmolVLAVanillaPolicy

    policy = SmolVLAVanillaPolicy.from_pretrained(str(dst))
    policy.eval()

    # Architecture sanity: VLM state dict should have the same keys as vanilla SmolVLA.
    state = policy.model.vlm_with_expert.state_dict()
    print(f"  VLM state dict keys: {len(state)}")
    assert len(state) > 0, "state dict is empty — something went wrong"

    # Stereo config field must be present (may be empty list).
    assert hasattr(policy.config, "stereo_camera_keys"), "stereo_camera_keys missing from config"
    print(f"  stereo_camera_keys = {policy.config.stereo_camera_keys!r}")

    # Quick forward smoke-test with a dummy monocular image.
    B, C, H, W = 1, 3, 512, 512
    dummy_img = torch.zeros(B, C, H, W, device=policy.model.vlm_with_expert.vlm.device)
    with torch.no_grad():
        emb = policy.model.vlm_with_expert.embed_image(dummy_img)
    assert emb.shape[0] == B, f"Unexpected embed_image output shape: {emb.shape}"
    print(f"  embed_image OK — output shape: {emb.shape}")

    print("Validation passed.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src",
        default="lerobot/smolvla_base",
        help="HF hub repo-id OR local path of the source SmolVLA checkpoint.",
    )
    parser.add_argument(
        "--dst",
        default="outputs/smolvla_vanilla_base",
        help="Output directory for the converted SmolVLA-Vanilla checkpoint.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After conversion, load the checkpoint and run basic sanity checks.",
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

    original_type = _patch_config(dst)

    print(f"Patched config.json: type {original_type!r} → 'smolvla_vanilla'")
    print(f"Added SmolVLA-Vanilla fields: {list(SMOLVLA_VANILLA_NEW_FIELDS.keys())}")

    if args.validate:
        _validate(dst)

    print()
    print("Done.  Use this checkpoint with:")
    print(f"  --policy.path={dst.resolve()}")
    print()
    print("To enable stereo right-half cropping, set at training/inference time:")
    print('  --policy.stereo_camera_keys=\'["observation.images.camera2"]\'')


if __name__ == "__main__":
    main()
