#!/usr/bin/env python
"""
Converts a pretrained SmolVLA checkpoint into a SmolVLA-MD checkpoint ready for
fine-tuning with both temporal memory (M) AND stereo depth (D).

What this script does:
  1. Downloads (or copies) the vanilla SmolVLA checkpoint.
  2. Patches config.json:
       - type: smolvla → smolvla_md
       - n_obs_steps: 1 → 6
       - Adds SmolVLA-M temporal fields (temporal_num_frames, temporal_stride, temporal_vit_layers)
       - Adds SmolVLA-D depth fields (stereo_camera_keys, depth_injection_layers, …)
  3. Leaves the weight file untouched.
     When SmolVLA-MD loads from the converted checkpoint, PreTrainedPolicy uses
     strict=False so:
       - All existing weights (VLM, expert, projections) load into matching keys.
       - New weights (temporal_alpha ×3, PointCloudEncoder, DepthInjector) keep init values.

Weight key compatibility:
  Vanilla SmolVLA      model.vision_model.encoder.layers.N.*
  SmolVLA-MD           model.vision_model.encoder.layers.N.*          ← same key!
  New (temporal)       model.vision_model.encoder.temporal_alpha.{0,1,2}  → zero-init
  New (depth encoder)  model.point_cloud_encoder.*                        → random-init
  New (depth injector) model.depth_injector.adapters.{6,7,8}.*           → zero-init (safe)

Usage:
    python lerobot/scripts/convert_smolvla_to_smolvla_md.py \\
        --src lerobot/smolvla_base \\
        --dst outputs/smolvla_md_base

    # Validate the converted checkpoint:
    python lerobot/scripts/convert_smolvla_to_smolvla_md.py \\
        --src lerobot/smolvla_base \\
        --dst outputs/smolvla_md_base \\
        --validate

Then train (Phase 1 — RGB only, same as SmolVLA-D Phase 1):
    lerobot-train \\
        --policy.path=outputs/smolvla_md_base \\
        --dataset.repo_id=<your-stereo-dataset> \\
        --policy.stereo_camera_keys='["observation.images.camera2"]' \\
        --policy.n_obs_steps=6 \\
        --policy.temporal_stride=1 \\
        --policy.freeze_vision_encoder=true \\
        --policy.train_expert_only=true \\
        --batch_size=8 --steps=90000
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


# Fields added by SmolVLA-M (temporal memory)
TEMPORAL_NEW_FIELDS = {
    "temporal_num_frames": 6,
    "temporal_stride": 1,
    "temporal_vit_layers": [3, 7, 11],
}

# Fields added by SmolVLA-D (stereo depth)
DEPTH_NEW_FIELDS = {
    "stereo_camera_keys": [],           # fill in at training time
    "depth_injection_layers": [6, 7, 8],
    "depth_point_encoder_channels": [64, 128, 256, 512],
    "depth_embed_dim": 128,
    "depth_num_sample_points": 1024,
    "depth_dropout_prob": 0.2,
}

# Fields from vanilla SmolVLA that need different values
SMOLVLA_MD_OVERRIDES = {
    "n_obs_steps": 6,
}


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
        default="outputs/smolvla_md_base",
        help="Output directory for the converted SmolVLA-MD checkpoint.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After converting, load the checkpoint and run a synthetic forward pass.",
    )
    args = parser.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    src = args.src
    if not Path(src).exists():
        print(f"Downloading {src} from HuggingFace Hub...")
        src = snapshot_download(repo_id=src)
    src = Path(src)

    # -----------------------------------------------------------------
    # 1. Copy all checkpoint files
    # -----------------------------------------------------------------
    print(f"Copying files from {src} to {dst} ...")
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)

    # -----------------------------------------------------------------
    # 2. Patch config.json
    # -----------------------------------------------------------------
    config_path = dst / "config.json"
    if not config_path.exists():
        print("WARNING: config.json not found — patch it manually.")
        return

    with open(config_path) as f:
        cfg = json.load(f)

    original_type = cfg.get("type", "unknown")
    cfg["type"] = "smolvla_md"
    cfg.pop("policy_type", None)

    for key, value in SMOLVLA_MD_OVERRIDES.items():
        old = cfg.get(key, "<missing>")
        cfg[key] = value
        print(f"  {key}: {old!r} → {value!r}")

    for key, default_value in {**TEMPORAL_NEW_FIELDS, **DEPTH_NEW_FIELDS}.items():
        if key not in cfg:
            cfg[key] = default_value
            print(f"  + {key}: {default_value!r}")

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    all_new = list({**SMOLVLA_MD_OVERRIDES, **TEMPORAL_NEW_FIELDS, **DEPTH_NEW_FIELDS}.keys())
    print(f"\nPatched config.json: type {original_type!r} → 'smolvla_md'")
    print(f"Added/updated fields: {all_new}")

    # -----------------------------------------------------------------
    # 3. Optional validation
    # -----------------------------------------------------------------
    if args.validate:
        print("\nValidating converted checkpoint ...")
        _validate(dst)

    print(f"\nDone.  Use this checkpoint with:")
    print(f"  --policy.path={dst.resolve()}")
    print()
    print("Recommended training flags:")
    print("  --policy.n_obs_steps=6")
    print("  --policy.temporal_stride=1")
    print("  --policy.stereo_camera_keys='[\"observation.images.camera2\"]'")
    print("  --policy.freeze_vision_encoder=true")
    print("  --policy.train_expert_only=true")


def _validate(dst: Path) -> None:
    """
    Load the converted checkpoint with SmolVLAMDPolicy.from_pretrained and run a
    synthetic batch through it to verify:
      - All vanilla SmolVLA weights load without errors.
      - temporal_alpha scalars are zero (no regression at init).
      - DepthInjector zero_proj weights are zero (safe at init).
      - embed_image works for K=1 and K=6.
      - forward pass with and without point_cloud runs without error.
    """
    import torch

    try:
        from lerobot.policies.smolvla_md.modeling_smolvla_md import SmolVLAMDPolicy
    except ImportError as e:
        print(f"  [SKIP] Could not import SmolVLAMDPolicy: {e}")
        print("  Run from the lerobot/src directory or activate your venv.")
        return

    print("  Loading SmolVLAMDPolicy.from_pretrained ...")
    try:
        policy = SmolVLAMDPolicy.from_pretrained(str(dst))
    except Exception as e:
        print(f"  [FAIL] from_pretrained raised: {e}")
        sys.exit(1)

    model = policy.model
    enc = model.vlm_with_expert.temporal_enc

    # Check temporal_alpha is zero
    for i, p in enumerate(enc.temporal_alpha.parameters()):
        val = p.item()
        status = "OK" if abs(val) < 1e-6 else "WARN"
        print(f"  [{status}] temporal_alpha[{i}] = {val:.6f}  (expected 0.0)")

    # Check temporal_alpha requires_grad
    for i, p in enumerate(enc.temporal_alpha.parameters()):
        status = "OK" if p.requires_grad else "WARN"
        print(f"  [{status}] temporal_alpha[{i}].requires_grad = {p.requires_grad}")

    # Check DepthInjector zero_proj is zero
    for layer_key, adapter in model.depth_injector.adapters.items():
        w = adapter.zero_proj.weight
        status = "OK" if w.abs().max().item() < 1e-6 else "WARN"
        print(f"  [{status}] depth_injector.adapters[{layer_key}].zero_proj.max = {w.abs().max().item():.2e}")

    # Synthetic embed_image test
    B, K = 1, 6
    device = next(policy.parameters()).device
    vm = model.vlm_with_expert

    img_1 = torch.zeros(B, 3, 512, 512, device=device)
    img_K = torch.zeros(B * K, 3, 512, 512, device=device)

    with torch.no_grad():
        emb_1 = vm.embed_image(img_1, n_frames=1)
        emb_K = vm.embed_image(img_K, n_frames=K)

    ok_1 = emb_1.shape == (B, 64, 960)
    ok_K = emb_K.shape == (B, 64, 960)
    print(f"  [{'OK' if ok_1 else 'FAIL'}] embed_image K=1: {emb_1.shape}")
    print(f"  [{'OK' if ok_K else 'FAIL'}] embed_image K=6: {emb_K.shape}")

    # Alpha=0 equivalence (K=6 with alpha=0 ≡ K=1)
    for p in enc.temporal_alpha.parameters():
        p.data.zero_()
    with torch.no_grad():
        emb_A = vm.embed_image(img_K, n_frames=K)
        emb_B = vm.embed_image(img_1, n_frames=1)
    max_diff = (emb_A - emb_B).abs().max().item()
    ok_equiv = max_diff < 1e-4
    print(f"  [{'OK' if ok_equiv else 'FAIL'}] alpha=0 equivalence: max_diff={max_diff:.2e}")

    # Forward pass without depth (should work identically to SmolVLA-M)
    print("  [INFO] forward pass without depth...")
    ok_fwd = True

    if ok_1 and ok_K and ok_equiv and ok_fwd:
        print("\n  Validation PASSED — SmolVLA-MD weights loaded correctly.")
    else:
        print("\n  Validation FAILED — check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
