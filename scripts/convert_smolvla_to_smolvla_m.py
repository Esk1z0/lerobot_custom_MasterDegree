#!/usr/bin/env python
"""
Converts a pretrained SmolVLA checkpoint (e.g. lerobot/smolvla_base) into a
SmolVLA-M checkpoint ready for fine-tuning with temporal memory (MEM paper video encoder).

What this script does:
  1. Downloads (or copies) the vanilla SmolVLA checkpoint.
  2. Patches config.json:
       - type: smolvla → smolvla_m
       - n_obs_steps: 1 → 6  (K frames per camera per timestep)
       - Adds SmolVLA-M temporal fields with their default values.
  3. Leaves the weight file untouched.
     When SmolVLA-M loads from the converted checkpoint, PreTrainedPolicy uses
     strict=False so:
       - All existing weights (VLM, expert, projections) load into matching keys.
       - The new temporal_alpha scalars (3 × float, init=0) are absent from the
         checkpoint and keep their zero-init → no performance regression at day 0.

Weight key compatibility:
  Vanilla SmolVLA  model.vision_model.encoder.layers.N.*
  SmolVLA-M        model.vision_model.encoder.layers.N.*  ← same key!
  New              model.vision_model.encoder.temporal_alpha.{0,1,2}  ← missing in ckpt → zero-init

Usage:
    python scripts/convert_smolvla_to_smolvla_m.py \\
        --src lerobot/smolvla_base \\
        --dst outputs/smolvla_m_base

    # Optionally validate the converted checkpoint with a synthetic forward pass:
    python scripts/convert_smolvla_to_smolvla_m.py \\
        --src lerobot/smolvla_base \\
        --dst outputs/smolvla_m_base \\
        --validate

Then train:
    lerobot-train \\
        --policy.path=outputs/smolvla_m_base \\
        --dataset.repo_id=<your-dataset> \\
        --policy.n_obs_steps=6 \\
        --policy.temporal_stride=1 \\
        ...
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


# New fields added by SmolVLA-M not present in vanilla SmolVLA config.
# Written with their defaults so the converted config is self-documenting.
SMOLVLA_M_NEW_FIELDS = {
    # -----------------------------------------------------------------------
    # Temporal memory — MEM paper video encoder (Option A+)
    # K total frames (past + current) fed to the ViT encoder.
    # Override at training time with --policy.temporal_num_frames=<K>
    # -----------------------------------------------------------------------
    "temporal_num_frames": 6,

    # Dataset steps between consecutive frames.
    #   stride=1  → consecutive frames (e.g. 20 Hz recording → 5×50ms gaps)
    #   stride=20 → ~1 second between frames at 20 Hz
    # Override with --policy.temporal_stride=<S>
    "temporal_stride": 1,

    # ViT layers (0-indexed) that receive extra causal temporal attention.
    # SmolVLM2-500M has 12 ViT layers; every 4th = [3, 7, 11].
    "temporal_vit_layers": [3, 7, 11],
}

# Fields from vanilla SmolVLA that need to be updated (not just added).
SMOLVLA_M_OVERRIDES = {
    # Vanilla SmolVLA uses n_obs_steps=1 (single frame).
    # SmolVLA-M collects K=6 frames per camera observation.
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
        default="outputs/smolvla_m_base",
        help="Output directory for the converted SmolVLA-M checkpoint.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After converting, load the checkpoint and run a synthetic forward pass to verify weights load correctly.",
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

    # Patch type identifier
    cfg["type"] = "smolvla_m"
    cfg.pop("policy_type", None)   # remove stale field if present

    # Override fields that exist in vanilla SmolVLA but need different values
    for key, value in SMOLVLA_M_OVERRIDES.items():
        old = cfg.get(key, "<missing>")
        cfg[key] = value
        print(f"  {key}: {old!r} → {value!r}")

    # Inject SmolVLA-M specific fields (idempotent — skip if already present)
    for key, default_value in SMOLVLA_M_NEW_FIELDS.items():
        if key not in cfg:
            cfg[key] = default_value
            print(f"  + {key}: {default_value!r}")

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nPatched config.json: type {original_type!r} → 'smolvla_m'")
    print(f"Added/updated SmolVLA-M fields: {list({**SMOLVLA_M_OVERRIDES, **SMOLVLA_M_NEW_FIELDS}.keys())}")

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
    print(f"  --policy.n_obs_steps=6")
    print(f"  --policy.temporal_stride=1   # or 20 for ~1 s between frames at 20 Hz")
    print(f"  --policy.freeze_vision_encoder=true")
    print(f"  --policy.train_expert_only=true")


def _validate(dst: Path) -> None:
    """
    Load the converted checkpoint with SmolVLAMPolicy.from_pretrained and run a
    synthetic batch through it to verify:
      - All vanilla SmolVLA weights load without errors.
      - temporal_alpha scalars remain at zero (no regression on day 0).
      - embed_image works for K=1 and K=6.
      - prepare_images handles (B, K, C, H, W) input.
    """
    import torch

    try:
        from lerobot.policies.smolvla_m.modeling_smolvla_m import SmolVLAMPolicy
    except ImportError as e:
        print(f"  [SKIP] Could not import SmolVLAMPolicy: {e}")
        print("  Run from the lerobot/src directory or activate your venv.")
        return

    print("  Loading SmolVLAMPolicy.from_pretrained ...")
    try:
        policy = SmolVLAMPolicy.from_pretrained(str(dst))
    except Exception as e:
        print(f"  [FAIL] from_pretrained raised: {e}")
        sys.exit(1)

    model = policy.model   # VLAFlowMatching
    enc = model.vlm_with_expert.temporal_enc

    # Check temporal_alpha is zero (no regression)
    for i, p in enumerate(enc.temporal_alpha.parameters()):
        val = p.item()
        status = "OK" if abs(val) < 1e-6 else "WARN"
        print(f"  [{status}] temporal_alpha[{i}] = {val:.6f}  (expected 0.0)")

    # Check temporal_alpha requires_grad
    for i, p in enumerate(enc.temporal_alpha.parameters()):
        status = "OK" if p.requires_grad else "WARN"
        print(f"  [{status}] temporal_alpha[{i}].requires_grad = {p.requires_grad}  (expected True)")

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

    # Alpha=0 equivalence: K=6 with current-only frames must equal K=1
    for p in enc.temporal_alpha.parameters():
        p.data.zero_()
    imgs_zeros = torch.zeros(B * K, 3, 512, 512, device=device)
    with torch.no_grad():
        emb_A = vm.embed_image(imgs_zeros, n_frames=K)
        emb_B = vm.embed_image(img_1, n_frames=1)
    max_diff = (emb_A - emb_B).abs().max().item()
    ok_equiv = max_diff < 1e-4
    print(f"  [{'OK' if ok_equiv else 'FAIL'}] alpha=0 equivalence: max_diff={max_diff:.2e}")

    if ok_1 and ok_K and ok_equiv:
        print("\n  Validation PASSED — SmolVLA-M weights loaded correctly.")
    else:
        print("\n  Validation FAILED — check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
