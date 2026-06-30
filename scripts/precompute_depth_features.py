#!/usr/bin/env python3
"""
Precompute stereo depth point clouds for a LeRobot dataset.

For each frame in observation.images.camera2 (side-by-side H × 2W stereo):
  1. Split into left / right halves
  2. Rectify using stereo calibration matrices
  3. SGBM + WLS filter → disparity map
  4. cv2.reprojectImageTo3D (Q matrix) → XYZ point cloud
  5. Filter valid points; sample N=1024 points uniformly
  6. Store as 'observation.depth.point_cloud' (flat float32 list, len N*3)
     in the dataset parquet file and register in meta/info.json

Run on a COPY of the stereo dataset — do NOT run on the original.

Usage
-----
python lerobot/scripts/precompute_depth_features.py \\
    --dataset_root /home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_final_dataset_120_eps_depth \\
    --calib_path camera/stereo_calibration_result.npz \\
    --n_pts 1024 \\
    --num_workers 4
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import av
import cv2
import cv2.ximgproc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Shared frame counter — set by pool initializer so workers can update it
_frame_counter: mp.Value = None  # type: ignore

def _pool_init(counter):
    global _frame_counter
    _frame_counter = counter


# ---------------------------------------------------------------------------
# Stereo processor helpers
# ---------------------------------------------------------------------------

def _build_rectify_maps(calib: dict, eye_wh: tuple[int, int]):
    """Build cv2 undistort-rectify maps for left and right cameras."""
    w, h = eye_wh
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        calib["K_l"], calib["D_l"], calib["R1"], calib["P1"], (w, h), cv2.CV_32FC1
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        calib["K_r"], calib["D_r"], calib["R2"], calib["P2"], (w, h), cv2.CV_32FC1
    )
    return map1_l, map2_l, map1_r, map2_r


def _build_sgbm(num_disp: int = 64, block: int = 7):
    """Create a SGBM left matcher and matching right matcher for WLS."""
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block**2,
        P2=32 * 3 * block**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    return left_matcher, right_matcher


def _build_wls():
    """Create WLS filter (must be linked to the same left_matcher later)."""
    # Placeholder — rebuilt in worker alongside the SGBM because WLS must
    # hold a reference to the left_matcher used to compute left disparity.
    pass


def _process_frame(
    frame_bgr: np.ndarray,
    maps: tuple,
    Q: np.ndarray,
    left_matcher,
    right_matcher,
    wls_filter,
    n_pts: int,
    max_z_mm: float = 2000.0,
) -> np.ndarray:
    """
    Process one stereo frame (H, 2W, 3) → sampled point cloud (n_pts, 3) float32.

    Returns zeros if no valid points found.
    """
    map1_l, map2_l, map1_r, map2_r = maps
    h, w_full = frame_bgr.shape[:2]
    w = w_full // 2

    left_bgr = frame_bgr[:, :w]
    right_bgr = frame_bgr[:, w:]

    left_rect = cv2.remap(left_bgr, map1_l, map2_l, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_bgr, map1_r, map2_r, cv2.INTER_LINEAR)

    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    disp_l = left_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp_r = right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0
    disp = wls_filter.filter(disp_l, left_gray, disparity_map_right=disp_r)

    pts3d = cv2.reprojectImageTo3D(disp, Q)  # (H, W, 3)

    valid = (
        np.isfinite(pts3d).all(axis=2)
        & (disp > 0)
        & (pts3d[:, :, 2] > 0)
        & (pts3d[:, :, 2] < max_z_mm)
    )
    valid_pts = pts3d[valid]  # (M, 3)

    if len(valid_pts) == 0:
        return np.zeros((n_pts, 3), dtype=np.float32)
    if len(valid_pts) >= n_pts:
        idx = np.random.choice(len(valid_pts), n_pts, replace=False)
    else:
        idx = np.random.choice(len(valid_pts), n_pts, replace=True)
    return valid_pts[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Per-video-file worker
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> tuple[int, np.ndarray]:
    """
    Process all frames in one video file.

    Args:
        args: (file_idx, video_path, episodes_df_rows, calib_path, n_pts)

    Returns:
        (global_start_idx, clouds) where clouds has shape (total_frames, n_pts, 3)
    """
    file_idx, video_path, episodes_rows, calib_path, n_pts = args

    calib = np.load(calib_path)
    eye_w = int(calib["img_size"][0])
    eye_h = int(calib["img_size"][1])
    Q = calib["Q"]

    maps = _build_rectify_maps(calib, (eye_w, eye_h))
    left_matcher, right_matcher = _build_sgbm()
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)

    # Sort episodes by from_timestamp so they match video frame order
    episodes = sorted(episodes_rows, key=lambda r: r["from_ts"])
    total_frames = sum(r["length"] for r in episodes)
    global_start = episodes[0]["global_from"]

    clouds = np.zeros((total_frames, n_pts, 3), dtype=np.float32)

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    local_idx = 0
    ep_cursor = 0
    ep_remaining = episodes[0]["length"]

    for frame in container.decode(stream):
        if local_idx >= total_frames:
            break
        img_bgr = frame.to_ndarray(format="bgr24")
        clouds[local_idx] = _process_frame(
            img_bgr, maps, Q, left_matcher, right_matcher, wls_filter, n_pts
        )
        local_idx += 1
        if _frame_counter is not None:
            with _frame_counter.get_lock():
                _frame_counter.value += 1
        ep_remaining -= 1
        if ep_remaining == 0:
            ep_cursor += 1
            if ep_cursor < len(episodes):
                ep_remaining = episodes[ep_cursor]["length"]

    container.close()

    if local_idx != total_frames:
        print(
            f"[WARNING] file-{file_idx:03d}: expected {total_frames} frames, "
            f"got {local_idx}. Padding with zeros."
        )

    return global_start, clouds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Root of the LeRobot dataset to modify IN-PLACE (use a depth copy!)",
    )
    parser.add_argument(
        "--calib_path",
        required=True,
        help="Path to stereo_calibration_result.npz",
    )
    parser.add_argument("--n_pts", type=int, default=1024)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of parallel video-file workers",
    )
    parser.add_argument(
        "--camera_key",
        default="observation.images.camera2",
        help="Dataset key of the side-by-side stereo camera",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    calib_path = str(Path(args.calib_path).resolve())
    n_pts = args.n_pts
    cam_key = args.camera_key

    # ------------------------------------------------------------------
    # Load episode metadata
    # ------------------------------------------------------------------
    eps_parquet = root / "meta" / "episodes" / "chunk-000"
    eps_df = pd.read_parquet(eps_parquet)

    file_col = f"videos/{cam_key}/file_index"
    ts_from_col = f"videos/{cam_key}/from_timestamp"
    ts_to_col = f"videos/{cam_key}/to_timestamp"

    video_dir = root / "videos" / cam_key / "chunk-000"
    num_files = int(eps_df[file_col].max()) + 1

    print(f"Dataset: {root.name}")
    print(f"Total episodes: {len(eps_df)}, video files: {num_files}, total frames: {eps_df['length'].sum()}")
    total_frames = int(eps_df["length"].sum())

    # ------------------------------------------------------------------
    # Build worker task list (one per video file)
    # ------------------------------------------------------------------
    tasks = []
    for fi in range(num_files):
        mask = eps_df[file_col] == fi
        file_eps = eps_df[mask]
        episodes_rows = [
            {
                "length": int(row["length"]),
                "global_from": int(row["dataset_from_index"]),
                "from_ts": float(row[ts_from_col]),
                "to_ts": float(row[ts_to_col]),
            }
            for _, row in file_eps.iterrows()
        ]
        video_path = video_dir / f"file-{fi:03d}.mp4"
        tasks.append((fi, video_path, episodes_rows, calib_path, n_pts))

    # ------------------------------------------------------------------
    # Run workers with real-time frame progress
    # ------------------------------------------------------------------
    n_workers = min(args.num_workers, num_files)
    print(f"\nProcessing {num_files} video files with {n_workers} workers …")
    t0 = time.time()

    all_clouds = np.zeros((total_frames, n_pts, 3), dtype=np.float32)
    frame_counter = mp.Value("i", 0)

    with mp.Pool(
        processes=n_workers,
        initializer=_pool_init,
        initargs=(frame_counter,),
    ) as pool:
        async_results = [pool.apply_async(_worker, (task,)) for task in tasks]

        # Poll counter every 0.5 s and update tqdm in main process
        completed = []
        with tqdm(total=total_frames, desc="frames", unit="fr", dynamic_ncols=True) as pbar:
            while len(completed) < len(async_results):
                time.sleep(0.5)

                # Update frame progress bar
                current = frame_counter.value
                pbar.n = current
                pbar.set_postfix(
                    files=f"{len(completed)}/{num_files}",
                    fps=f"{current / max(time.time() - t0, 1):.1f}",
                )
                pbar.refresh()

                # Collect any finished workers
                for r in async_results:
                    if r not in completed and r.ready():
                        global_start, clouds = r.get()
                        n = clouds.shape[0]
                        all_clouds[global_start : global_start + n] = clouds
                        completed.append(r)

    elapsed = time.time() - t0
    fps_achieved = total_frames / elapsed
    print(f"\nDone in {elapsed/3600:.2f}h  ({fps_achieved:.1f} frames/s)")

    # ------------------------------------------------------------------
    # Write point clouds into the parquet file
    # ------------------------------------------------------------------
    parquet_path = root / "data" / "chunk-000" / "file-000.parquet"
    print(f"\nUpdating parquet: {parquet_path} …")
    table = pq.read_table(parquet_path)

    # Flatten (N, 3) → list of 3*N floats per row, ordered by global index
    flat = all_clouds.reshape(total_frames, n_pts * 3).tolist()
    flat_col = pa.array(flat, type=pa.list_(pa.float32()))

    feature_name = "observation.depth.point_cloud"
    if feature_name in table.schema.names:
        col_idx = table.schema.names.index(feature_name)
        table = table.set_column(col_idx, feature_name, flat_col)
    else:
        table = table.append_column(
            pa.field(feature_name, pa.list_(pa.float32())), flat_col
        )

    pq.write_table(table, parquet_path, compression="snappy")
    print(f"Parquet written ({parquet_path.stat().st_size / 1e9:.2f} GB)")

    # ------------------------------------------------------------------
    # Update meta/info.json
    # ------------------------------------------------------------------
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["features"][feature_name] = {
        "dtype": "float32",
        "shape": [n_pts, 3],
        "names": None,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print("info.json updated.")

    print("\nAll done. The depth copy now contains 'observation.depth.point_cloud'.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
