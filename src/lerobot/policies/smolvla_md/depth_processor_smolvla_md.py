"""
Stereo depth processing pipeline for SmolVLA-D.

Mirrors the software-downsampling stereo pipeline from
``camera/calibracion_camara.ipynb`` (Section 6, "Modo Software").

Pipeline overview (matches the notebook exactly):
  1. Receive a full 2560×800 BGR side-by-side frame from the SVPRO stereo camera.
  2. Split at the horizontal midpoint → left (1280×800) and right (1280×800).
  3. Resize each half with INTER_AREA to 640×400  (scale = 0.5).
  4. Apply pre-computed stereo-rectification maps (cv2.remap).
  5. Convert to greyscale and compute StereoSGBM + WLS disparity.
  6. Reproject disparity to 3-D with cv2.reprojectImageTo3D (uses the Q matrix
     from the calibration .npz file).
  7. Return the rectified left image (RGB) and a (N, 3) float32 array of valid
     3-D points in mm.

At inference, the heavy SGBM step runs in a background thread
(``StereoDepthWorker``) and caches its result; the model reads from the cache
at each action step, so the depth computation is effectively free per-step.

Usage at inference::

    policy.setup_stereo_depth("camera/stereo_calibration_result.npz")

    # in the inference loop, each time a new frame arrives:
    frame_bgr = ...  # 2560×800 numpy uint8 BGR
    policy.update_stereo_frame(frame_bgr)

    # select_action() reads the cached point cloud automatically
    action = policy.select_action(batch)

For training, pre-compute depth features offline (e.g. with a dataset
post-processing script) and store them as ``batch["depth_point_cloud"]``
(shape: ``(B, K, 3)``).  The model handles the ``None`` case gracefully by
falling back to vanilla flow-matching behaviour (injection = 0 at init).
"""

import threading
from typing import Optional

import cv2
import numpy as np

try:
    import cv2.ximgproc  # noqa: F401 — required for WLS filter (opencv-contrib)
    _XIMGPROC_AVAILABLE = True
except ImportError:
    _XIMGPROC_AVAILABLE = False


class StereoDepthProcessor:
    """
    CPU stereo depth pipeline for the SVPRO side-by-side camera.

    Parameters mirror ``calibracion_camara.ipynb`` Section 6:
      - target_width / target_height: size per eye after software downscale (640×400).
      - num_disparities: must be divisible by 16.  Default 48 works well at 640×400.
      - block_size: StereoSGBM block size (odd, default 5).
      - wls_lambda: WLS smoothness weight (default 8 000).
      - wls_sigma: WLS edge sensitivity (default 1.5).
    """

    TARGET_W = 640
    TARGET_H = 400

    def __init__(
        self,
        calib_path: str,
        target_width: int = 640,
        target_height: int = 400,
        num_disparities: int = 48,
        block_size: int = 5,
        wls_lambda: float = 8_000.0,
        wls_sigma: float = 1.5,
    ):
        if not _XIMGPROC_AVAILABLE:
            raise ImportError(
                "opencv-contrib-python is required for WLS filtering. "
                "Install it with:  pip install opencv-contrib-python"
            )

        self.target_w = target_width
        self.target_h = target_height
        calib = np.load(calib_path)

        # Scale intrinsic matrices to the downsampled resolution
        scale_x = target_width / (calib["img_size"][0] / 2)  # img_size[0] is full stereo width
        scale_y = target_height / calib["img_size"][1]

        def _scale_K(K, sx, sy):
            K = K.copy()
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy
            return K

        K_l = _scale_K(calib["K_l"], scale_x, scale_y)
        K_r = _scale_K(calib["K_r"], scale_x, scale_y)
        D_l, D_r = calib["D_l"], calib["D_r"]
        R1, R2 = calib["R1"], calib["R2"]

        # Scale projection matrices
        P1 = calib["P1"].copy()
        P2 = calib["P2"].copy()
        P1[0, :] *= scale_x
        P1[1, :] *= scale_y
        P2[0, :] *= scale_x
        P2[1, :] *= scale_y

        sz = (target_width, target_height)
        self.map_lx, self.map_ly = cv2.initUndistortRectifyMap(
            K_l, D_l, R1, P1, sz, cv2.CV_32FC1
        )
        self.map_rx, self.map_ry = cv2.initUndistortRectifyMap(
            K_r, D_r, R2, P2, sz, cv2.CV_32FC1
        )

        # Q matrix also needs to be scaled (translation components in pixels)
        Q = calib["Q"].copy()
        Q[0, 3] *= scale_x   # -cx
        Q[1, 3] *= scale_y   # -cy
        Q[2, 3] *= scale_x   # focal length (approximately, along x)
        # Q[3,2] is 1/Tx (baseline); Q[3,3] is (cx_r - cx_l)/Tx — scale-invariant
        # Only the pixel-coordinate offsets need scaling:
        self.Q = Q.astype(np.float32)

        # StereoSGBM + WLS — same parameters as the notebook
        num_ch = 3  # we pass colour images; P1/P2 are per-channel
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * num_ch * block_size ** 2,
            P2=32 * num_ch * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        self.wls_filter.setLambda(wls_lambda)
        self.wls_filter.setSigmaColor(wls_sigma)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self, stereo_frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process one 2560×800 side-by-side stereo frame.

        Args:
            stereo_frame_bgr: uint8 BGR array of shape (800, 2560, 3).

        Returns:
            left_rect_bgr: Rectified left image (target_h, target_w, 3) uint8.
            point_cloud:   (N, 3) float32 array of valid XYZ points in mm.
        """
        mid = stereo_frame_bgr.shape[1] // 2
        left_raw = stereo_frame_bgr[:, :mid]
        right_raw = stereo_frame_bgr[:, mid:]

        sz = (self.target_w, self.target_h)
        left_small = cv2.resize(left_raw, sz, interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_raw, sz, interpolation=cv2.INTER_AREA)

        left_rect = cv2.remap(left_small, self.map_lx, self.map_ly, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_small, self.map_rx, self.map_ry, cv2.INTER_LINEAR)

        disp = self._compute_disparity(left_rect, right_rect)
        points_3d = cv2.reprojectImageTo3D(disp, self.Q)

        valid = (disp > 0) & np.isfinite(points_3d).all(axis=2)
        point_cloud = points_3d[valid].astype(np.float32)  # (N, 3)

        return left_rect, point_cloud

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_disparity(
        self, left_rect: np.ndarray, right_rect: np.ndarray
    ) -> np.ndarray:
        """Returns float32 disparity map (pixel units) with WLS hole-filling."""
        gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        disp_l = self.left_matcher.compute(gray_l, gray_r)
        disp_r = self.right_matcher.compute(gray_r, gray_l)

        # WLS filter: uses left-right consistency + guided filter for hole-filling
        disp_filtered = self.wls_filter.filter(disp_l, gray_l, None, disp_r)

        return (disp_filtered / 16.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class StereoDepthWorker:
    """
    Runs ``StereoDepthProcessor.process_frame`` in a background thread and
    caches the latest result so that the policy can read from it without
    blocking the inference loop.

    Usage::

        worker = StereoDepthWorker(StereoDepthProcessor("calibration.npz"))

        # Each time a new stereo frame arrives (e.g. every 1–2 s):
        worker.submit(stereo_bgr_frame)

        # Inside the inference loop (fast, ~µs):
        pts = worker.get_latest_points()  # (N, 3) float32 or None
    """

    def __init__(self, processor: StereoDepthProcessor):
        self._processor = processor
        self._latest_points: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._active_thread: Optional[threading.Thread] = None

    def submit(self, stereo_frame_bgr: np.ndarray) -> None:
        """
        Start computing depth from *stereo_frame_bgr* in a background thread.
        If a computation is already running, the new frame is silently dropped
        (the caller should submit again on the next opportunity).
        """
        if self._active_thread is not None and self._active_thread.is_alive():
            return
        self._active_thread = threading.Thread(
            target=self._run,
            args=(stereo_frame_bgr.copy(),),  # copy to avoid data races with the camera
            daemon=True,
        )
        self._active_thread.start()

    def get_latest_points(self) -> Optional[np.ndarray]:
        """Return the most recently computed (N, 3) float32 point cloud, or None."""
        with self._lock:
            return self._latest_points

    # ------------------------------------------------------------------

    def _run(self, frame: np.ndarray) -> None:
        _, points = self._processor.process_frame(frame)
        with self._lock:
            self._latest_points = points
