import cv2
import numpy as np


class ReIDFeatureExtractor:
    """
    Lightweight ReID feature extractor based on spatial color histograms.

    Instead of a neural network, this builds a descriptor from HSV color
    histograms computed on the top / middle / bottom thirds of a person
    crop.  Each region gets a hue histogram and a saturation histogram,
    capturing clothing-color information with spatial layout awareness
    (e.g. shirt vs. trousers).

    The final feature vector is the L2-normalized concatenation of all
    six histograms (3 regions × 2 channels).

    This uses only OpenCV and NumPy — no PyTorch, no torchreid — so it
    fits within tight memory budgets (< 512 MB).
    """

    # Histogram parameters
    HUE_BINS = 36          # 0-180 → 5° per bin
    SAT_BINS = 32          # 0-256 → 8 levels per bin
    REGIONS  = 3           # top / middle / bottom

    # Total feature dimensionality: 3 × (36 + 32) = 204
    FEATURE_DIM = REGIONS * (HUE_BINS + SAT_BINS)

    def __init__(self, model_name=None, device=None):
        """
        Parameters are accepted for API-compatibility with the previous
        torchreid-based extractor but are silently ignored.
        """
        print("[OK] Lightweight ReID loaded (color histogram)")

    # ── Single-crop extraction ──────────────────────────────────────────────
    def extract(self, crop_bgr):
        """Extract an L2-normalized feature vector from a BGR person crop.

        Returns
        -------
        np.ndarray of shape (FEATURE_DIM,) or None if the crop is invalid.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        h, w = crop_bgr.shape[:2]
        if h < 20 or w < 10:
            return None

        try:
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        except Exception:
            return None

        # Split into top / middle / bottom thirds
        third = max(h // self.REGIONS, 1)
        parts = [
            hsv[0          : third,     :],   # top
            hsv[third      : 2 * third, :],   # middle
            hsv[2 * third  : h,         :],   # bottom
        ]

        histograms = []
        for part in parts:
            if part.size == 0:
                histograms.append(np.zeros(self.HUE_BINS + self.SAT_BINS,
                                           dtype=np.float32))
                continue

            # Hue histogram (channel 0, range 0-180)
            h_hist = cv2.calcHist([part], [0], None,
                                  [self.HUE_BINS], [0, 180])
            cv2.normalize(h_hist, h_hist)

            # Saturation histogram (channel 1, range 0-256)
            s_hist = cv2.calcHist([part], [1], None,
                                  [self.SAT_BINS], [0, 256])
            cv2.normalize(s_hist, s_hist)

            histograms.append(
                np.concatenate([h_hist.flatten(), s_hist.flatten()])
            )

        feature = np.concatenate(histograms).astype(np.float32)

        # L2-normalize the full vector
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm

        return feature

    # ── Batch extraction ────────────────────────────────────────────────────
    def extract_batch(self, crops_bgr):
        """Extract features for multiple crops.

        Returns
        -------
        list[np.ndarray | None] — one entry per input crop.
        """
        return [self.extract(crop) for crop in crops_bgr]
