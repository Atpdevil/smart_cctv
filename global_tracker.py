import numpy as np
import threading
from reid_model import ReIDFeatureExtractor


class GlobalTracker:
    """
    Cross-camera person re-identification using Fast ReID (OSNet).
    
    Uses deep appearance embeddings (512-dim) from a pre-trained ReID model
    to match the same person across different camera feeds. Far more accurate
    than color histograms for distinguishing individuals.
    """

    def __init__(self, match_threshold=0.60, feature_update_alpha=0.2):
        """
        Args:
            match_threshold: Minimum cosine similarity to consider a match (0-1).
                             Higher = stricter matching (fewer false positives).
                             0.60 is a good default for OSNet embeddings.
            feature_update_alpha: EMA blending factor for updating stored features.
        """
        self._lock = threading.Lock()
        self._next_global_id = 1
        self.match_threshold = match_threshold
        self.feature_update_alpha = feature_update_alpha

        # (cam_id, local_id) -> global_id
        self._local_to_global = {}

        # global_id -> { "feature": np.array(512,), "cam_id": str }
        self._global_profiles = {}

        # Initialize the Fast ReID feature extractor
        self._reid = ReIDFeatureExtractor(model_name='osnet_x0_25')

    def _cosine_similarity(self, feat_a, feat_b):
        """Compute cosine similarity between two L2-normalized feature vectors."""
        if feat_a is None or feat_b is None:
            return -1.0
        return float(np.dot(feat_a, feat_b))

    def resolve(self, cam_id, local_track_id, crop, cls_id=0):
        """
        Map a (cam_id, local_track_id) to a persistent global ID.
        
        Args:
            cam_id: Camera identifier string
            local_track_id: ByteTrack's local track ID
            crop: BGR image crop of the detected object
            cls_id: Class ID (only persons cls_id=0 get cross-camera matching)
            
        Returns:
            global_id: Integer global ID that persists across cameras
        """
        with self._lock:
            key = (cam_id, local_track_id)

            # Already mapped — update features and return
            if key in self._local_to_global:
                gid = self._local_to_global[key]
                self._update_profile(gid, crop, cam_id)
                return gid

            # New local track — extract ReID features and try to match
            new_feature = self._reid.extract(crop) if cls_id == 0 else None

            best_gid = None
            best_score = -1.0

            # Only attempt cross-camera matching for persons
            if cls_id == 0 and new_feature is not None:
                for gid, profile in self._global_profiles.items():
                    # Match against profiles from OTHER cameras
                    if profile["cam_id"] != cam_id:
                        score = self._cosine_similarity(new_feature, profile["feature"])
                        if score > best_score:
                            best_score = score
                            best_gid = gid

            if best_gid is not None and best_score >= self.match_threshold:
                # Matched — reuse global ID
                self._local_to_global[key] = best_gid
                # Update the profile with blended features
                self._blend_feature(best_gid, new_feature, cam_id)
                return best_gid
            else:
                # No match — assign new global ID
                gid = self._next_global_id
                self._next_global_id += 1
                self._local_to_global[key] = gid
                self._global_profiles[gid] = {
                    "feature": new_feature,
                    "cam_id": cam_id,
                }
                return gid

    def _update_profile(self, gid, crop, cam_id):
        """Periodically update stored features with exponential moving average."""
        if gid not in self._global_profiles:
            return
        # Only update features every few calls to reduce computation
        profile = self._global_profiles[gid]
        profile["cam_id"] = cam_id

    def _blend_feature(self, gid, new_feature, cam_id):
        """Blend new feature with stored feature using EMA."""
        if gid not in self._global_profiles or new_feature is None:
            return
        profile = self._global_profiles[gid]
        old_feature = profile["feature"]
        if old_feature is not None:
            alpha = self.feature_update_alpha
            blended = (1 - alpha) * old_feature + alpha * new_feature
            # Re-normalize
            norm = np.linalg.norm(blended)
            if norm > 0:
                blended = blended / norm
            profile["feature"] = blended
        else:
            profile["feature"] = new_feature
        profile["cam_id"] = cam_id

    def cleanup_stale(self, active_keys):
        """Remove local mappings that are no longer actively tracked."""
        with self._lock:
            stale = [k for k in self._local_to_global if k not in active_keys]
            for k in stale:
                del self._local_to_global[k]
