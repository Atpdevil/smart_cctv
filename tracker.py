import numpy as np


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


class PersonTracker:
    """
    Lightweight IoU tracker — drop-in replacement for supervision.ByteTrack.
    Keeps the same update(frame, detections) -> [(x1,y1,x2,y2,track_id,cls_id)]
    interface, with greedy IoU matching, max_age coasting and min_hits confirm.
    """

    def __init__(self, iou_threshold=0.3, max_age=25, min_hits=3):
        self.iou_threshold = iou_threshold   # min IoU to associate det ↔ track
        self.max_age = max_age               # frames a track survives with no match
        self.min_hits = min_hits             # matches before a track is reported
        self._tracks = []                    # dict(id, box, cls, misses, hits)
        self._next_id = 1

    def update(self, frame, detections):
        if len(detections) == 0:
            dets = []
        else:
            # Highest-confidence detections match first
            dets = sorted(detections, key=lambda d: d[4], reverse=True)

        # Age all tracks
        for t in self._tracks:
            t["misses"] += 1

        results = []
        used_dets = set()

        # Match existing tracks to detections (greedy IoU)
        for t in self._tracks:
            best_iou, best_j = 0.0, -1
            for j, d in enumerate(dets):
                if j in used_dets:
                    continue
                v = _iou(t["box"], d[:4])
                if v > best_iou:
                    best_iou, best_j = v, j

            if best_j >= 0 and best_iou >= self.iou_threshold:
                d = dets[best_j]
                used_dets.add(best_j)
                t["box"] = (float(d[0]), float(d[1]), float(d[2]), float(d[3]))
                t["cls"] = int(d[5])
                t["misses"] = 0
                t["hits"] += 1
            elif t["misses"] > self.max_age:
                t["dead"] = True

        # Spawn new tracks for unmatched detections
        for j, d in enumerate(dets):
            if j in used_dets:
                continue
            self._tracks.append({
                "id": self._next_id,
                "box": (float(d[0]), float(d[1]), float(d[2]), float(d[3])),
                "cls": int(d[5]),
                "misses": 0,
                "hits": 1,
            })
            self._next_id += 1

        # Report confirmed tracks
        for t in self._tracks:
            if t["hits"] >= self.min_hits:
                results.append(self._format(t))

        # Drop dead tracks
        self._tracks = [t for t in self._tracks if not t.get("dead")]

        return results

    @staticmethod
    def _format(t):
        x1, y1, x2, y2 = t["box"]
        return (int(x1), int(y1), int(x2), int(y2), t["id"], t["cls"])
