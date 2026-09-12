"""
Lightweight YOLOv8n detector using ONNX Runtime — no PyTorch needed at runtime.

The ONNX model is exported once during Docker build and loaded here with
onnxruntime, which uses ~50 MB RAM vs ~300 MB for torch.
"""
import cv2
import numpy as np
import onnxruntime as ort
import os


class HumanDetector:
    """YOLOv8n object detector using ONNX Runtime inference."""

    # COCO class IDs we care about
    TARGET_CLASSES = [0, 2, 3, 5, 7]
    #  0 = person,  2 = car,  3 = motorcycle,  5 = bus,  7 = truck

    def __init__(self, model_path=None, conf_threshold=0.25):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "yolov8n.onnx")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Limit to 2 threads to reduce memory pressure
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 2

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Read input shape from the model  →  (1, 3, H, W)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        _, _, self.input_h, self.input_w = input_meta.shape

        self.conf_threshold = conf_threshold
        self.iou_threshold = 0.45

        print(
            f"[OK] YOLOv8n ONNX loaded  "
            f"({self.input_w}×{self.input_h}, "
            f"conf≥{self.conf_threshold})"
        )

    # ── Pre-process: letterbox + normalize ───────────────────────────────────
    def _preprocess(self, frame):
        """Resize with letterbox padding to (input_h, input_w) and normalize."""
        h, w = frame.shape[:2]

        # Scale to fit inside input size
        scale = min(self.input_w / w, self.input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # Create padded canvas
        canvas = np.full(
            (self.input_h, self.input_w, 3), 114, dtype=np.uint8
        )
        dx = (self.input_w - new_w) // 2
        dy = (self.input_h - new_h) // 2
        canvas[dy : dy + new_h, dx : dx + new_w] = resized

        # HWC→CHW, BGR→RGB, uint8→float32 [0,1]
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[np.newaxis, ...], scale, dx, dy

    # ── Post-process: parse output + NMS ─────────────────────────────────────
    def _postprocess(self, output, scale, dx, dy):
        """
        YOLOv8 output shape: (1, 84, N)
        84 = 4 (cx, cy, w, h) + 80 (class scores)
        """
        predictions = output[0].T  # → (N, 84)

        # Split box coords and class scores
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        # Best class per detection
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence and target classes
        mask = confidences >= self.conf_threshold
        target_mask = np.isin(class_ids, self.TARGET_CLASSES)
        mask = mask & target_mask

        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert xywh → xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Remove letterbox padding and rescale to original frame coords
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - dx) / scale
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - dy) / scale
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - dx) / scale
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - dy) / scale

        # OpenCV NMS
        boxes_for_nms = []
        for b in boxes_xyxy:
            boxes_for_nms.append([float(b[0]), float(b[1]),
                                  float(b[2] - b[0]), float(b[3] - b[1])])

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )

        detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                x1, y1, x2, y2 = boxes_xyxy[i]
                detections.append((
                    int(x1), int(y1), int(x2), int(y2),
                    float(confidences[i]),
                    int(class_ids[i]),
                ))

        return detections

    # ── Public API (same signature as the old ultralytics-based detector) ────
    def detect(self, frame):
        """Run detection on a BGR frame. Returns list of (x1,y1,x2,y2,conf,cls_id)."""
        blob, scale, dx, dy = self._preprocess(frame)
        output = self.session.run(None, {self.input_name: blob})[0]
        return self._postprocess(output, scale, dx, dy)