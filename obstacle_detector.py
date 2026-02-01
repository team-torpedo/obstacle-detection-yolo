import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from typing import List, Dict, Any, Optional


class ObstacleDetector:
    def __init__(
        self,
        model_path: str = 'yolov11m.pt',
        confidence_threshold: float = 0.5,
        max_history_length: int = 30,
        focal_length: float = 500,
        real_height: float = 1.7,
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracking_history = defaultdict(list)
        self.max_history_length = max_history_length
        self.focal_length = focal_length
        self.real_height = real_height

    def detect_obstacles(self, frame: np.ndarray):
        results = self.model(frame, conf=self.confidence_threshold)
        return results[0]

    def track_obstacles(self, frame: np.ndarray):
        results = self.model.track(frame, persist=True, conf=self.confidence_threshold)
        return results[0]

    def calculate_distance(self, bbox: List[float], frame_height: int) -> float:
        box_height = bbox[3] - bbox[1]
        distance = (self.real_height * self.focal_length) / box_height if box_height > 0 else 0
        return distance

    def draw_detections(self, frame: np.ndarray, results) -> np.ndarray:
        annotated_frame = frame.copy()

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                distance = self.calculate_distance([x1, y1, x2, y2], frame.shape[0])

                color = (0, 255, 0) if distance > 5 else (0, 165, 255) if distance > 2 else (0, 0, 255)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                label = f'{class_name} {confidence:.2f} | {distance:.1f}m'
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    self.tracking_history[track_id].append((center_x, center_y))

                    if len(self.tracking_history[track_id]) > self.max_history_length:
                        self.tracking_history[track_id].pop(0)

                    points = np.array(self.tracking_history[track_id], dtype=np.int32)
                    if points.shape[0] >= 2:
                        cv2.polylines(annotated_frame, [points], False, color, 2)

        return annotated_frame

    def get_obstacle_info(self, results) -> List[Dict[str, Any]]:
        obstacles: List[Dict[str, Any]] = []

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                distance = self.calculate_distance([x1, y1, x2, y2], results.orig_shape[0])

                obstacle_data: Dict[str, Any] = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'distance': distance,
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                }

                if hasattr(box, 'id') and box.id is not None:
                    obstacle_data['track_id'] = int(box.id[0])

                obstacles.append(obstacle_data)

        return obstacles
