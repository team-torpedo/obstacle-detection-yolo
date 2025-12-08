import cv2
import time
from typing import Any


class VideoObstacleDetector:
    def __init__(
        self,
        detector: Any,
        source=0,
        show_display: bool = True,
        save_output: bool = False,
        output_path: str = 'output.avi',
    ):
        self.detector = detector
        self.source = source
        self.capture = None
        self.show_display = show_display
        self.save_output = save_output
        self.output_path = output_path

    def start_detection(self):
        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video source: {self.source}")

        frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.capture.get(cv2.CAP_PROP_FPS)) or 30

        video_writer = None
        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break

                start_time = time.time()

                results = self.detector.track_obstacles(frame)
                annotated_frame = self.detector.draw_detections(frame, results)
                obstacles = self.detector.get_obstacle_info(results)

                processing_time = time.time() - start_time
                current_fps = 1 / processing_time if processing_time > 0 else 0

                cv2.putText(annotated_frame, f'FPS: {current_fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'Obstacles: {len(obstacles)}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if self.save_output and video_writer is not None:
                    video_writer.write(annotated_frame)

                if self.show_display:
                    cv2.imshow('Obstacle Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            self.capture.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
