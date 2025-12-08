from config import load_config
from obstacle_detector import ObstacleDetector
from video_detector import VideoObstacleDetector


def main():
    cfg = load_config()

    detector = ObstacleDetector(
        model_path=cfg.get('model_path', 'yolov8n.pt'),
        confidence_threshold=cfg.get('confidence_threshold', 0.5),
        max_history_length=cfg.get('max_history_length', 30),
        focal_length=cfg.get('focal_length', 500),
        real_height=cfg.get('real_height', 1.7),
    )

    video_detector = VideoObstacleDetector(
        detector=detector,
        source=cfg.get('video_source', 0),
        show_display=cfg.get('show_display', True),
        save_output=cfg.get('save_output', False),
        output_path=cfg.get('output_path', 'output.avi'),
    )

    video_detector.start_detection()


if __name__ == '__main__':
    main()