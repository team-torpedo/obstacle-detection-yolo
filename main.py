from config import load_config
from obstacle_detector import ObstacleDetector
from video_detector import VideoObstacleDetector


def main():
    cfg = load_config()

    # Get config sections with defaults
    model_cfg = cfg.get('model', {})
    video_cfg = cfg.get('video', {})
    tracker_cfg = cfg.get('tracker', {})

    detector = ObstacleDetector(
        model_config=model_cfg,
        tracker_config=tracker_cfg,
    )

    video_detector = VideoObstacleDetector(
        detector=detector,
        source=video_cfg.get('source', 0),
        show_display=video_cfg.get('show_display', True),
        save_output=video_cfg.get('save_output', False),
        output_path=video_cfg.get('output_path', 'output.avi'),
    )

    video_detector.start_detection()


if __name__ == '__main__':
    main()