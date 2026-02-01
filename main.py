from ultralytics import YOLO
import torch

if __name__ == '__main__':
    torch.cuda.empty_cache()

    MODEL_VARIANT = r'<model_architecture_path>'
    DATASET_CONFIG = 'data.yaml'
    EXPERIMENT_NAME = 'Train_ver_0'

    if not torch.cuda.is_available():
        print(" GPU not available!")
        exit(1)

    print(f"\n GPU: {torch.cuda.get_device_name(0)}")
    print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"\nLoading model: {MODEL_VARIANT}")
    model = YOLO(MODEL_VARIANT)
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f" Total parameters: {total_params:,}")

    training_config = {

        'data': DATASET_CONFIG,
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'device': 0,

        'optimizer': 'AdamW',
        'lr0': 0.002,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # Memory optimization
        'amp': True,
        'cache': False,
        'workers': 0,

        # Learning rate
        'cos_lr': True,
        'warmup_epochs': 3,

        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.0,
        'erasing': 0.4,

        # Training settings
        'patience': 100,
        'save': True,
        'save_period': -1,
        'plots': True,
        'val': True,
        'deterministic': True,

        # System
        'project': 'runs/detect',
        'name': EXPERIMENT_NAME,
        'exist_ok': False,
        'verbose': True,
        'seed': 0,
        'close_mosaic': 10,
    }

    print("\nTraining Configuration:")
    print(f"  Batch size: {training_config['batch']}")
    print(f"  Image size: {training_config['imgsz']}")
    print(f"  Workers: {training_config['workers']}")
    print(f"  Mixed Precision: {training_config['amp']}")

    print("\n" + "=" * 80)
    print("Starting GPU Training...")
    print("=" * 80 + "\n")

    try:
        print(f"GPU Memory before training:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print()
        results = model.train(**training_config)

        print("\n" + "=" * 80)
        print("Training Completed Successfully!")
        print("=" * 80)
        print(f"\n✓ Results saved to: {results.save_dir}")
        print(f"✓ Best weights: {results.save_dir}/weights/best.pt")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "=" * 80)
            print(" GPU OUT OF MEMORY ERROR")

        else:
            print(f"\n Training failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"\n Training failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clear GPU cache
        torch.cuda.empty_cache()
        print(f"\nFinal GPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")