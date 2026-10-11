
from pathlib import Path
import sys
import subprocess
import torch

# ==================================================
# CONFIGURATION
# ==================================================
YOLOV7_DIR = Path(
    r"C:\Users\Aarati\PycharmProjects\YouTubeDataset\yolov7"
)

DATA_YAML = Path(
    r"C:\Users\Aarati\PycharmProjects\YouTubeDataset\data.yaml"
)

WEIGHTS = YOLOV7_DIR / "yolov7_training.pt"

# Use a configuration that actually exists in cfg/training
CFG_DIR = YOLOV7_DIR / "cfg" / "training"

OUTPUT_NAME = "mining_yolov7"
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 4

DEVICE = "0" if torch.cuda.is_available() else "cpu"


# ==================================================
# RUN YOLOV7 SCRIPTS
# ==================================================
def run_script(script_name, args):
    script_path = YOLOV7_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script not found: {script_path}"
        )

    command = [
        sys.executable,
        str(script_path),
        *[str(arg) for arg in args],
    ]

    print("\nRunning command:")
    print(" ".join(f'"{arg}"' if " " in arg else arg
                   for arg in command), flush=True)

    # Stream output directly so training progress is visible.
    subprocess.run(
        command,
        cwd=str(YOLOV7_DIR),
        check=True,
    )


# ==================================================
# MAIN
# ==================================================
def main():

    print("=" * 60)
    print("YOLOv7 MINING OBJECT DETECTION")
    print("=" * 60)

    # Check repository and required files
    if not (YOLOV7_DIR / "train.py").exists():
        raise FileNotFoundError(
            f"train.py not found: {YOLOV7_DIR}"
        )

    if not (YOLOV7_DIR / "test.py").exists():
        raise FileNotFoundError(
            f"test.py not found: {YOLOV7_DIR}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Pretrained weights not found: {WEIGHTS}"
        )

    # Find a real training configuration
    if not CFG_DIR.exists():
        raise FileNotFoundError(
            f"Configuration directory not found: {CFG_DIR}"
        )

    cfg_files = sorted(CFG_DIR.glob("*.yaml"))

    print("\nAvailable training configurations:")
    for cfg in cfg_files:
        print(" -", cfg.name)

    # Prefer the standard YOLOv7 architecture if present.
    cfg_candidates = [
        CFG_DIR / "yolov7.yaml",
        CFG_DIR / "yolov7-custom.yaml",
    ]

    CFG = next(
        (path for path in cfg_candidates if path.exists()),
        None,
    )

    if CFG is None:
        raise FileNotFoundError(
            "Neither cfg/training/yolov7.yaml nor "
            "cfg/training/yolov7-custom.yaml exists. "
            "Check the configurations printed above and "
            "select the correct architecture file."
        )

    # Check dataset configuration
    print("\nDataset YAML:", DATA_YAML)
    print("Model configuration:", CFG)
    print("Initial weights:", WEIGHTS)

    print("\nPyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Training will use CPU.")

    # ==================================================
    # STEP 1: TRAIN
    # ==================================================
    print("\n" + "=" * 60)
    print("STEP 1: STARTING TRAINING")
    print("=" * 60)

    run_script(
        "train.py",
        [
            "--workers", WORKERS,
            "--device", DEVICE,
            "--batch-size", BATCH_SIZE,
            "--data", str(DATA_YAML),
            "--img-size", IMAGE_SIZE, IMAGE_SIZE,
            "--cfg", str(CFG),
            "--weights", str(WEIGHTS),
            "--name", OUTPUT_NAME,
            "--hyp", "data/hyp.scratch.custom.yaml",
            "--epochs", EPOCHS,
        ],
    )

    # ==================================================
    # STEP 2: FIND TRAINED CHECKPOINT
    # ==================================================
    print("\n" + "=" * 60)
    print("STEP 2: CHECKING TRAINED WEIGHTS")
    print("=" * 60)

    # Original YOLOv7 commonly writes to runs/train,
    # but search for the exact run name in case the
    # repository uses a different output location.
    run_dirs = list(YOLOV7_DIR.rglob(OUTPUT_NAME))
    best_candidates = [
        run_dir / "weights" / "best.pt"
        for run_dir in run_dirs
    ]

    best_weights = next(
        (path for path in best_candidates if path.is_file()),
        None,
    )

    if best_weights is None:
        expected = (
            YOLOV7_DIR / "runs" / "train" /
            OUTPUT_NAME / "weights"
        )

        last_candidates = [
            path for path in YOLOV7_DIR.rglob("last.pt")
            if OUTPUT_NAME in path.parts
        ]

        if last_candidates:
            print("Found last.pt checkpoint(s):")
            for path in last_candidates:
                print(" -", path)

        raise FileNotFoundError(
            f"Training finished without a discoverable best.pt.\n"
            f"Expected weights directory: {expected}\n"
            "Check the training output and run directory."
        )

    print("Best checkpoint:", best_weights)

    # ==================================================
    # STEP 3: VALIDATE / TEST
    # ==================================================
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATING BEST CHECKPOINT")
    print("=" * 60)

    run_script(
        "test.py",
        [
            "--data", str(DATA_YAML),
            "--img-size", IMAGE_SIZE,
            "--batch-size", BATCH_SIZE,
            "--conf-thres", "0.001",
            "--iou-thres", "0.65",
            "--device", DEVICE,
            "--weights", str(best_weights),
            "--task", "val",
        ],
    )

    print("\n" + "=" * 60)
    print("TRAINING AND VALIDATION COMPLETED")
    print("Best checkpoint:", best_weights)
    print("=" * 60)


if __name__ == "__main__":
    main()
