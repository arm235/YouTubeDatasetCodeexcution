
from pathlib import Path
import subprocess
import sys
import torch

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------
YOLOV7_DIR = Path(__file__).resolve().parent
PROJECT_DIR = YOLOV7_DIR.parent

DATA_YAML = PROJECT_DIR / "data_yolov7.yaml"

TRAIN_SCRIPT = YOLOV7_DIR / "train.py"
VAL_SCRIPT = YOLOV7_DIR / "test.py"

CFG = YOLOV7_DIR / "cfg" / "training" / "yolov7.yaml"
HYP = YOLOV7_DIR / "data" / "hyp.scratch.p5.yaml"
WEIGHTS = YOLOV7_DIR / "yolov7.pt"

RUNS_DIR = PROJECT_DIR / "runs" / "train"

# --------------------------------------------------
# 2. TRAINING SETTINGS
# --------------------------------------------------
EPOCHS = 100
BATCH_SIZE = 8
IMAGE_SIZE = 640
WORKERS = 0  # Often more reliable on Windows

DEVICE = "0" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# 3. CHECK REQUIRED FILES
# --------------------------------------------------
required_files = {
    "YOLOv7 training script": TRAIN_SCRIPT,
    "YOLOv7 validation script": VAL_SCRIPT,
    "Model configuration": CFG,
    "Hyperparameter configuration": HYP,
    "Dataset YAML": DATA_YAML,
    "Pretrained weights": WEIGHTS,
}

print("\nChecking required files...")

missing = False

for description, path in required_files.items():
    if path.exists():
        print(f"[OK] {description}: {path}")
    else:
        print(f"[MISSING] {description}: {path}")
        missing = True

if missing:
    raise FileNotFoundError(
        "\nSome required files are missing. "
        "Check your YOLOv7 repository and dataset YAML paths."
    )

# --------------------------------------------------
# 4. DISPLAY DEVICE
# --------------------------------------------------
print("\n" + "=" * 60)
print("YOLOv7 CUSTOM OBJECT DETECTION")
print("=" * 60)
print(f"Project directory : {PROJECT_DIR}")
print(f"Dataset YAML      : {DATA_YAML}")
print(f"Device            : {DEVICE}")
print(f"Epochs            : {EPOCHS}")
print(f"Batch size        : {BATCH_SIZE}")
print(f"Image size        : {IMAGE_SIZE}")

if torch.cuda.is_available():
    print(f"GPU               : {torch.cuda.get_device_name(0)}")
else:
    print("GPU unavailable; training will use CPU.")

# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------
train_command = [
    sys.executable,
    str(TRAIN_SCRIPT),
    "--weights", str(WEIGHTS),
    "--cfg", str(CFG),
    "--data", str(DATA_YAML),
    "--hyp", str(HYP),
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--img-size", str(IMAGE_SIZE), str(IMAGE_SIZE),
    "--device", DEVICE,
    "--workers", str(WORKERS),
    "--project", str(RUNS_DIR),
    "--name", "mine_yolov7",
    "--exist-ok",
]

print("\nStarting YOLOv7 training...\n")
subprocess.run(train_command, cwd=str(YOLOV7_DIR), check=True)

# --------------------------------------------------
# 6. LOCATE BEST TRAINED WEIGHTS
# --------------------------------------------------
BEST_WEIGHTS = RUNS_DIR / "mine_yolov7" / "weights" / "best.pt"

if not BEST_WEIGHTS.exists():
    raise FileNotFoundError(
        f"Training finished, but best.pt was not found at:\n"
        f"{BEST_WEIGHTS}"
    )

print(f"\nBest model found: {BEST_WEIGHTS}")

# --------------------------------------------------
# 7. VALIDATE TRAINED MODEL
# --------------------------------------------------
val_command = [
    sys.executable,
    str(VAL_SCRIPT),
    "--data", str(DATA_YAML),
    "--weights", str(BEST_WEIGHTS),
    "--img-size", str(IMAGE_SIZE),
    "--batch-size", str(BATCH_SIZE),
    "--device", DEVICE,
    "--task", "val",
    "--project", str(PROJECT_DIR / "runs" / "val"),
    "--name", "mine_yolov7_validation",
    "--exist-ok",
]

print("\nStarting validation...\n")
subprocess.run(val_command, cwd=str(YOLOV7_DIR), check=True)

# --------------------------------------------------
# 8. COMPLETION
# --------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING AND VALIDATION COMPLETED")
print("=" * 60)
print(f"Best weights : {BEST_WEIGHTS}")
print(f"Training log : {RUNS_DIR / 'mine_yolov7'}")
print(
    "Validation results:",
    PROJECT_DIR / "runs" / "val" / "mine_yolov7_validation"
)
