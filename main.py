
from pathlib import Path
import csv
import torch
import yaml
from ultralytics import YOLO

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_DIR = Path(
    r"C:\Users\Aarati\PycharmProjects\YouTubeDataset"
)

DATASET_DIR = PROJECT_DIR / "datasets"

DATA_YAML = PROJECT_DIR / "data.yaml"
TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

VALID_IMAGES = DATASET_DIR / "valid" / "images"
VALID_LABELS = DATASET_DIR / "valid" / "labels"

TEST_IMAGES = DATASET_DIR / "test" / "images"
TEST_LABELS = DATASET_DIR / "test" / "labels"

OUTPUT_DIR = PROJECT_DIR / "training_results_2nd_itr"
CSV_PATH = OUTPUT_DIR / "training_comparison.csv"

IMAGE_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 20
SEED = 42
WORKERS = 4

MODELS = {
    "YOLOv5u-small": "yolov5su.pt",
    "YOLOv8n": "yolov8n.pt",
    "YOLO11n": "yolo11n.pt",
}


# ==========================================================
# STEP 1: CHECK DATASET AND VALIDATION PATHS
# ==========================================================

def check_dataset():
    print("\n" + "=" * 65)
    print("STEP 1: CHECKING DATASET")
    print("=" * 65)

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Missing data.yaml: {DATA_YAML}")

    if not VALID_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Validation images folder not found: {VALID_IMAGES}"
        )

    if not VALID_LABELS.is_dir():
        raise FileNotFoundError(
            f"Validation labels folder not found: {VALID_LABELS}"
        )

    extensions = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff", ".webp"
    }

    images = [
        p for p in VALID_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if not images:
        raise FileNotFoundError(
            f"No images found in {VALID_IMAGES}"
        )

    # Validate that the YAML points to the intended validation images.
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("data.yaml must contain a YAML mapping.")

    data_root = Path(data.get("path", PROJECT_DIR))
    if not data_root.is_absolute():
        data_root = PROJECT_DIR / data_root

    val_entry = data.get("val")
    if not val_entry:
        raise ValueError("data.yaml does not define a 'val' path.")

    val_path = Path(val_entry)
    if not val_path.is_absolute():
        val_path = data_root / val_path

    if val_path.resolve() != VALID_IMAGES.resolve():
        raise ValueError(
            "The validation path in data.yaml does not match "
            f"valid/images.\nYAML path resolves to: {val_path.resolve()}\n"
            f"Expected: {VALID_IMAGES.resolve()}\n"
            "Correct data.yaml before training."
        )

    print("Dataset YAML:", DATA_YAML)
    print("Validation images:", VALID_IMAGES)
    print("Validation labels:", VALID_LABELS)
    print("Number of validation images:", len(images))

    return data


# ==========================================================
# STEP 2: TRAIN AND VALIDATE MODELS
# ==========================================================

def main():
    data = check_dataset()

    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else "cpu"

    print("\nPyTorch version:", torch.__version__)
    print("CUDA available:", use_gpu)

    if use_gpu:
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU. Training may be slower.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_name, initial_weights in MODELS.items():
        print("\n" + "=" * 65)
        print("TRAINING:", model_name)
        print("=" * 65)

        model = YOLO(initial_weights)

        model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=device,
            workers=WORKERS,
            patience=PATIENCE,
            seed=SEED,
            deterministic=True,
            amp=use_gpu,
            pretrained=True,
            project=str(OUTPUT_DIR),
            name=model_name,
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True,
        )

        # --------------------------------------------------
        # STEP 3: CHECK TRAINED CHECKPOINT
        # --------------------------------------------------

        best_weights = (
            OUTPUT_DIR / model_name / "weights" / "best.pt"
        )

        print("\nChecking best checkpoint:", best_weights)

        if not best_weights.is_file():
            print("WARNING: best.pt not found. Skipping evaluation.")
            del model
            if use_gpu:
                torch.cuda.empty_cache()
            continue

        print("Checkpoint exists:", best_weights)
        print("Checkpoint size:", f"{best_weights.stat().st_size:,} bytes")

        # --------------------------------------------------
        # STEP 4: VALIDATE BEST CHECKPOINT
        # --------------------------------------------------

        print("\nEvaluating:", model_name)

        best_model = YOLO(str(best_weights))

        metrics = best_model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=device,
            workers=WORKERS,
            plots=True,
            project=str(OUTPUT_DIR),
            name=f"{model_name}_validation",
            exist_ok=True,
            verbose=True,
        )

        row = {
            "model": model_name,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "best_weights": str(best_weights),
        }

        all_results.append(row)

        print("\nRESULTS:", model_name)
        print(f"Precision       : {row['precision']:.4f}")
        print(f"Recall          : {row['recall']:.4f}")
        print(f"mAP@0.5         : {row['mAP50']:.4f}")
        print(f"mAP@0.5:0.95    : {row['mAP50_95']:.4f}")

        del model, best_model
        if use_gpu:
            torch.cuda.empty_cache()

        # Save results after each model, so completed evaluations
        # remain recorded if a later model fails.
        save_results(all_results)

    print("\n" + "=" * 65)
    print("ALL COMPLETED MODEL EVALUATIONS")
    print("=" * 65)

    for row in all_results:
        print(
            f"{row['model']:16s} "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"mAP50={row['mAP50']:.4f} "
            f"mAP50-95={row['mAP50_95']:.4f}"
        )

    print("\nCSV saved to:", CSV_PATH)


# ==========================================================
# STEP 5: SAVE COMPARISON CSV
# ==========================================================

def save_results(results):
    if not results:
        return

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
