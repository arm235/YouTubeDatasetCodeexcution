
from pathlib import Path
import yaml

# ==========================================================
# PATH CONFIGURATION
# ==========================================================

PROJECT_DIR = Path(
    r"C:\Users\Aarati\PycharmProjects\YouTubeDataset"
)

YOLOV7_DIR = PROJECT_DIR / "yolov7"
DATASET_DIR = PROJECT_DIR / "datasets"

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
VALID_IMAGES = DATASET_DIR / "valid" / "images"
TEST_IMAGES = DATASET_DIR / "test" / "images"

TRAIN_LABELS = DATASET_DIR / "train" / "labels"
VALID_LABELS = DATASET_DIR / "valid" / "labels"
TEST_LABELS = DATASET_DIR / "test" / "labels"

DATA_YAML = PROJECT_DIR / "data_yolov7.yaml"

CLASS_NAMES = ["Dozer", "DumpTruck", "Excavator"]

# ==========================================================
# STEP 1: CHECK DATASET FOLDERS
# ==========================================================

print("\n" + "=" * 60)
print("STEP 1: CHECKING YOLOV7 DATASET")
print("=" * 60)

required_dirs = [
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VALID_IMAGES,
    VALID_LABELS,
    TEST_IMAGES,
    TEST_LABELS,
]

for folder in required_dirs:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{folder}"
        )
    print("FOUND:", folder)

# ==========================================================
# STEP 2: COUNT IMAGES
# ==========================================================

extensions = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

for split, folder in [
    ("Train", TRAIN_IMAGES),
    ("Validation", VALID_IMAGES),
    ("Test", TEST_IMAGES),
]:
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if not images:
        raise FileNotFoundError(
            f"No images found in {folder}"
        )

    print(f"{split} images: {len(images)}")

# ==========================================================
# STEP 3: CREATE YOLOV7 DATA YAML
# ==========================================================

data = {
    "train": str(TRAIN_IMAGES.resolve()).replace("\\", "/"),
    "val": str(VALID_IMAGES.resolve()).replace("\\", "/"),
    "test": str(TEST_IMAGES.resolve()).replace("\\", "/"),
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES,
}

with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("\n" + "=" * 60)
print("DATA YAML CREATED SUCCESSFULLY")
print("=" * 60)
print("YAML file:", DATA_YAML)
print("Train:", data["train"])
print("Validation:", data["val"])
print("Test:", data["test"])

print("\nYOLOv7 dataset configuration completed.")
