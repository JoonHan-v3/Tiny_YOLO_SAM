import argparse
import torch
import numpy as np
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

YOLO_REPO = ROOT / "Yolov12"
TINYSAM_REPO = ROOT / "TinySAM"
sys.path.insert(0, str(YOLO_REPO))
sys.path.insert(0, str(TINYSAM_REPO))

from Yolov12.predict import load_yolo_model
from Yolov12.predict import yolo12_predict
from TinySAM.bbox_tiny_predict import tinySam_predict

# Check for CUDA availability
print("-" * 50)
print("Checking for CUDA availability...")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

def yolo12_tinysam_pipeline(image, yolov12_checkpoint=str(ROOT / "checkpoints" / "yolov12x.pt"), classes=None):
    yolo_model = load_yolo_model(yolov12_checkpoint)
    # Step 1: YOLOv12 Prediction
    det = yolo12_predict(image, yolo_model, classes=classes)
    if isinstance(det, dict):  # nothing detected
        return {"image": image, "instances": []}
    xyxy, conf, cls = det
    if len(xyxy) == 0:  # everything filtered out by `classes`
        return {"image": image, "instances": []}
    # Segment each detection with TinySAM
    seg_results = tinySam_predict(image, xyxy, use_center_points=True,
                                  checkpoint=ROOT / "checkpoints" / "tinysam_42.3.pth")

    # Bundle outputs
    instances = []
    for i, r in enumerate(seg_results):
        instances.append({
            "class": int(cls[i]),
            "det_conf": float(conf[i]),
            "box_xyxy": xyxy[i].tolist(),
            "mask": r["mask"],         # boolean HxW
            "seg_score": r["score"]    # TinySAM's best candidate score
        })

    return {"image": image, "instances": instances}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default=str(ROOT / "figs" / "demo.png"), help="Path to the input image")
    args = parser.parse_args()

    boxes = yolo12_tinysam_pipeline(args.image_path)
    print("Detected boxes:", boxes)

if __name__ == "__main__":
    main()