import argparse, hashlib, os, sys, json
from pathlib import Path

import numpy as np
import torch
import cv2
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent.parent
YOLO_REPO = ROOT / "Yolov12"
TINYSAM_REPO = ROOT / "TinySAM"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(YOLO_REPO))
sys.path.insert(0, str(TINYSAM_REPO))

from Yolov12.predict import load_yolo_model, yolo12_predict
from TinySAM.tinysam import sam_model_registry, SamPredictor

YOLO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

def build_yolo_to_coco_mapping(coco):
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    name_to_catid = {c["name"]: c["id"] for c in cats}

    mapping = {}
    for yolo_id, name in enumerate(YOLO_CLASS_NAMES):
        if name not in name_to_catid:
            raise ValueError(f"Class '{name}' not found in COCO categories.")
        mapping[yolo_id] = name_to_catid[name]
    return mapping


def generate_bboxes(val_img_path, in_json, yolo_checkpoint, out_json="coco_bboxes.json", max_count=None):
    coco = COCO(in_json)
    yolo_to_coco = build_yolo_to_coco_mapping(coco)
    img_ids = coco.getImgIds()
    all_records = []

    yolo_model = load_yolo_model(yolo_checkpoint)
    if max_count is not None:
        img_ids = img_ids[:max_count]
    for img_id in img_ids:

        img_file_name = f"{img_id:012d}.jpg"
        img_path = os.path.join(val_img_path, img_file_name)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # ultralytics treats numpy arrays as BGR (it flips to RGB internally)
        result = yolo12_predict(image_bgr, yolo_model)

        # yolo12_predict returns a dict (no tuple) when nothing is detected
        if not isinstance(result, (list, tuple)) or len(result) not in (2, 3):
            continue
        if len(result) == 3:
            boxes, scores, classes = result
        else:
            boxes, classes = result
            scores = np.ones(len(classes), dtype=np.float32)

        boxes = np.asarray(boxes)
        scores = np.asarray(scores)
        classes = np.asarray(classes)

        if boxes.size == 0:
            continue

        boxes = np.atleast_2d(boxes)
        scores = np.atleast_1d(scores)
        classes = np.atleast_1d(classes)

        for b_idx in range(boxes.shape[0]):
            x1, y1, x2, y2 = map(float, boxes[b_idx][:4])

            # Convert xyxy → xywh (COCO-style)
            w = x2 - x1
            h = y2 - y1
            bbox_xywh = [x1, y1, w, h]

            cls_idx = int(classes[b_idx])            # YOLO class index
            cat_id = int(yolo_to_coco[cls_idx])      # COCO category_id
            cat_name = YOLO_CLASS_NAMES[cls_idx]     # optional, human-readable

            record = {
                "image_id": int(img_id),
                "file_name": img_file_name,
                "bbox": bbox_xywh,
                "score": float(scores[b_idx]),
                "class": cls_idx,          # YOLO index
                "category_id": cat_id,     # COCO category id (for COCOeval)
                "category_name": cat_name  # optional, just for readability
            }
            all_records.append(record)


    outputs_dir = str(ROOT / "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    out_json = os.path.join(outputs_dir, out_json)  # where out_json_name is the filename

    with open(out_json, "w") as f:
        json.dump(all_records, f)

    print(f"Saved to: {out_json}")

def _file_signature(path):
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": st.st_size, "mtime": int(st.st_mtime)}


def _dir_digest(path):
    """Hash of every file's name, size and mtime, so edited/replaced images invalidate the cache."""
    h = hashlib.sha256()
    for entry in sorted(os.scandir(path), key=lambda e: e.name):
        st = entry.stat()
        h.update(f"{entry.name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def eval_zero_shot(val_img_path,val_json_path,sam_checkpoint_path, yolo_checkpoint_path, bbox_file_path = "coco_bboxes.json", max_count=None, use_center_points=False):
    outputs_dir = str(ROOT / "outputs")
    if max_count is not None:
        # a subset run must not reuse (or overwrite) the full-set box cache
        stem, ext = os.path.splitext(bbox_file_path)
        bbox_file_path = f"{stem}_{max_count}{ext}"
    bbox_file = os.path.join(outputs_dir, bbox_file_path)
    # the cached boxes are only valid for the same detector weights, annotations and subset
    signature = {
        "yolo_checkpoint": _file_signature(yolo_checkpoint_path),
        "annotations": _file_signature(val_json_path),
        "image_dir": os.path.abspath(val_img_path),
        "images": _dir_digest(val_img_path),
        "max_count": max_count,
    }
    meta_file = bbox_file + ".meta.json"
    cache_valid = os.path.exists(bbox_file) and os.path.exists(meta_file)
    if cache_valid:
        with open(meta_file) as f:
            cache_valid = json.load(f) == signature
        if not cache_valid:
            print("Cached boxes are stale (checkpoint/annotations changed); regenerating...")
    if not cache_valid:
        print("Generating bounding boxes using YOLOv12...")
        generate_bboxes(val_img_path, val_json_path, yolo_checkpoint_path, out_json=bbox_file_path, max_count=max_count )
        with open(meta_file, "w") as f:
            json.dump(signature, f)

    print("============== Evaluating on COCO dataset:",sam_checkpoint_path)
    with open(bbox_file) as f:
         res = json.load(f)
    model_type = "vit_t"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    pre_img_id=0
    total_time=0
    print("Total instances num:",len(res))

    for i,res_ins in enumerate(res):
        res_ins=res[i]
        if i%10000==0:
            print(i)
        img_id=res_ins['image_id']
        img_file_name=f'{img_id:012d}'+'.jpg'

        if pre_img_id!=img_id:
            image = cv2.imread(os.path.join(val_img_path, img_file_name))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            pre_img_id=img_id

        input_box=res_ins['bbox']
        input_box=[input_box[0],input_box[1],input_box[0]+input_box[2],input_box[1]+input_box[3]]
        input_box = np.array(input_box)
        point_coords = point_labels = None
        if use_center_points:
            point_coords = np.array([[(input_box[0] + input_box[2]) / 2, (input_box[1] + input_box[3]) / 2]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
        masks, ious, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box[None, :],
        )
        
        new_seg=mask_util.encode(np.array(masks[np.argmax(ious)],order="F", dtype="uint8"))
        new_seg["counts"] = new_seg["counts"].decode("utf-8")
        res[i]["segmentation"]=new_seg

    res = [c for c in res if "segmentation" in c]  # drop instances of unreadable images
    for c in res:
         c.pop("bbox", None)
    save_res_json_file= os.path.join(outputs_dir, 'coco_res_tinysam.json' if max_count is None else f'coco_res_tinysam_{max_count}.json')
    
    with open(save_res_json_file, 'w') as fnew:
        json.dump(res, fnew)
        
    if not res:
        print("No detections/segmentations were produced; nothing to evaluate (AP = 0).")
        return

    cocoGT= COCO(val_json_path)
    coco_dt = cocoGT.loadRes(res)
    coco_eval = COCOeval(cocoGT, coco_dt, "segm")
    if max_count is not None:
        coco_eval.params.imgIds = cocoGT.getImgIds()[:max_count]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=None, help="evaluate only the first N images")
    parser.add_argument("--center-point", action="store_true", help="add a centre-point prompt to each box")
    cli = parser.parse_args()

    sam_checkpoint_path = str(ROOT / 'checkpoints' / 'tinysam_42.3.pth')
    yolo_checkpoint_path = str(ROOT / 'checkpoints' / 'yolov12x.pt')
    val_img_path = str(ROOT / 'dataset' / 'coco' / 'images' / 'val2017') + os.sep
    val_json_path = str(ROOT / 'dataset' / 'coco' / 'annotations' / 'instances_val2017.json')

    # Set max_images=None to evaluate all; use a small number (e.g., 10) for quick testing
    eval_zero_shot(
        val_img_path=val_img_path,
        val_json_path=val_json_path,
        sam_checkpoint_path=sam_checkpoint_path,
        yolo_checkpoint_path=yolo_checkpoint_path,
        max_count=cli.max_images,
        use_center_points=cli.center_point,
    )
