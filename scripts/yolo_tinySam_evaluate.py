import os, sys, json
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
    count = 0
    for img_id in img_ids:
        if max_count is not None and count >= max_count:
            break

        img_file_name = f"{img_id:012d}.jpg"
        img_path = os.path.join(val_img_path, img_file_name)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = yolo12_predict(image, yolo_model)

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

        count += 1

    cwd = os.getcwd()
    outputs_dir = os.path.join(cwd, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    out_json = os.path.join(outputs_dir, out_json)  # where out_json_name is the filename

    with open(out_json, "w") as f:
        json.dump(all_records, f)

    print(f"Saved to: {out_json}")

def eval_zero_shot(val_img_path,val_json_path,sam_checkpoint_path, yolo_checkpoint_path, bbox_file_path = "coco_bboxes.json", max_count=None ):
    cwd = os.getcwd()
    bbox_file = os.path.join(cwd, 'outputs', bbox_file_path)
    if os.path.exists(bbox_file) is False:
        print("Generating bounding boxes using YOLOv12...")
        generate_bboxes(val_img_path, val_json_path, yolo_checkpoint_path, out_json=bbox_file_path, max_count=max_count )

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
        masks, ious, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
        )
        
        new_seg=mask_util.encode(np.array(masks[np.argmax(ious)],order="F", dtype="uint8"))
        new_seg["counts"] = new_seg["counts"].decode("utf-8")
        res[i]["segmentation"]=new_seg

    for c in res:
         c.pop("bbox", None)
    save_res_json_file= os.path.join(cwd, 'outputs', 'coco_res_tinysam.json')
    
    with open(save_res_json_file, 'w') as fnew:
        json.dump(res, fnew)
        
    cocoGT= COCO(val_json_path)
    coco_dt = cocoGT.loadRes(res)
    coco_eval = COCOeval(cocoGT, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 

if __name__ == "__main__":
    cwd = os.getcwd()
    sam_checkpoint_path = os.path.join(cwd, 'checkpoints', 'tinysam_42.3.pth')
    yolo_checkpoint_path = os.path.join(cwd, 'checkpoints', 'yolov12x.pt')
    val_img_path = os.path.join(cwd, 'dataset', 'coco', 'images', 'val2017') + os.sep
    val_json_path = os.path.join(cwd, 'dataset', 'coco', 'annotations', 'instances_val2017.json')

    # Set max_images=None to evaluate all; use a small number (e.g., 10) for quick testing
    eval_zero_shot(
        val_img_path=val_img_path,
        val_json_path=val_json_path,
        sam_checkpoint_path=sam_checkpoint_path,
        yolo_checkpoint_path=yolo_checkpoint_path,
    )
