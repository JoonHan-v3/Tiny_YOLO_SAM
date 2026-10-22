"""Segment-everything with two pipelines, usable as a library or on one image.

TinySamEverything: TinySAM's hierarchical auto mask generator (dense point grid).
YoloTinySam:       YOLOv12 boxes -> TinySAM box prompts. With sparse=True it also
                   prompts a sparse point grid on pixels the boxes left uncovered
                   (background / missed objects), merging overlaps with mask NMS.

Both expose segment(bgr) -> (instances, timing) where each instance is
    {"rle": COCO RLE dict, "score": float, "class_id": int | None, "source": "box" | "point" | "grid"}
and timing is {"detect": s, "segment": s, "total": s}.

Single image (from anywhere):
    python scripts/everything_segmentation.py --pipeline yolo_tinysam_everything --image figs/demo.png
    python scripts/everything_segmentation.py --pipeline tinysam --image figs/demo.png --out out.png
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Yolov12"))
sys.path.insert(0, str(ROOT / "TinySAM"))

from tinysam import SamHierarchicalMaskGenerator, SamPredictor, sam_model_registry  # noqa: E402
from ultralytics import YOLO  # noqa: E402

EVERYTHING_PIPELINES = ("tinysam", "yolo_tinysam_everything")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def encode_mask(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def load_sam(checkpoint, device):
    sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    return sam


class TinySamEverything:
    """TinySAM alone: dense hierarchical grid prompting, no detector, no classes."""

    def __init__(self, sam, device, points_per_side=32, max_det=100):
        self.device, self.max_det = device, max_det
        self.generator = SamHierarchicalMaskGenerator(
            sam, points_per_side=points_per_side, output_mode="coco_rle")

    @torch.no_grad()
    def segment(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sync(self.device)
        t0 = time.perf_counter()
        anns = self.generator.hierarchical_generate(rgb)
        sync(self.device)
        dt = time.perf_counter() - t0
        anns.sort(key=lambda a: a["predicted_iou"], reverse=True)
        instances = [{"rle": a["segmentation"], "score": float(a["predicted_iou"]),
                      "class_id": None, "source": "grid"} for a in anns[:self.max_det]]
        return instances, {"detect": 0.0, "segment": dt, "total": dt}


@dataclass
class SparseConfig:
    grid: int = 16               # NxN grid of candidate points
    iou_thresh: float = 0.8      # min TinySAM predicted IoU for a point mask
    min_area: int = 100          # min number of new pixels in a point mask
    score_scale: float = 0.5     # point-mask scores are scaled so detector masks rank first
    merge_iou: float = 0.5       # mask-IoU NMS threshold among point masks
    points_per_batch: int = 64


class YoloTinySam:
    """YOLOv12 boxes -> TinySAM; optionally + sparse points on uncovered pixels."""

    def __init__(self, sam, device, yolo_checkpoint, sparse=False, sparse_cfg=None,
                 imgsz=1024, conf=0.15, iou=0.5, tta=False, max_det=100, center_point=False):
        self.device, self.sparse, self.cfg = device, sparse, sparse_cfg or SparseConfig()
        self.yolo_kw = dict(imgsz=imgsz, conf=conf, iou=iou, agnostic_nms=True,
                            max_det=max_det, augment=tta, verbose=False)
        self.max_det, self.center_point = max_det, center_point
        self.yolo = YOLO(str(yolo_checkpoint))
        self.yolo.to(device)
        self.names = self.yolo.names
        self.predictor = SamPredictor(sam)

    @torch.no_grad()
    def segment(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sync(self.device)
        t0 = time.perf_counter()
        # ultralytics treats numpy arrays as BGR (it flips to RGB internally)
        det = self.yolo(bgr, **self.yolo_kw)[0]
        sync(self.device)
        t1 = time.perf_counter()

        boxes = det.boxes.xyxy.cpu().numpy()
        scores = det.boxes.conf.cpu().numpy()
        classes = det.boxes.cls.cpu().numpy().astype(int)
        covered = np.zeros(rgb.shape[:2], dtype=bool)
        instances = []
        if len(boxes) or self.sparse:
            self.predictor.set_image(rgb)
        for box, score, cls in zip(boxes, scores, classes):
            kwargs = {}
            if self.center_point:
                kwargs = dict(
                    point_coords=np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32))
            masks, ious, _ = self.predictor.predict(box=box, **kwargs)
            best = masks[int(np.argmax(ious))]
            covered |= best
            instances.append({"rle": encode_mask(best), "score": float(score),
                              "class_id": int(cls), "source": "box"})
        if self.sparse:
            for m, sc in self._sparse_point_masks(covered):
                instances.append({"rle": encode_mask(m), "score": sc * self.cfg.score_scale,
                                  "class_id": None, "source": "point"})
        # enforce the per-image limit on the combined, ranked list
        instances.sort(key=lambda i: i["score"], reverse=True)
        instances = instances[:self.max_det]
        sync(self.device)
        t2 = time.perf_counter()
        return instances, {"detect": t1 - t0, "segment": t2 - t1, "total": t2 - t0}

    def _sparse_point_masks(self, covered):
        """Segment grid points on uncovered pixels; returns [(bool mask of new pixels, pred IoU)]."""
        c, p = self.cfg, self.predictor
        h, w = covered.shape
        xs = (np.arange(c.grid) + 0.5) / c.grid * w
        ys = (np.arange(c.grid) + 0.5) / c.grid * h
        pts = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        pts = pts[~covered[pts[:, 1].astype(int), pts[:, 0].astype(int)]]
        if len(pts) == 0:
            return []

        covered_t = torch.as_tensor(covered, device=self.device)
        cands = []
        for i in range(0, len(pts), c.points_per_batch):
            batch = p.transform.apply_coords(pts[i:i + c.points_per_batch], (h, w))
            coords = torch.as_tensor(batch, dtype=torch.float, device=self.device)[:, None, :]
            labels = torch.ones(coords.shape[:2], dtype=torch.int, device=self.device)
            masks, ious, _ = p.predict_torch(coords, labels)  # B x 3 x H x W
            best = ious.argmax(1)
            idx = torch.arange(len(best), device=self.device)
            masks, ious = masks[idx, best] & ~covered_t, ious[idx, best]  # keep only new pixels
            keep = (ious >= c.iou_thresh) & (masks.flatten(1).sum(1) >= c.min_area)
            cands += [(m.cpu().numpy(), float(s)) for m, s in zip(masks[keep], ious[keep])]

        # greedy mask-IoU NMS, best predicted IoU first
        cands.sort(key=lambda x: x[1], reverse=True)
        kept, kept_rles = [], []
        for m, sc in cands:
            rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
            if kept_rles and mask_util.iou([rle], kept_rles, [0] * len(kept_rles)).max() > c.merge_iou:
                continue
            kept.append((m, sc))
            kept_rles.append(rle)
        return kept


def build_pipeline(name, sam, device, yolo_checkpoint=None, max_det=100, points_per_side=32,
                   sparse_cfg=None, **yolo_kw):
    if name == "tinysam":
        return TinySamEverything(sam, device, points_per_side=points_per_side, max_det=max_det)
    if name in ("yolo_tinysam", "yolo_tinysam_everything"):
        return YoloTinySam(sam, device, yolo_checkpoint, sparse=name == "yolo_tinysam_everything",
                           sparse_cfg=sparse_cfg, max_det=max_det, **yolo_kw)
    raise ValueError(f"unknown pipeline: {name}")


def visualize(bgr, instances, out_path, alpha=0.5, seed=0):
    """Overlay every mask in a random colour; detector boxes' masks are outlined."""
    rng = np.random.default_rng(seed)
    canvas = bgr.astype(np.float32)
    # draw large masks first so small ones stay visible
    for inst in sorted(instances, key=lambda i: -mask_util.area(i["rle"])):
        m = mask_util.decode(inst["rle"]).astype(bool)
        color = rng.integers(40, 255, 3).astype(np.float32)
        canvas[m] = (1 - alpha) * canvas[m] + alpha * color
    out = canvas.astype(np.uint8)
    for inst in instances:
        if inst["source"] == "box":
            m = mask_util.decode(inst["rle"]).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline", choices=EVERYTHING_PIPELINES, default="yolo_tinysam_everything")
    p.add_argument("--image", type=Path, default=ROOT / "figs/demo.png")
    p.add_argument("--out", type=Path, default=None, help="overlay image (default: outputs/<pipeline>_<image>.png)")
    p.add_argument("--sam-checkpoint", type=Path, default=ROOT / "checkpoints/tinysam_42.3.pth")
    p.add_argument("--yolo-checkpoint", type=Path, default=ROOT / "checkpoints/yolov12x.pt")
    p.add_argument("--sparse-grid", type=int, default=16)
    p.add_argument("--points-per-side", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        sys.exit(f"cannot read image: {args.image}")
    pipe = build_pipeline(args.pipeline, load_sam(args.sam_checkpoint, device), device,
                          yolo_checkpoint=args.yolo_checkpoint, points_per_side=args.points_per_side,
                          sparse_cfg=SparseConfig(grid=args.sparse_grid))
    pipe.segment(bgr)  # warm-up
    instances, t = pipe.segment(bgr)

    covered = np.zeros(bgr.shape[:2], dtype=bool)
    for i in instances:
        covered |= mask_util.decode(i["rle"]).astype(bool)
    n_box = sum(i["source"] == "box" for i in instances)
    print(f"{args.pipeline}: {len(instances)} masks ({n_box} from boxes), "
          f"coverage {covered.mean():.0%}, {t['total'] * 1000:.0f} ms "
          f"(detect {t['detect'] * 1000:.0f} + segment {t['segment'] * 1000:.0f})")

    out = args.out or ROOT / "outputs" / f"{args.pipeline}_{args.image.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    visualize(bgr, instances, out)
    print(f"Saved overlay to {out}")


if __name__ == "__main__":
    main()
