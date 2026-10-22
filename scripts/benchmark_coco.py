"""Benchmark COCO instance segmentation: YOLOv12 + TinySAM vs. TinySAM alone.

Pipeline A (yolo_tinysam): YOLOv12 proposes boxes -> TinySAM segments each box.
Pipeline B (tinysam):      TinySAM "segment everything" (hierarchical auto mask
                           generator), no detector and no class labels.

Pipeline C (yolo_tinysam_everything): "everything" mode guided by YOLO. YOLO boxes
                           are segmented as in A, then a sparse point grid is
                           placed only on pixels the box masks left uncovered and
                           TinySAM segments those points (background / missed
                           objects). Overlapping point masks are merged with NMS.

Metrics (COCO segm AP via pycocotools) and per-image latency are written to
outputs/benchmark/. Because pipeline B has no class labels, both pipelines are
scored class-agnostically (useCats=0) for an apples-to-apples comparison;
pipeline A is additionally scored class-aware (the standard COCO metric).

Usage (from the project root):
    python scripts/benchmark_coco.py --max-images 100
    python scripts/benchmark_coco.py --pipelines tinysam --max-images 500
    python scripts/benchmark_coco.py            # full val2017
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent.parent

from everything_segmentation import SparseConfig, build_pipeline, load_sam  # noqa: E402

PIPELINES = ("yolo_tinysam", "tinysam", "yolo_tinysam_everything")


def yolo_names_to_coco_ids(names, coco):
    """Map YOLO class index -> COCO category id by class name."""
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    return {i: name_to_id[n] for i, n in names.items()}


def run_pipeline(pipe, args, coco, img_ids):
    """Run a pipeline from everything_segmentation over img_ids -> (COCO results, timings)."""
    cls_to_cat = yolo_names_to_coco_ids(pipe.names, coco) if hasattr(pipe, "names") else {}
    results, times = [], []
    for n, img_id in enumerate(img_ids):
        path = args.images / f"{img_id:012d}.jpg"
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"  skip (unreadable): {path}")
            continue
        instances, t = pipe.segment(bgr)
        for inst in instances:
            cid = inst["class_id"]
            results.append({
                "image_id": int(img_id),
                # unlabeled masks (grid / sparse points) get a dummy category: class-agnostic eval only
                "category_id": int(cls_to_cat[cid]) if cid is not None else 1,
                "segmentation": inst["rle"],
                "score": inst["score"],
            })
        times.append((t["detect"], t["segment"], t["total"]))
        if (n + 1) % 100 == 0:
            print(f"  [{n + 1}/{len(img_ids)}] {np.mean([x[2] for x in times]) * 1000:.0f} ms/img")
    return results, times


def mean_coverage(results, coco, img_ids):
    """Mean fraction of image pixels covered by the union of predicted masks."""
    by_img = {}
    for r in results:
        by_img.setdefault(r["image_id"], []).append(r["segmentation"])
    fracs = []
    for i in img_ids:
        info = coco.imgs[i]
        union = np.zeros((info["height"], info["width"]), dtype=bool)
        for rle in by_img.get(i, []):
            union |= mask_util.decode(rle).astype(bool)
        fracs.append(union.mean())
    return float(np.mean(fracs))


def coco_eval(coco_gt, results, img_ids, use_cats):
    stats = {}
    if not results:
        return {k: 0.0 for k in ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR100")}
    dt = coco_gt.loadRes(copy.deepcopy(results))  # loadRes mutates its input
    ev = COCOeval(coco_gt, dt, "segm")
    ev.params.imgIds = img_ids
    ev.params.useCats = int(use_cats)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    stats.update(AP=s[0], AP50=s[1], AP75=s[2], APs=s[3], APm=s[4], APl=s[5], AR100=s[8])
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipelines", nargs="+", choices=PIPELINES, default=list(PIPELINES))
    p.add_argument("--images", type=Path, default=ROOT / "dataset/coco/images/val2017")
    p.add_argument("--annotations", type=Path, default=ROOT / "dataset/coco/annotations/instances_val2017.json")
    p.add_argument("--sam-checkpoint", type=Path, default=ROOT / "checkpoints/tinysam_42.3.pth")
    p.add_argument("--yolo-checkpoint", type=Path, default=ROOT / "checkpoints/yolov12x.pt")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/benchmark")
    p.add_argument("--max-images", type=int, default=None, help="evaluate only the first N images (default: all)")
    p.add_argument("--max-det", type=int, default=100, help="max instances per image (COCO maxDets=100)")
    p.add_argument("--yolo-imgsz", type=int, default=1024)
    p.add_argument("--yolo-conf", type=float, default=0.15)
    p.add_argument("--yolo-iou", type=float, default=0.5)
    p.add_argument("--yolo-tta", action="store_true", help="enable YOLO test-time augmentation (slower)")
    p.add_argument("--center-point", action="store_true", help="add a centre-point prompt alongside each box")
    p.add_argument("--sparse-grid", type=int, default=16, help="yolo_tinysam_everything: grid size (NxN) for uncovered-area points")
    p.add_argument("--sparse-iou-thresh", type=float, default=0.8, help="min TinySAM predicted IoU for point masks")
    p.add_argument("--sparse-min-area", type=int, default=100, help="min new-pixel area (px) for point masks")
    p.add_argument("--sparse-score-scale", type=float, default=0.5, help="point-mask scores are scaled by this so detector masks rank first")
    p.add_argument("--merge-iou", type=float, default=0.5, help="mask-IoU NMS threshold among point masks")
    p.add_argument("--points-per-batch", type=int, default=64)
    p.add_argument("--points-per-side", type=int, default=32, help="TinySAM everything-mode grid density")
    p.add_argument("--class-agnostic", action="store_true",
                   help="only report class-agnostic metrics (skip the class-aware eval of yolo_tinysam)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(args.annotations))
    img_ids = sorted(coco.getImgIds())
    if args.max_images:
        img_ids = img_ids[:args.max_images]
    print(f"Benchmarking on {len(img_ids)} images, device={device}")

    sam = load_sam(args.sam_checkpoint, device)
    sparse_cfg = SparseConfig(grid=args.sparse_grid, iou_thresh=args.sparse_iou_thresh,
                              min_area=args.sparse_min_area, score_scale=args.sparse_score_scale,
                              merge_iou=args.merge_iou, points_per_batch=args.points_per_batch)

    summary = {}
    for name in args.pipelines:
        print(f"\n===== Pipeline: {name} =====")
        pipe = build_pipeline(
            name, sam, device, yolo_checkpoint=args.yolo_checkpoint, max_det=args.max_det,
            points_per_side=args.points_per_side, sparse_cfg=sparse_cfg, imgsz=args.yolo_imgsz,
            conf=args.yolo_conf, iou=args.yolo_iou, tta=args.yolo_tta, center_point=args.center_point)
        # warm-up so CUDA init / cudnn autotune don't skew the first image
        run_pipeline(pipe, args, coco, img_ids[:1])
        results, times = run_pipeline(pipe, args, coco, img_ids)
        del pipe

        with open(args.out_dir / f"{name}_results.json", "w") as f:
            json.dump(results, f)

        if not times:
            print(f"No images were processed for {name} (all unreadable?); skipping.")
            continue
        t = np.array(times)
        entry = {
            "num_images": len(times),
            "num_instances": len(results),
            "ms_per_image": float(t[:, 2].mean() * 1000),
            "ms_detect": float(t[:, 0].mean() * 1000),
            "ms_segment": float(t[:, 1].mean() * 1000),
            "coverage": mean_coverage(results, coco, img_ids),
            "class_agnostic": coco_eval(coco, results, img_ids, use_cats=False),
        }
        if name == "yolo_tinysam" and not args.class_agnostic:
            entry["class_aware"] = coco_eval(coco, results, img_ids, use_cats=True)
        summary[name] = entry

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== Summary (segm, class-agnostic) =====")
    print(f"{'pipeline':<25}{'AP':>7}{'AP50':>7}{'AP75':>7}{'APs':>7}{'APm':>7}{'APl':>7}{'AR100':>7}{'ms/img':>9}{'inst/img':>10}{'cover':>7}")
    for name, e in summary.items():
        a = e["class_agnostic"]
        print(f"{name:<25}{a['AP']:>7.3f}{a['AP50']:>7.3f}{a['AP75']:>7.3f}{a['APs']:>7.3f}"
              f"{a['APm']:>7.3f}{a['APl']:>7.3f}{a['AR100']:>7.3f}{e['ms_per_image']:>9.1f}"
              f"{e['num_instances'] / max(e['num_images'], 1):>10.1f}{e['coverage']:>7.2f}")
    if "class_aware" in summary.get("yolo_tinysam", {}):
        a = summary["yolo_tinysam"]["class_aware"]
        print(f"\nYOLO+TinySAM class-aware: AP={a['AP']:.3f} AP50={a['AP50']:.3f} AP75={a['AP75']:.3f}")
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
