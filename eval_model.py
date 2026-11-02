# -*- coding: utf-8 -*-
"""
eval_model.py —— 在同一个验证集上量模型精度（任务书第 6 步「前后对比」用）

为什么要单独一个脚本：
    第一轮的基线模型（COCO 预训练的 yolo26n-seg）和第二轮的自训模型（model/best.pt）
    必须在【同一批验证图、同一套标注】上量，数字才能放在一起比。
    验证集里 40 张全是人工修正过的标注，不是模型自己画的草稿，所以这个 mAP 是可信的。

用法：
    python eval_model.py --model yolo26n-seg.pt --tag baseline   # 改造前
    python eval_model.py --model model/best.pt   --tag trained    # 改造后

结果写到 records/metrics_<tag>.json，compare.py 会读它生成对比页。
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
RECORDS = ROOT / "records"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="要评估的权重，例如 yolo26n-seg.pt 或 model/best.pt")
    ap.add_argument("--tag", required=True, help="这次评估的名字：baseline / trained")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--split", default="val", help="在哪个划分上评估，默认 val")
    ap.add_argument("--conf", type=float, default=0.001, help="mAP 要用很低的阈值，让 PR 曲线完整")
    a = ap.parse_args()

    yml = DATASET / "seg" / "data.yaml"
    if not yml.exists():
        raise SystemExit(f"× 找不到 {yml}，先跑 python export.py")

    mpath = Path(a.model)
    if not mpath.is_absolute() and not mpath.exists():
        mpath = ROOT / a.model
    if not mpath.exists():
        raise SystemExit(f"× 找不到权重 {a.model}")

    from ultralytics import YOLO

    print("=" * 58)
    print(f"  评估 {mpath.name}　划分={a.split}　尺寸={a.imgsz}")
    print("=" * 58)

    t0 = time.time()
    model = YOLO(str(mpath))
    r = model.val(
        data=str(yml),
        split=a.split,
        imgsz=a.imgsz,
        conf=a.conf,
        device="cpu",
        workers=0,
        plots=True,
        verbose=False,
    )

    box, seg = r.box, r.seg
    names = r.names
    per_class = {}
    # 每个类别的 mAP50，用来看「哪几类变好了、哪几类还是不行」
    if seg is not None and getattr(seg, "ap_class_index", None) is not None:
        for i, c in zip(seg.ap_class_index, seg.ap50):
            per_class[names[int(c)]] = round(float(i), 4)

    metrics = {
        "tag": a.tag,
        "model": str(mpath),
        "split": a.split,
        "imgsz": a.imgsz,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
        # 框：B = Box
        "box_map50": round(float(box.map50), 4),
        "box_map50_95": round(float(box.map), 4),
        "box_precision": round(float(box.mp), 4),
        "box_recall": round(float(box.mr), 4),
    }
    if seg is not None:
        # 掩码：M = Mask
        metrics.update({
            "mask_map50": round(float(seg.map50), 4),
            "mask_map50_95": round(float(seg.map), 4),
            "mask_precision": round(float(seg.mp), 4),
            "mask_recall": round(float(seg.mr), 4),
            "mask_map50_per_class": per_class,
        })

    RECORDS.mkdir(exist_ok=True)
    out = RECORDS / f"metrics_{a.tag}.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  框  mAP50 {metrics['box_map50']:.4f}　mAP50-95 {metrics['box_map50_95']:.4f}")
    if seg is not None:
        print(f"  掩码 mAP50 {metrics['mask_map50']:.4f}　mAP50-95 {metrics['mask_map50_95']:.4f}")
    print(f"  耗时 {metrics['seconds']} 秒　→ {out}")
    print("=" * 58)


if __name__ == "__main__":
    main()
