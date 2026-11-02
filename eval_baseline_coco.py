# -*- coding: utf-8 -*-
"""
eval_baseline_coco.py —— 公平地量出「改造前」的精度（第二轮回标前的基线）

问题在哪：
    我们的数据集是 10 类，编号 0~9（person=0, chair=1, ...）。
    但 yolo26n-seg.pt 是在 COCO 80 类上训的，它输出的 chair 是 56、couch 是 57。
    直接拿它去 val 我们的数据集，大部分预测会因为「类别编号对不上」被判成误检，
    mAP 会低到 0.09 —— 这不是模型差，是编号错位。

这个脚本怎么办：
    不改变任何模型，只把【验证集的标签临时翻译回 COCO 编号】（chair 1 → 56），
    再用 COCO 的 80 类名字去评估。这样算出来的 mAP 和自训模型的 mAP 是同一把尺子：
    都是「10 个类别各自的 AP50 取平均」。

产物：records/metrics_baseline.json（和 eval_model.py 的格式一致，compare.py 能直接读）
"""

import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
RECORDS = ROOT / "records"
TMP = DATASET / "_cocoval"


def main():
    cfg = yaml.safe_load((ROOT / "classes.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    coco_ids = cfg["coco_id"]              # 我们的 0~9 → COCO 编号

    from ultralytics import YOLO

    model = YOLO(str(ROOT / "yolo26n-seg.pt"))
    coco_names = model.names               # COCO 80 类，顺序就是模型输出的顺序

    # 1) 搭一个临时的 COCO 编号验证集
    if TMP.exists():
        shutil.rmtree(TMP)
    for sub in ("images/val", "labels/val"):
        (TMP / sub).mkdir(parents=True)

    src_img = DATASET / "seg" / "images" / "val"
    src_lb = DATASET / "seg" / "labels" / "val"
    n_lb = 0
    for lb in sorted(src_lb.glob("*.txt")):
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            cand = src_img / (lb.stem + ext)
            if cand.exists():
                img = cand
                break
        if img is None:
            continue
        shutil.copyfile(img, TMP / "images" / "val" / img.name)
        out = []
        for line in lb.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            cls = int(float(parts[0]))
            if cls >= len(coco_ids):
                continue
            out.append(" ".join([str(coco_ids[cls])] + parts[1:]))
        (TMP / "labels" / "val" / lb.name).write_text("\n".join(out) + "\n", encoding="utf-8")
        n_lb += 1

    yml = TMP / "data.yaml"
    yml.write_text(
        f"path: {TMP.as_posix()}\ntrain: images/val\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in coco_names.items()),
        encoding="utf-8")

    print("=" * 58)
    print(f"  基线评估（COCO 编号对齐后）　验证图 {n_lb} 张")
    print("=" * 58)

    r = model.val(data=str(yml), split="val", imgsz=512, conf=0.001,
                  device="cpu", workers=0, plots=False, verbose=False)

    def collect(m):
        """只取我们这 10 类的 AP50，取平均 —— 与自训模型的 'all' 口径一致"""
        idx = list(getattr(m, "ap_class_index", []))
        ap50 = list(getattr(m, "ap50", []))
        per = {}
        for cls_id, ap in zip(idx, ap50):          # cls_id 是 COCO 编号，ap 是该类 AP50
            if int(cls_id) in coco_ids:
                per[names[coco_ids.index(int(cls_id))]] = round(float(ap), 4)
        vals = [v for v in per.values()]
        return (round(sum(vals) / len(vals), 4) if vals else 0.0), per

    box_map50, box_per = collect(r.box)
    seg_map50, seg_per = (collect(r.seg) if r.seg is not None else (0.0, {}))

    metrics = {
        "tag": "baseline",
        "model": "yolo26n-seg.pt（COCO 预训练，未用本数据集训练）",
        "split": "val",
        "imgsz": 512,
        "note": "标签已临时翻译成 COCO 编号再评估，与本项目 10 类口径等价",
        "val_images": n_lb,
        "box_map50": box_map50,
        "box_map50_95": round(float(r.box.map), 4),
        "box_precision": round(float(r.box.mp), 4),
        "box_recall": round(float(r.box.mr), 4),
        "box_map50_per_class": box_per,
    }
    if r.seg is not None:
        metrics.update({
            "mask_map50": seg_map50,
            "mask_map50_95": round(float(r.seg.map), 4),
            "mask_precision": round(float(r.seg.mp), 4),
            "mask_recall": round(float(r.seg.mr), 4),
            "mask_map50_per_class": seg_per,
        })

    RECORDS.mkdir(exist_ok=True)
    out = RECORDS / "metrics_baseline.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  框  mAP50 {box_map50:.4f}　掩码 mAP50 {seg_map50:.4f}")
    for n in names:
        if n in box_per:
            print(f"    {n:<14} 框 {box_per[n]:.3f}　掩码 {seg_per.get(n, 0):.3f}")
    print(f"  → {out}")
    print("=" * 58)


if __name__ == "__main__":
    main()
