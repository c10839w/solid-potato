# -*- coding: utf-8 -*-
"""
round2_compare.py —— 第二轮回标对比（任务书第 6 步的核心证据）

要回答的问题：
    「人改过一轮数据、模型重训之后，模型的草稿是不是变好了？」
    好多少 —— 用数字说话，而不是感觉。

怎么量（对同一批人工核对过的验证图，跑两遍草稿）：
    第一轮草稿：用原始 COCO 模型 yolo26n-seg.pt 画      ← 改造前
    第二轮草稿：用自己训出来的 model/best.pt 画          ← 改造后
    两者都拿「人工确认过的标注」当标准答案，逐个物体比对：
      · 类别对 + 框重合度 IoU≥0.5  → 这一处不用改
      · 类别错 / 位置歪 / 漏标     → 这一处要人工修
      · 模型多标的物体             → 这一处要人工删

    修正量从 M1 降到 M2，就是「人机协同迭代」的效果。

用法：
    python round2_compare.py                 # 默认在验证集 40 张上比
    python round2_compare.py --limit 20      # 只想快速看 20 张

产物：
    records/round2_report.json   逐图、逐类的对比数据
    records/compare.html         一页给人看的对比报告
"""

import argparse
import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
RECORDS = ROOT / "records"
ANN = RECORDS / "annotations"
MODEL_DIR = ROOT / "model"
IOU_OK = 0.5          # 框重合度达到 0.5 就算「框对了」


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def poly_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def count_fixes(gt, drafts):
    """比对标准答案和模型草稿，返回「需要人工处理的处数」及明细"""
    used = set()
    miss, wrong_cls, drift = 0, 0, 0

    for g in gt:
        gb = g.get("bbox") or poly_bbox(g["polygon"])
        best_i, best_iou = None, 0.0
        for i, d in enumerate(drafts):
            if i in used:
                continue
            iou = bbox_iou(gb, d["bbox"])
            if iou > best_iou:
                best_i, best_iou = i, iou

        if best_i is None or best_iou < 0.1:
            miss += 1                      # 完全没标出来
            continue
        used.add(best_i)
        d = drafts[best_i]
        if d["class_id"] != g["class_id"]:
            wrong_cls += 1                 # 框对了但类别认错
        elif best_iou < IOU_OK:
            drift += 1                     # 类别对但位置歪

    extra = len(drafts) - len(used)         # 模型多标的、要人工删掉
    return {
        "need_fix": miss + wrong_cls + drift + extra,
        "miss": miss,
        "wrong_class": wrong_cls,
        "drift": drift,
        "extra": extra,
        "gt_objects": len(gt),
        "draft_objects": len(drafts),
    }


def draft_with(model, img_path, coco_mode=False, conf=0.25):
    """画一版草稿。
    coco_mode=True 表示这是 COCO 80 类的原始模型：必须像界面预标注那样
    先按十类过滤、再把编号翻译成我们的 0~9，否则它会把 80 类的框都算进来，
    两边口径不一致，比出来的数字没有意义。"""
    import prelabel as P

    kw = {"classes": P.COCO_IDS} if coco_mode else {}
    r = model.predict(str(img_path), conf=conf, verbose=False, **kw)[0]
    out = []
    if r.masks is None:
        return out
    for i, (cls, score) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())):
        # 统一走 prelabel 里那份"扣掉补边 + 只取最大外轮廓"的实现，
        # 免得这里另写一份、坐标又偏回去（对比数据必须和界面草稿同一口径）
        poly = P._mask_to_polygon(r.masks.data[i], r.orig_shape, min_area=20)
        if poly is None:
            continue
        cid = P.COCO_TO_OURS[int(cls)] if coco_mode else int(cls)
        out.append({"class_id": cid, "polygon": poly, "bbox": poly_bbox(poly),
                    "conf": round(float(score), 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只看前 N 张验证图（0=全部）")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="两个模型共用的置信度阈值（必须一致才算公平对比）")
    ap.add_argument("--out", default="round2_report.json", help="结果文件名（记录在 records/ 下）")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "classes.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]

    best = MODEL_DIR / "best.pt"
    if not best.exists():
        raise SystemExit("× 找不到 model/best.pt，先跑 python train.py")

    val_dir = DATASET / "seg" / "images" / "val"
    imgs = sorted([p for p in val_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if a.limit:
        imgs = imgs[:a.limit]

    from ultralytics import YOLO
    m1 = YOLO(str(ROOT / "yolo26n-seg.pt"))
    m2 = YOLO(str(best))

    print("=" * 58)
    print(f"  第二轮回标对比　验证图 {len(imgs)} 张　置信度阈值 {a.conf}")
    print("=" * 58)

    rows = []
    tot1 = {"need_fix": 0, "miss": 0, "wrong_class": 0, "drift": 0, "extra": 0, "gt_objects": 0}
    tot2 = {k: 0 for k in tot1}
    t0 = time.time()

    for k, img in enumerate(imgs, 1):
        ann = ANN / f"{img.stem}.json"
        if not ann.exists():
            continue
        d = json.loads(ann.read_text(encoding="utf-8"))
        gt = d.get("objects", [])
        if not gt:
            continue

        a1 = count_fixes(gt, draft_with(m1, img, coco_mode=True, conf=a.conf))
        a2 = count_fixes(gt, draft_with(m2, img, conf=a.conf))
        for key in tot1:
            tot1[key] += a1[key]
            tot2[key] += a2[key]
        rows.append({"image": img.name, "round1": a1, "round2": a2})
        if k % 5 == 0 or k == len(imgs):
            print(f"  {k}/{len(imgs)} 张　第一轮需修 {tot1['need_fix']} 处　"
                  f"第二轮需修 {tot2['need_fix']} 处", flush=True)

    def drop(a, b):
        return round((a - b) / a * 100, 1) if a else 0.0

    n = len(rows)
    summary = {
        "images": n,
        "conf": a.conf,
        "round1": {**tot1, "per_image": round(tot1["need_fix"] / n, 2) if n else 0},
        "round2": {**tot2, "per_image": round(tot2["need_fix"] / n, 2) if n else 0},
        "improvement_pct": {
            "need_fix": drop(tot1["need_fix"], tot2["need_fix"]),
            "miss": drop(tot1["miss"], tot2["miss"]),
            "wrong_class": drop(tot1["wrong_class"], tot2["wrong_class"]),
            "drift": drop(tot1["drift"], tot2["drift"]),
            "extra": drop(tot1["extra"], tot2["extra"]),
        },
        "seconds": round(time.time() - t0, 1),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report = {"summary": summary, "rows": rows, "names": names}
    RECORDS.mkdir(exist_ok=True)
    (RECORDS / a.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 58)
    print(f"  第一轮：{n} 张图共需人工处理 {tot1['need_fix']} 处"
          f"（漏标 {tot1['miss']} / 认错 {tot1['wrong_class']} / 框歪 {tot1['drift']} / 多标 {tot1['extra']}）")
    print(f"  第二轮：{n} 张图共需人工处理 {tot2['need_fix']} 处"
          f"（漏标 {tot2['miss']} / 认错 {tot2['wrong_class']} / 框歪 {tot2['drift']} / 多标 {tot2['extra']}）")
    print(f"  修正量下降 {summary['improvement_pct']['need_fix']}%")
    print(f"  明细 → {RECORDS / a.out}")
    print("=" * 58)


if __name__ == "__main__":
    main()
