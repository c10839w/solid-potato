# -*- coding: utf-8 -*-
"""
export.py —— 把标好的数据导出成「能直接拿去训练」的标准数据集

一份标注，导出三套东西（任务书第五章）：
    1. 目标检测   dataset/labels/{split}/*.txt       YOLO 格式：类别 cx cy w h（0~1 的比例）
    2. 实例分割   dataset/masks/{split}/*.png        每个物体一个编号，0 是背景
    3. 语义分割   dataset/semantic/{split}/*.png     每个像素直接写类别号，1~10

同时生成两份训练配置（ultralytics 会按 images/labels 的目录结构自动配对）：
    dataset/data.yaml        检测训练：读 labels/（bbox 标签）
    dataset/seg/data.yaml    分割训练：读 seg/labels/（多边形标签）

还附送：
    dataset/classes.txt              十个英文类名，一行一个
    dataset/instances.json           每张图里"第几个物体是哪个类别"的对照表
    dataset/export_report.json       本次导出的统计
    data/preview/export/             语义图的彩色预览（肉眼看效果用）

用法：
    python export.py                 # 默认 8:2 划分训练/验证
    python export.py --val-ratio 0.15
    python export.py --only-human    # 只导出人工改过的（严格模式，先别用）
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

import prelabel as P

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
ANN = ROOT / "records" / "annotations"
DRAFT = ROOT / "records" / "drafts"
OUT = ROOT / "dataset"
PREVIEW = ROOT / "data" / "preview" / "export"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NC = len(P.NAMES)


# ------------------------------------------------------------------ 读标注
def label_file(name):
    """优先用人工确认过的标注；没有再退回模型草稿"""
    stem = Path(name).stem
    for d in (ANN, DRAFT):
        p = d / (stem + ".json")
        if p.exists():
            return p
    return None


def load_objects(name):
    p = label_file(name)
    if p is None:
        return [], None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return [], None
    objs = []
    for o in d.get("objects", []):
        poly = o.get("polygon") or []
        if len(poly) < 3:                      # 没轮廓，退回用框拼一个矩形
            bb = o.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x1, y1, x2, y2 = bb
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        cid = int(o.get("class_id", -1))
        if not (0 <= cid < NC):
            continue
        objs.append({
            "class_id": cid,
            "polygon": [[float(x), float(y)] for x, y in poly],
            "conf": o.get("conf"),
            "source": o.get("source", "model"),
        })
    return objs, ("human" if any(o["source"] == "human" for o in objs) else "model")


# ------------------------------------------------------------------ 各种换算
def polygon_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def to_det_line(cid, poly, w, h, min_size=2.0):
    """轮廓 -> 一行检测标签（外接矩形），太小的返回 None"""
    x1, y1, x2, y2 = polygon_bbox(poly)
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    bw, bh = x2 - x1, y2 - y1
    if bw < min_size or bh < min_size:         # 太小了，训练时只会变成噪声
        return None
    return f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def to_seg_line(cid, poly, w, h):
    """轮廓 -> 一行分割标签：类别号 + 一串归一化顶点坐标"""
    pts = []
    for x, y in poly:
        pts.append(max(0.0, min(float(x), w)) / w)
        pts.append(max(0.0, min(float(y), h)) / h)
    if len(pts) < 6:                           # 至少 3 个点
        return None
    return f"{cid} " + " ".join(f"{v:.6f}" for v in pts)


def fill_instance(objs, w, h):
    """实例掩码：第 k 个物体涂 k+1（0 留给背景）"""
    m = np.zeros((h, w), np.uint8)
    for k, o in enumerate(objs, 1):
        pts = np.array(o["polygon"], np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], min(k, 255))
    return m


def fill_semantic(objs, w, h):
    """语义掩码：直接涂 类别号+1（0 是背景，1~10 对应十个类）"""
    m = np.zeros((h, w), np.uint8)
    for o in objs:
        pts = np.array(o["polygon"], np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], o["class_id"] + 1)
    return m


def sem_palette():
    """语义图配色，只用于肉眼看效果"""
    pal = np.zeros((NC + 1, 3), np.uint8)
    base = [(220, 60, 60), (60, 160, 220), (80, 200, 120), (230, 160, 40),
            (150, 100, 220), (210, 90, 190), (40, 200, 200), (200, 200, 60),
            (120, 180, 60), (90, 130, 240)]
    for i in range(NC):
        pal[i + 1] = base[i % len(base)]
    return pal


def _fresh_dirs():
    """清掉上一次的产物，避免旧文件混进新数据集"""
    for c in OUT.rglob("*.cache"):      # ultralytics 生成的标签缓存，换了数据必须清掉
        c.unlink(missing_ok=True)
    groups = [
        OUT / "images", OUT / "labels",                 # 检测
        OUT / "seg" / "images", OUT / "seg" / "labels",  # 分割
        OUT / "masks", OUT / "semantic",                # 两种掩码
    ]
    for base in groups:
        for sp in ("train", "val"):
            d = base / sp
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)


def _yaml_text(root, n_val):
    val = "images/val" if n_val else "images/train"
    lines = [
        "# 自动生成，请勿手改（改 export.py 再重新导出）",
        f"path: {root.as_posix()}",
        "train: images/train",
        f"val: {val}",
        "",
        f"nc: {NC}",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(P.NAMES)]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ 主流程
def export(val_ratio=0.2, seed=42, only_human=False, verbose=True, val_list=None):
    """val_list：固定验证集名单（文件名去扩展名）。给了就严格按它划分，
    不再按比例重算 —— 加了新图之后验证集仍是最早那 40 张人工核对过的图，
    前后两次训练的 mAP 才可比。"""
    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    # 1) 收集有内容的图片
    items, skipped_empty, skipped_human = [], [], []
    for p in sorted(RAW.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        objs, kind = load_objects(p.name)
        if not objs:
            skipped_empty.append(p.name)
            continue
        if only_human and kind != "human":
            skipped_human.append(p.name)
            continue
        items.append((p, objs, kind))

    if not items:
        msg = "没有可导出的标注。请先在画布上点「预标注」，或手动标几个物体。"
        if verbose:
            print("×", msg)
        return {"ok": False, "message": msg}

    # 2) 划分训练 / 验证（固定随机种子，每次结果一致，方便复现）
    #    人工修正过的图优先划入验证集：mAP 要对着「人标的答案」算，
    #    否则等于拿模型自己的草稿考模型，数字会虚高。
    rnd = random.Random(seed)
    n_val = 0
    n_val_human = 0
    if val_list:
        wanted = set(val_list)
        val_ids = {i for i, (p, _, _) in enumerate(items) if p.stem in wanted}
        n_val = len(val_ids)
        n_val_human = len({i for i in val_ids if items[i][2] == "human"})
        val_ids = set(sorted(val_ids))
    elif len(items) >= 2:
        n_val = max(1, int(round(len(items) * val_ratio)))
        n_val = min(n_val, len(items) - 1)
        human_idx = [i for i, (_, _, k) in enumerate(items) if k == "human"]
        other_idx = [i for i, (_, _, k) in enumerate(items) if k != "human"]
        rnd.shuffle(human_idx)
        rnd.shuffle(other_idx)
        order = human_idx + other_idx
        val_ids = set(order[:n_val])
        n_val_human = len(set(human_idx) & val_ids)

    _fresh_dirs()

    pal = sem_palette()
    instances, class_count = {}, [0] * NC
    n_train = n_val_actual = 0
    n_obj = n_skip = 0

    for k, (p, objs, kind) in enumerate(items):
        split = "val" if k in val_ids else "train"
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = p.stem

        # 检测标签 + 分割标签
        det_lines, seg_lines, kept = [], [], []
        for o in objs:
            d_line = to_det_line(o["class_id"], o["polygon"], w, h)
            if d_line is None:
                n_skip += 1
                continue
            det_lines.append(d_line)
            s_line = to_seg_line(o["class_id"], o["polygon"], w, h)
            if s_line:
                seg_lines.append(s_line)
            kept.append(o)

        if not kept:
            continue

        # 图片原样复制两份（ultralytics 靠 images/ 与 labels/ 同级来配对）
        for img_dir in (OUT / "images" / split, OUT / "seg" / "images" / split):
            shutil.copyfile(p, img_dir / p.name)

        (OUT / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(det_lines) + "\n", encoding="utf-8")
        (OUT / "seg" / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(seg_lines) + "\n", encoding="utf-8")

        # 实例掩码 + 语义掩码
        cv2.imwrite(str(OUT / "masks" / split / f"{stem}.png"), fill_instance(kept, w, h))
        sem = fill_semantic(kept, w, h)
        cv2.imwrite(str(OUT / "semantic" / split / f"{stem}.png"), sem)

        # 彩色语义图（只是给人看，不参与训练）
        pv = PREVIEW / split
        pv.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(pv / f"{stem}_sem.png"), pal[sem])

        instances[f"{split}/{p.name}"] = [
            {"instance": i, "class_id": o["class_id"], "class_name": P.NAMES[o["class_id"]],
             "source": o["source"]}
            for i, o in enumerate(kept, 1)]
        for o in kept:
            class_count[o["class_id"]] += 1
        n_obj += len(kept)
        if split == "val":
            n_val_actual += 1
        else:
            n_train += 1

    # 3) 两份训练配置
    (OUT / "data.yaml").write_text(_yaml_text(OUT, n_val_actual), encoding="utf-8")
    (OUT / "seg" / "data.yaml").write_text(
        _yaml_text(OUT / "seg", n_val_actual), encoding="utf-8")
    (OUT / "classes.txt").write_text("\n".join(P.NAMES) + "\n", encoding="utf-8")
    (OUT / "instances.json").write_text(
        json.dumps(instances, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "ok": True,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "images_total": n_train + n_val_actual,
        "train": n_train,
        "val": n_val_actual,
        "val_human": n_val_human,
        "objects": n_obj,
        "box_skipped": n_skip,
        "skipped_empty": len(skipped_empty),
        "skipped_not_human": len(skipped_human),
        "per_class": {P.NAMES[i]: c for i, c in enumerate(class_count)},
        "dataset_dir": str(OUT),
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print("=" * 58)
        print(f"  导出完成　用时 {report['seconds']} 秒")
        print(f"  图片 {report['images_total']} 张 → 训练 {n_train} / 验证 {n_val_actual}"
              f"（其中人工修正过 {n_val_human} 张）")
        print(f"  物体 {n_obj} 个，跳过过小框 {n_skip} 个")
        print(f"  空标注跳过 {len(skipped_empty)} 张")
        print("-" * 58)
        for i, n in enumerate(P.NAMES):
            if class_count[i]:
                print(f"    {i:>2}  {n:<14} {class_count[i]:>4} 个")
        missing = [P.NAMES[i] for i, c in enumerate(class_count) if c == 0]
        if missing:
            print(f"  还没标到的类别：{', '.join(missing)}")
        print("-" * 58)
        print(f"  数据集目录：{OUT}")
        print(f"  检测配置  ：{OUT / 'data.yaml'}")
        print(f"  分割配置  ：{OUT / 'seg' / 'data.yaml'}")
        print("=" * 58)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    ap.add_argument("--seed", type=int, default=42, help="划分随机种子，默认 42")
    ap.add_argument("--only-human", action="store_true", help="只导出人工改过的图")
    ap.add_argument("--val-list", default=None,
                    help="固定验证集名单文件（每行一个图片名），加了新图后仍用同一套验证集")
    a = ap.parse_args()
    val_list = None
    if a.val_list:
        vl_path = Path(a.val_list)
        if not vl_path.is_absolute():
            vl_path = ROOT / vl_path
        val_list = [ln.strip() for ln in vl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    export(val_ratio=a.val_ratio, seed=a.seed, only_human=a.only_human, val_list=val_list)
