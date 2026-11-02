# -*- coding: utf-8 -*-
"""
prelabel.py —— 模型预标注（自动画草稿）

这是整个系统里唯一"调用 AI 模型"的地方。它做一件事：
    给一张图片 -> 返回一串物体，每个物体带 类别 + 轮廓(多边形) + 外接框

关键设计（任务书第五章）：以轮廓为准，框由轮廓自动算出来，
所以这里绝不单独让模型去"检测框"，框永远是轮廓的外接矩形。
"""

from pathlib import Path
import json

import yaml

ROOT = Path(__file__).parent


def load_config():
    with open(ROOT / "classes.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()
NAMES = CFG["names"]                              # ['person', 'chair', ...]
ZH = CFG["zh"]                                    # ['人', '椅子', ...]
COCO_IDS = [int(c) for c in CFG["coco_id"]]        # [0, 56, 60, ...]
COCO_TO_OURS = {c: i for i, c in enumerate(COCO_IDS)}   # COCO编号 -> 我们的0~9


def _bbox_from_polygon(poly):
    """轮廓 -> 外接矩形 [x1, y1, x2, y2]（像素坐标）"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


_MODELS = {}


def load_model(model_path="yolo26n-seg.pt"):
    """加载模型并缓存。加载一次约 5~10 秒，批量处理时必须复用同一个。"""
    from ultralytics import YOLO

    key = str(model_path)
    if key not in _MODELS:
        _MODELS[key] = YOLO(key)
    return _MODELS[key]


def model_classes(model):
    """
    判断这个权重"说的是哪套编号"。

    两种权重差别很大：
      yolo26n-seg.pt   -> 80 个 COCO 类别，编号 0~79（椅子是 56，不是 1），要翻译
      model/best.pt    -> 我们自己训练的 10 类，编号 0~9 就是 classes.yaml 的顺序，不用翻译
    返回 (喂给 predict 的 classes 列表, 是否还需要 COCO 翻译)
    """
    names = model.names
    vals = [str(names[i]) for i in sorted(names)]
    if vals[:len(NAMES)] == NAMES:            # 自训权重：前十个名字和我们完全一致
        return list(range(len(NAMES))), False
    return COCO_IDS, True


def prelabel_seg(img_path, model_path="yolo26n-seg.pt", conf=0.25):
    """
    用实例分割模型画草稿：得到「轮廓」，框再由轮廓派生。
    返回 [{'class_id': 0, 'polygon': [[x,y],...], 'bbox': [x1,y1,x2,y2], 'conf': 0.93}, ...]
    """
    model = load_model(model_path)
    classes, need_map = model_classes(model)
    result = model.predict(
        str(img_path),
        conf=conf,
        classes=classes,      # 通用权重：只保留我们关心的十类，其余 70 类直接丢掉
        verbose=False,
    )[0]

    return objects_from_result(result, need_map)


def objects_from_result(result, need_map=True):
    """把 ultralytics 的预测结果转成我们的物体列表（界面和脚本共用这一份逻辑）"""
    objs = []
    if result.masks is None:          # 一张图里一个目标都没认出来
        return objs

    for i, (cls, score) in enumerate(zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())):
        pts = _mask_to_polygon(result.masks.data[i], result.orig_shape)
        if pts is None:
            continue
        idx = COCO_TO_OURS[int(cls)] if need_map else int(cls)
        if idx >= len(NAMES):         # 兜底：模型吐出没见过的类别
            continue
        objs.append({
            "class_id": idx,          # 统一成我们的 0~9
            "class_name": NAMES[idx],
            "polygon": pts,
            "bbox": _bbox_from_polygon(pts),
            "conf": round(float(score), 3),
        })
    return objs


def _mask_to_polygon(mask, orig_shape=None, min_area=40, eps_ratio=0.004):
    """
    把模型输出的一张掩码变成"一条干净的外轮廓"，坐标是**原图像素**。

    两个坑，这里都处理掉了：

    坑一：坐标偏移。模型吐出来的掩码是"补过边的模型输入尺寸"（比如原图 640×426，
      掩码却是 640×448，上下各补了 11 像素黑边）。直接取轮廓当原图坐标，
      框就会整体往右下偏 11 像素。所以先借 ultralytics 自己的 scale_masks
      把掩码缩回原图大小（它会自动扣掉补边、按比例还原），再取轮廓。

    坑二：掩码里有洞。物体被别的东西挡住时（比如猫坐在沙发上），掩码会留一个洞。
      ultralytics 的 masks.xy 会把外轮廓和洞拼成一条线，画出来是"框绕一圈、
      又拐进去绕着猫"，看着像框住了一只猫，其实是坏数据。
      这里只取面积最大的外轮廓，洞直接丢掉 —— 一条轮廓 = 一个物体。
    """
    import numpy as np
    import cv2

    if orig_shape is not None and tuple(mask.shape[-2:]) != tuple(orig_shape):
        from ultralytics.utils import ops

        # mask: (H,W) -> (1,1,H,W) 才能进 scale_masks；它会缩到原图并扣掉补边
        mask = ops.scale_masks(mask[None, None].float(), orig_shape)[0, 0]

    m = (mask.cpu().numpy() > 0.5).astype(np.uint8)
    if m.sum() == 0:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    # 点太多界面会卡，按周长比例抽稀，形状基本不变
    c = cv2.approxPolyDP(c, eps_ratio * cv2.arcLength(c, True), True)
    pts = [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in c]
    return pts if len(pts) >= 3 else None


def prelabel_world(img_path, model_path="yolov8s-worldv2.pt", conf=0.25):
    """
    用 YOLO-World（开放词表）画检测草稿：直接告诉它"找这十个东西"，不用训练。
    返回 [] 表示没认出来；抛异常表示模型本身用不了（例如下载失败）。
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model.set_classes(NAMES)          # 这里直接把我们的十类英文名喂给模型
    result = model.predict(str(img_path), conf=conf, verbose=False)[0]

    objs = []
    for box, cls, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        idx = int(cls)
        if idx >= len(NAMES):         # 兜底：模型吐出没见过的类别
            continue
        objs.append({
            "class_id": idx,
            "class_name": NAMES[idx],
            "bbox": [round(v, 1) for v in box],
            "conf": round(float(score), 3),
        })
    return objs


def save_draft(objs, img_path, out_dir=None):
    """把草稿存成 json，方便下一步界面读取"""
    out_dir = Path(out_dir or (ROOT / "records" / "drafts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(img_path).stem + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"image": Path(img_path).name, "objects": objs}, f, ensure_ascii=False, indent=2)
    return out
