# -*- coding: utf-8 -*-
"""
批量预标注：把 data/raw 里的图一次性跑一遍模型，每张生成一份草稿

和画布上的「预标注全部」按钮是同一件事，只是不开网页也能跑。

用法（Miniforge Prompt，先 conda activate imglabel）：
    cd /d D:\imglabel
    python prelabel_all.py                  # 只处理还没有人工改过的图
    python prelabel_all.py --force          # 全部重新跑（会覆盖草稿）
    python prelabel_all.py --limit 20       # 只跑前 20 张，先试试
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

import prelabel as P

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
ANN = ROOT / "records" / "annotations"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已有的图也重新跑")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全部）")
    args = ap.parse_args()

    ANN.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in sorted(RAW.iterdir()) if p.suffix.lower() in IMG_EXTS]

    todo = []
    for p in imgs:
        dst = ANN / (p.stem + ".json")
        if dst.exists() and not args.force:
            try:
                d = json.loads(dst.read_text(encoding="utf-8"))
                if any(o.get("source") == "human" for o in d.get("objects", [])):
                    continue          # 人工改过了，千万别覆盖
            except Exception:
                pass
            if not args.force:
                continue              # 已经有草稿了，跳过
        todo.append(p)
    if args.limit:
        todo = todo[:args.limit]

    print(f"共 {len(imgs)} 张图，本次要预标注 {len(todo)} 张")
    if not todo:
        print("没有需要处理的图。")
        return

    t0 = time.time()
    model = P.load_model()            # 只加载一次，之后复用
    classes, need_map = P.model_classes(model)   # 自训权重不做 COCO 编号翻译
    total_objs = 0
    for i, p in enumerate(todo, 1):
        t = time.time()
        try:
            with Image.open(p) as im:
                w, h = im.size
            r = model.predict(str(p), conf=0.25, classes=classes, verbose=False)[0]
            objs = [{
                "id": f"o{i}_{k}",
                "class_id": o["class_id"],
                "polygon": o["polygon"],
                "conf": o["conf"],
                "source": "model",
            } for k, o in enumerate(P.objects_from_result(r, need_map))]
            data = {"image": p.name, "width": w, "height": h, "objects": objs}
            (ANN / (p.stem + ".json")).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            total_objs += len(objs)
            print(f"  [{i}/{len(todo)}] {p.name}  {len(objs)} 个物体  {time.time()-t:.1f}s", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {p.name}  失败：{type(e).__name__} {e}", flush=True)

    dt = time.time() - t0
    print(f"\n完成 {len(todo)} 张，共标注 {total_objs} 个物体，用时 {dt/60:.1f} 分钟")
    print("打开画布就能逐张修了：双击 启动.bat")


if __name__ == "__main__":
    main()
