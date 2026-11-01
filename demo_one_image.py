# -*- coding: utf-8 -*-
"""
demo_one_image.py —— 第一个"能跑通"的验证脚本

用法（在 Miniforge Prompt 里，先 conda activate imglabel，再 cd 到本目录）：
    python demo_one_image.py                 # 自动用 data/raw 里的第一张图
    python demo_one_image.py 我的照片.jpg     # 指定一张图

它会做三件事：
    1. 用 yolo26n-seg 给图片画轮廓，框由轮廓自动算出来
    2. （可选）用 YOLO-World 按十类词表直接找物体
    3. 把画好标注的图存到 data/preview/，并在屏幕上报告结果

这一步跑通 = 整个项目最大的不确定性解决了，后面都是写界面和接口。
"""

import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

import prelabel as P

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PREVIEW_DIR = ROOT / "data" / "preview"
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"

# 每个类别一个颜色，画图用
PALETTE = [
    (226, 75, 74), (55, 138, 221), (99, 153, 34), (239, 159, 39), (127, 119, 221),
    (29, 158, 117), (212, 83, 126), (186, 117, 23), (136, 135, 128), (12, 68, 124),
]


def pick_image():
    """决定这次用哪张图：命令行指定 > data/raw 里第一张 > 下载示例图"""
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            sys.exit(f"找不到这张图：{p}")
        return p

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in RAW_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if imgs:
        return imgs[0]

    sample = RAW_DIR / "bus.jpg"
    print("data/raw 里还没有图片，先下载一张示例图用来测试……")
    try:
        urllib.request.urlretrieve(SAMPLE_URL, sample)
        print(f"示例图已保存：{sample}")
        return sample
    except Exception as e:
        sys.exit(f"示例图下载失败（{e}）。请自己放一张室内照片到 {RAW_DIR} 再运行。")


def draw(objects, img_path, out_path, with_polygon=True):
    """把标注画到图上：轮廓 + 由轮廓派生的框 + 类别名"""
    img = Image.open(img_path).convert("RGB")
    draw_obj = ImageDraw.Draw(img)
    line_w = max(2, round(min(img.size) / 400))

    for ob in objects:
        color = PALETTE[ob["class_id"] % len(PALETTE)]
        x1, y1, x2, y2 = ob["bbox"]
        draw_obj.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        if with_polygon and ob.get("polygon"):
            pts = [tuple(p) for p in ob["polygon"]]
            if len(pts) >= 3:
                draw_obj.polygon(pts, outline=color)

        label = f'{ob["class_name"]} {ob["conf"]:.2f}'
        ty = max(0, y1 - 14)
        draw_obj.rectangle([x1, ty, x1 + 7 * len(label) + 6, ty + 13], fill=color)
        draw_obj.text((x1 + 3, ty + 1), label, fill=(255, 255, 255))

    img.save(out_path)
    return out_path


def report(title, objects):
    print(f"\n【{title}】共 {len(objects)} 个物体")
    if not objects:
        print("  （这张图里没认出来我们关心的十类物体）")
        return
    counter = {}
    for ob in objects:
        counter[ob["class_name"]] = counter.get(ob["class_name"], 0) + 1
    for name, n in sorted(counter.items(), key=lambda kv: -kv[1]):
        zh = P.ZH[P.NAMES.index(name)]
        print(f"  {name:<14} {zh:<6} × {n}")


def main():
    img_path = pick_image()
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n使用的图片：{img_path}")
    print("=" * 56)

    # ---- 1. 实例分割：轮廓 + 派生框（这是主线模式）----
    t0 = time.time()
    seg_objs = P.prelabel_seg(img_path)
    seg_cost = time.time() - t0
    report("实例分割预标注 yolo26n-seg", seg_objs)
    print(f"  耗时 {seg_cost:.1f} 秒")
    if seg_objs:
        out = PREVIEW_DIR / f"{img_path.stem}_seg.png"
        draw(seg_objs, img_path, out, with_polygon=True)
        print(f"  预览图：{out}")
        print(f"  草稿数据：{P.save_draft(seg_objs, img_path)}")

    # ---- 2. YOLO-World：开放词表检测（可能是第一次下载，会慢一点）----
    print("\n" + "=" * 56)
    try:
        t0 = time.time()
        world_objs = P.prelabel_world(img_path)
        world_cost = time.time() - t0
        report("开放词表检测 YOLO-World", world_objs)
        print(f"  耗时 {world_cost:.1f} 秒")
        if world_objs:
            out = PREVIEW_DIR / f"{img_path.stem}_world.png"
            draw(world_objs, img_path, out, with_polygon=False)
            print(f"  预览图：{out}")
    except Exception as e:
        print("【YOLO-World】这次没能跑起来：")
        print(f"  {type(e).__name__}: {str(e)[:300]}")
        print("  不影响主线——实例分割那条路已经通了，明天再处理它。")

    print("\n" + "=" * 56)
    print("完成。打开 data/preview/ 里的图片看看模型画得准不准。")


if __name__ == "__main__":
    main()
