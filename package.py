# -*- coding: utf-8 -*-
"""
package.py —— 一键打包交付物

任务书要求交付：能跑的源码 + 训练好的 model/best.pt + 修正记录 + 说明文档。
这个脚本把该交的收进一个 zip，不该交的（几百 MB 的图片、可以随时重新生成的掩码）排除掉。

产物：dist/人机协同图像标注系统_交付包.zip

包含：
    app.py / prelabel.py / export.py / train.py / eval_model.py
    eval_baseline_coco.py / round2_compare.py / make_report.py / fetch_coco.py / prelabel_all.py
    canvas/（界面三件套）  启动.bat  requirements.txt  classes.yaml
    README.md  ai_log.md
    model/best.pt（自训权重，任务书硬性要求）
    records/annotations（202 份标注）、fixes.jsonl（修正账本）、
    records/metrics_*.json、round2_report.json、compare.html
    dataset/data.yaml、dataset/seg/data.yaml、dataset/classes.txt、dataset/export_report.json
    dataset/{labels,seg/labels}（YOLO 标签文本，很小）

不包含（体积大、可再生成，README 里写明了怎么重建）：
    data/raw 原始图片、dataset/images、dataset/masks、dataset/semantic、runs/、权重缓存
"""

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "dist"

INCLUDE_FILES = [
    "app.py", "prelabel.py", "prelabel_all.py", "export.py", "train.py",
    "eval_model.py", "eval_baseline_coco.py", "round2_compare.py",
    "make_report.py", "fetch_coco.py", "demo_one_image.py",
    "启动.bat", "requirements.txt", "classes.yaml", "README.md", "ai_log.md",
    "答辩演示.md", "records/val_split.txt", "records/categories.json",
    ".gitignore",
]
INCLUDE_DIRS = [
    ("canvas", None),                       # 界面三件套，全部打包
    ("model", {"best.pt", "last.pt"}),      # 只放权重
    ("records", {"*.json", "*.jsonl", "*.html"}),
    ("dataset", {"*.yaml", "*.txt", "*.json"}),
    ("dataset/labels", None),
    ("dataset/seg/labels", None),
]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    stamp = ""
    rep = ROOT / "dataset" / "export_report.json"
    if rep.exists():
        try:
            stamp = "_" + json.loads(rep.read_text(encoding="utf-8"))["time"][:10]
        except Exception:
            stamp = ""
    out = OUT_DIR / f"人机协同图像标注系统_交付包{stamp}.zip"

    n_files, n_bytes = 0, 0
    seen = set()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        def add(p, arcname):
            nonlocal n_files, n_bytes
            if arcname in seen:          # 两个通配规则都盖到的文件只写一次，避免 zip 里重名
                return
            seen.add(arcname)
            z.write(p, arcname=arcname)
            n_files += 1
            n_bytes += p.stat().st_size

        for name in INCLUDE_FILES:
            p = ROOT / name
            if p.exists():
                add(p, name)

        for dirname, patterns in INCLUDE_DIRS:
            d = ROOT / dirname
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if not p.is_file():
                    continue
                if patterns is not None and not any(p.match(pat) for pat in patterns):
                    continue
                # 掩码/图片/缓存体积大且能重新生成，跳过
                if any(part in {"__pycache__", "runs", "_cache", "_cand", "_cocoval"}
                       for part in p.parts):
                    continue
                add(p, p.relative_to(ROOT).as_posix())

        # 附一份打包说明，评委打开就知道怎么跑
        readme = f"""交付包内容说明
============================================================
源文件 {n_files} 个，原始体积 {n_bytes/1024/1024:.1f} MB

一、怎么运行
    1) 装 Miniforge（或任意 Python 3.10 环境）
    2) conda create -n imglabel python=3.10 -y && conda activate imglabel
    3) pip install -r requirements.txt
    4) 双击 启动.bat，或：python app.py
    5) 浏览器打开 http://127.0.0.1:5000

二、自训权重
    model/best.pt 是用本项目数据集训练得到的（461 张室内图 / 2211 个标注 / 50 轮 / 640 尺寸）
    用第二轮精度对比：python app.py --model model/best.pt
    精度对比报告：records/compare.html

三、数据集没有全被打包（体积原因）
    原始图片 data/raw 与掩码 dataset/masks 体积大，但都能重建：
      python fetch_coco.py            # 从 COCO 按标注重新下载这 461 张室内图
      python export.py                # 重新导出检测标签 / 分割标签 / 两种掩码
    标注本身（records/annotations，449 份 json）已全部打包，里面含人工修正结果。

四、复现全流程
    python fetch_coco.py            # 采集（可选，已经有图可跳过）
    python prelabel_all.py          # 批量预标注，生成草稿
    python app.py                   # 人工修正（本项目改了 67 张图共 687 处）
    python export.py                # 导出数据集
    python eval_baseline_coco.py    # 第一轮基线精度
    python train.py                 # 训练，产出 model/best.pt
    python eval_model.py --model model/best.pt --tag trained
    python round2_compare.py        # 第二轮回标：草稿需修改处数对比（--conf 可扫阈值）
    python make_report.py           # 生成 records/compare.html
    界面截图在 screenshots/，四个页面的样子先看它
"""
        z.writestr("交付说明.txt", readme)
        n_files += 1

    size = out.stat().st_size
    print("=" * 58)
    print(f"  打包完成　{n_files} 个文件")
    print(f"  压缩后 {size/1024/1024:.1f} MB（原始 {n_bytes/1024/1024:.1f} MB）")
    print(f"  → {out}")
    print("=" * 58)


if __name__ == "__main__":
    main()
