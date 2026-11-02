# -*- coding: utf-8 -*-
"""
train.py —— 用自己标的数据训练模型（任务书第 5 步）

流程：export.py 导出的 dataset/ → 在这里训练 → 产出 model/best.pt
训完的 best.pt 可以直接替换界面的预标注模型，第二次预标注就会比第一次准。

用法（在 Miniforge Prompt 里，先 conda activate imglabel）：
    cd /d D:\imglabel
    python train.py                 # 默认：实例分割，30 轮
    python train.py --epochs 60     # 想练久一点
    python train.py --task det      # 只练检测框（更快，但没有轮廓）

参数说明（都给了保守默认值，CPU 也能跑）：
    --epochs  训练轮数，太少学不会，太多会过拟合。150 张图建议 40~80
    --imgsz   输入尺寸。CPU 上 512 比较快，640 更准但慢一倍
    --batch   一次看几张图。CPU 内存小就用 2~4
    --task    seg = 实例分割（有轮廓）；det = 只检框
    --model   底子模型。yolo26n-seg.pt 是最小的，CPU 友好
"""

import argparse
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
MODEL_DIR = ROOT / "model"


def check_dataset(task):
    """训练前先确认数据齐了，别等跑了十分钟才报错"""
    yml = (DATASET / "seg" / "data.yaml") if task == "seg" else (DATASET / "data.yaml")
    if not yml.exists():
        raise SystemExit(
            f"× 找不到 {yml}\n"
            "  先导出数据集：界面点「导出数据集」，或运行 python export.py")

    img_dir = (DATASET / "seg" / "images" / "train") if task == "seg" else (DATASET / "images" / "train")
    n = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
    if n == 0:
        raise SystemExit(f"× 训练集是空的（{img_dir}）。先在界面上标几张图再导出。")
    if n < 20:
        print(f"! 警告：训练集只有 {n} 张图，训出来的模型基本不能用，只能验证流程。")
        print("  任务书要求 ≥150 张，先攒图再来。\n")
    return yml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--task", choices=["seg", "det"], default="seg")
    ap.add_argument("--model", default=None)
    ap.add_argument("--name", default=None, help="这次训练的名字（默认按时间自动起）")
    ap.add_argument("--lr0", type=float, default=0.0015,
                    help="初始学习率。小数据集必须调小，默认 0.01 会把预训练权重冲坏")
    ap.add_argument("--patience", type=int, default=20, help="多少轮没提升就提前停")
    ap.add_argument("--close-mosaic", type=int, default=8,
                    help="最后几轮关掉马赛克增强，通常能提一点精度")
    ap.add_argument("--no-copy", action="store_true",
                    help="不覆盖 model/best.pt（对比实验时用，免得把好模型换掉）")
    a = ap.parse_args()

    yml = check_dataset(a.task)
    base = a.model or ("yolo26n-seg.pt" if a.task == "seg" else "yolo26n.pt")
    if not (ROOT / base).exists() and not Path(base).exists():
        print(f"! 本地没有 {base}，会自动从网上下载（约 6MB，第一次慢一点）")

    from ultralytics import YOLO

    name = a.name or time.strftime("train_%m%d_%H%M")
    print("=" * 58)
    print(f"  开始训练　任务={a.task}　轮数={a.epochs}　尺寸={a.imgsz}　批量={a.batch}")
    print(f"  数据：{yml}")
    print(f"  底子：{base}　CPU 训练，中途可以关掉窗口停（已训的轮次会保留）")
    print("=" * 58)

    t0 = time.time()
    model = YOLO(base)
    results = model.train(
        data=str(yml),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device="cpu",          # 这台电脑没独显，显式写死免得它去试 CUDA
        workers=0,             # Windows 上多进程容易卡死，设 0 最稳
        project=str(ROOT / "runs"),
        name=name,
        exist_ok=True,
        lr0=a.lr0,
        lrf=0.01,              # 收尾学习率 = lr0 * 0.01，让最后几轮稳定收敛
        cos_lr=True,           # 余弦退火，比阶梯下降更平滑
        patience=a.patience,
        close_mosaic=a.close_mosaic,
        plots=True,
        verbose=True,
    )

    # 训练产物统一收到 model/ 下，任务书要求 model/best.pt
    best_src = Path(model.trainer.best)
    best_dst = MODEL_DIR / "best.pt"
    if a.no_copy:
        print(f"  （--no-copy）本次权重留在 {best_src}，没有覆盖 model/best.pt")
    elif best_src.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        shutil.copyfile(best_src, best_dst)
        shutil.copyfile(model.trainer.last, MODEL_DIR / "last.pt")

    print("=" * 58)
    print(f"  训练结束　耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"  最好权重：{best_dst}")
    print(f"  全部产物：{ROOT / 'runs' / name}")
    try:
        m = results.results_dict
        for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                  "metrics/mAP50(M)", "metrics/mAP50-95(M)"):
            if k in m:
                print(f"  {k:<24} {m[k]:.4f}")
    except Exception:
        pass
    print("-" * 58)
    print("  下一步：界面里点「预标注全部」，这次用的就是新模型（第二轮）")
    print("=" * 58)


if __name__ == "__main__":
    main()
