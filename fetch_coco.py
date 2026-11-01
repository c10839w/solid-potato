# -*- coding: utf-8 -*-
"""
从 COCO val2017 批量下载「十类室内图」

思路：
  1. 不下载 250MB 的整个标注包，只从 zip 里精确取出 instances_val2017.json 一个成员
     （zip 中央目录 → 定位成员偏移 → 只取那几个字节 → 解压）
  2. 用标注筛选出「确实含有任务书十类、且明显是室内场景」的图片
  3. 多线程下载图片，并按 classes.yaml 的规格压缩（长边 960、JPG 质量 85）
  4. 存进 data/raw/，文件名形如 coco_000000000139.jpg

用法（在 Miniforge Prompt 里，先 conda activate imglabel）：
    cd /d D:\imglabel
    python fetch_coco.py                 # 默认下 200 张左右
    python fetch_coco.py --limit 150     # 只要 150 张
    python fetch_coco.py --dry-run       # 只筛选不下载，看看能选出什么
"""

import argparse
import io
import json
import ssl
import struct
import sys
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image

RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "_cache"

ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ZIP_MEMBER = "annotations/instances_val2017.json"
IMG_URL = "http://images.cocodataset.org/val2017/{name}"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
SSL_CTX = ssl._create_unverified_context()

# COCO 标注文件里的 category_id → 我们的类别号（classes.yaml 里的顺序 0~9）
# 注意：这和 classes.yaml 里的 coco_id 不是一回事
#       classes.yaml 的 coco_id 是 ultralytics 内部重新编号过的 0~79
#       这里用的是 COCO 官方 json 里的原始 id（有跳号，chair=62 而不是 56）
COCO_CAT_TO_OURS = {
    1: 0,    # person        人
    62: 1,   # chair         椅子
    67: 2,   # dining table  桌子
    63: 3,   # couch         沙发
    72: 4,   # tv            电视
    73: 5,   # laptop        笔记本电脑
    47: 6,   # cup           杯子
    44: 7,   # bottle        瓶子
    64: 8,   # potted plant  盆栽
    65: 9,   # bed           床
}

# 家具类：只有这些"看得清"才算合格的室内照
FURNITURE = {1, 2, 3, 4, 8, 9}   # 椅子 / 桌子 / 沙发 / 电视 / 盆栽 / 床
FOOD_IDS = {52, 53, 54, 55, 56, 57, 58, 59, 60, 61}          # 水果汉堡披萨甜甜圈蛋糕
WARE_IDS = {46, 48, 49, 50, 51}                              # 酒杯刀叉碗
OUTDOOR_IDS = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14}       # 车船飞机红绿灯
MIN_FURN_AREA = 0.040            # 家具至少占画面 4%，否则学不到形状

# 每个类别至少要有多少张（任务书：≥6 类，每类 ≥15 张）
MIN_PER_CLASS = 15
TARGET_PER_CLASS = 55


# ----------------------------------------------------------------- 取标注文件
def _range_get(url, start, end, timeout=90, tries=6):
    """取一段字节。被临时拒绝时退避重试。"""
    last = None
    for att in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
                return r.read()
        except Exception as e:
            last = e
            if att == tries - 1:
                break
            time.sleep(3 * (att + 1))
    raise last


def _range_get_multi(url, start, end, workers=10, chunk=512 * 1024, tries=6):
    """大块数据分片并发取回再拼起来。
    COCO 服务器对单条连接限速约 12KB/s，多条叠加才快；
    但并发太多会被临时拒绝(403)，所以并发压在 10 以内并做退避重试。"""
    total = end - start + 1
    if total <= chunk:
        return _range_get(url, start, end, tries=tries)
    spans = [(s, min(s + chunk - 1, end)) for s in range(start, end + 1, chunk)]
    parts = [None] * len(spans)

    def work(i, a, b):
        for att in range(tries):
            try:
                parts[i] = _range_get(url, a, b)
                return
            except Exception:
                if att == tries - 1:
                    raise
                time.sleep(3 * (att + 1))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in [ex.submit(work, i, a, b) for i, (a, b) in enumerate(spans)]:
            f.result()
    return b"".join(parts)


def _fetch_zip_member(url, member, out_path):
    """只从远程 zip 中取一个成员，不下载整个包"""
    # ① 一次请求同时拿到「文件总长」和「尾部 128KB」（少一次往返，也少被拒的机会）
    tail, total = None, None
    for att in range(6):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": "bytes=-131072"})
            with urllib.request.urlopen(req, timeout=90, context=SSL_CTX) as r:
                tail = r.read()
                cr = r.headers.get("Content-Range", "")
            total = int(cr.rsplit("/", 1)[1])
            break
        except Exception as e:
            if att == 5:
                raise
            print(f"  连接被拒（{type(e).__name__}），{3*(att+1)} 秒后重试...", flush=True)
            time.sleep(3 * (att + 1))
    print(f"  远程包大小 {total/1024/1024:.1f} MB", flush=True)

    # ② 在尾部找 EOCD（zip 结尾中央目录记录）
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0:
        raise RuntimeError("没找到 zip 的中央目录记录")
    (_sig, _d1, _d2, _n1, _n, cd_size, cd_off, _clen) = struct.unpack("<4s4H2LH", tail[pos:pos + 22])
    print(f"  中央目录位于 {cd_off} 字节，大小 {cd_size} 字节", flush=True)

    # ③ 取中央目录，逐条找目标成员
    cd = _range_get_multi(url, cd_off, cd_off + cd_size - 1)
    i, hit = 0, None
    while i < len(cd) - 4 and cd[i:i + 4] == b"PK\x01\x02":
        f = struct.unpack("<4s6H3L5H2L", cd[i:i + 46])
        method, csize, usize = f[4], f[8], f[9]
        fnlen, extralen, clen = f[10], f[11], f[12]
        local_off = f[16]
        name = cd[i + 46:i + 46 + fnlen].decode("utf-8", "ignore")
        if name == member:
            hit = (method, csize, usize, local_off)
            break
        i += 46 + fnlen + extralen + clen
    if not hit:
        raise RuntimeError(f"包里没有 {member}")
    method, csize, usize, local_off = hit
    print(f"  命中 {member}：压缩 {csize/1024/1024:.2f} MB → 解开 {usize/1024/1024:.1f} MB", flush=True)

    # ④ 读本地头，算出数据真正的起点
    head = _range_get(url, local_off, local_off + 29)
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError("本地文件头校验失败")
    fnlen, extralen = struct.unpack("<H", head[26:28])[0], struct.unpack("<H", head[28:30])[0]
    data_start = local_off + 30 + fnlen + extralen

    # ⑤ 只取成员数据并解压
    raw = _range_get_multi(url, data_start, data_start + csize - 1, workers=24)
    blob = zlib.decompress(raw, -15) if method == 8 else raw
    if len(blob) != usize:
        raise RuntimeError(f"解压后大小不符：{len(blob)} != {usize}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    return out_path


def ensure_annotations():
    p = CACHE / "instances_val2017.json"
    if p.exists() and p.stat().st_size > 5 * 1024 * 1024:
        print(f"标注文件已存在，跳过下载：{p}", flush=True)
        return p
    print("从 COCO 服务器按字节范围提取标注（只取需要的那一块）...", flush=True)
    t = time.time()
    _fetch_zip_member(ZIP_URL, ZIP_MEMBER, p)
    print(f"完成，耗时 {time.time()-t:.0f} 秒\n", flush=True)
    return p


# ----------------------------------------------------------------- 筛选图片
def pick_images(ann_path, limit, seed=42):
    names = yaml.safe_load((ROOT / "classes.yaml").read_text(encoding="utf-8"))["names"]
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    imgs = {im["id"]: im for im in data["images"]}
    per_img = {}          # image_id -> {cls:{类别:面积占比}, food, outdoor, ware}
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        im = imgs.get(a["image_id"])
        if not im:
            continue
        cid = a["category_id"]
        area = (a["bbox"][2] * a["bbox"][3]) / float(im["width"] * im["height"])
        rec = per_img.setdefault(a["image_id"],
                                 {"cls": {}, "food": 0.0, "outdoor": 0, "ware": 0})
        c = COCO_CAT_TO_OURS.get(cid)
        if c is not None:
            if 0.006 <= area <= 0.92:        # 太小看不清、太大等于整张图
                rec["cls"][c] = max(rec["cls"].get(c, 0.0), area)
        elif cid in FOOD_IDS:                 # 画面被食物占领 → 食物特写，不要
            rec["food"] += area
        elif cid in OUTDOOR_IDS:              # 有车、红绿灯 → 街景
            rec["outdoor"] += 1
        elif cid in WARE_IDS:                 # 碗盘刀叉 → 餐桌特写
            rec["ware"] += 1

    # 打分：要「房间照片」——家具看得清，而且不能被食物/餐具占满
    cands = []
    for iid, rec in per_img.items():
        cls_area = rec["cls"]
        classes = set(cls_area)
        furn = classes & FURNITURE
        if not furn:                          # 一件家具都没有 → 不要
            continue
        fmax = max(cls_area[c] for c in furn)
        if fmax < MIN_FURN_AREA:              # 家具只是画面里一个小角落 → 学不到形状
            continue
        if rec["food"] >= 0.08:               # 食物占画面 8% 以上 → 餐桌特写
            continue
        if rec["ware"] >= 3 and fmax < 0.20:  # 一堆杯盘且看不到大件家具
            continue
        if rec["outdoor"] >= 2:               # 街景
            continue
        score = (fmax * 3.0                        # 家具越大越好
                 + len(furn) * 2.0                 # 家具种类越多越好
                 + (2.5 if classes & {3, 9} else 0)     # 有沙发/床 = 客厅卧室
                 + (1.5 if 4 in classes else 0)         # 有电视
                 + len(classes) * 0.6
                 - rec["food"] * 15 - rec["outdoor"] * 0.8)
        cands.append({"id": iid, "name": imgs[iid]["file_name"], "w": imgs[iid]["width"],
                      "h": imgs[iid]["height"], "classes": sorted(classes),
                      "fmax": round(fmax, 3), "score": round(score, 2)})
    cands.sort(key=lambda x: -x["score"])

    # 贪心挑选：优先补「还缺的类别」，同类信息重复的往后排
    need = {i: TARGET_PER_CLASS for i in range(len(names))}
    chosen, seen_sets = [], {}
    for c in cands:
        if len(chosen) >= limit:
            break
        useful = [k for k in c["classes"] if need[k] > 0]
        if not useful:
            continue
        key = tuple(c["classes"])
        if seen_sets.get(key, 0) >= 45:       # 类别组合完全相同的图，最多留 25 张
            continue
        chosen.append(c)
        seen_sets[key] = seen_sets.get(key, 0) + 1
        for k in c["classes"]:
            need[k] -= 1

    counts = {i: 0 for i in range(len(names))}
    for c in chosen:
        for k in c["classes"]:
            counts[k] += 1
    return chosen, counts, names


# ----------------------------------------------------------------- 下载图片
def download_one(item):
    info = item["item"]
    out = RAW / ("coco_" + info["name"])
    if out.exists() and out.stat().st_size > 4096:
        return "skip", out.name
    for attempt in range(3):
        try:
            req = urllib.request.Request(IMG_URL.format(name=info["name"]), headers=UA)
            with urllib.request.urlopen(req, timeout=45, context=SSL_CTX) as r:
                blob = r.read()
            im = Image.open(io.BytesIO(blob))
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > 960:                       # 任务书：长边 960
                s = 960 / max(w, h)
                im = im.resize((max(1, int(w * s)), max(1, int(h * s))), RESAMPLE)
            tmp = out.with_suffix(".tmp")
            im.save(tmp, "JPEG", quality=85, optimize=True)   # 任务书：质量 85
            tmp.replace(out)
            return "ok", out.name
        except Exception as e:
            if attempt == 2:
                return "fail:" + type(e).__name__, info["name"]
            time.sleep(1 + attempt)
    return "fail", info["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="最多下载多少张")
    ap.add_argument("--workers", type=int, default=24, help="并发线程数")
    ap.add_argument("--dry-run", action="store_true", help="只筛选，不下载")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    ann = ensure_annotations()
    print("正在用标注筛选「含十类物体 + 室内场景」的图片...", flush=True)
    chosen, counts, names = pick_images(ann, args.limit)

    print(f"\n筛选出 {len(chosen)} 张，各类覆盖情况：")
    for i, n in enumerate(names):
        flag = "OK " if counts[i] >= MIN_PER_CLASS else ("少 " if counts[i] else "缺 ")
        print(f"  {flag}{i} {n:<14} {counts[i]:>3} 张")
    covered = sum(1 for i in range(len(names)) if counts[i] >= MIN_PER_CLASS)
    print(f"达标类别（≥{MIN_PER_CLASS} 张）：{covered}/10")

    if args.dry_run:
        print("\n只筛选模式，没有下载。前 10 张示例：")
        for c in chosen[:10]:
            print("  ", c["name"], [names[k] for k in c["classes"]])
        return

    print(f"\n开始下载（{args.workers} 线程）...", flush=True)
    t = time.time()
    ok = fail = skip = 0
    tasks = [{"item": c} for c in chosen]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (status, name) in enumerate(ex.map(download_one, tasks), 1):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"   下载失败 {name} ({status})", flush=True)
            if i % 20 == 0 or i == len(tasks):
                print(f"  进度 {i}/{len(tasks)}　成功 {ok}　跳过 {skip}　失败 {fail}"
                      f"　用时 {time.time()-t:.0f}s", flush=True)

    total = len(list(RAW.glob("*.[jJ][pP][gG]")))
    print(f"\n完成：新增 {ok} 张，跳过 {skip} 张，失败 {fail} 张")
    print(f"data\\raw 现有图片 {total} 张，耗时 {time.time()-t:.0f} 秒")
    print("\n下一步：回到画布点「预标注全部」，或运行  python prelabel_all.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断")
    except Exception as e:
        print(f"\n[出错] {type(e).__name__}: {e}")
        sys.exit(1)
