# -*- coding: utf-8 -*-
"""
app.py —— 人机协同图像标注系统的服务端

启动方式（双击 启动.bat，或在 Miniforge Prompt 里）：
    cd /d D:\imglabel
    python app.py

浏览器会自动打开 http://127.0.0.1:5000
"""

import json
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

import prelabel as P

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PREVIEW_DIR = ROOT / "data" / "preview"
ANN_DIR = ROOT / "records" / "annotations"
FIX_LOG = ROOT / "records" / "fixes.jsonl"
REPORT = ROOT / "records" / "compare.html"

for d in (RAW_DIR, PREVIEW_DIR, ANN_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(ROOT / "canvas"), static_url_path="/static")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 预标注用哪个模型：默认用通用的 yolo26n-seg；
# 自己训练出 model/best.pt 之后，用 python app.py --model model/best.pt 切换（第二轮）
MODEL_PATH = ROOT / "yolo26n-seg.pt"

# ---------------------------------------------------------------- 模型（全局只加载一次）
_model = None
_model_lock = threading.Lock()


def get_model():
    """第一次调用时加载模型，之后复用。加载一次约 10 秒，不能每张图都重载。"""
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            _model = YOLO(str(MODEL_PATH))
        return _model


# ---------------------------------------------------------------- 后台任务状态
PROGRESS = {"running": False, "total": 0, "done": 0, "current": "", "message": ""}
_progress_lock = threading.Lock()


def set_progress(**kw):
    with _progress_lock:
        PROGRESS.update(kw)


# ---------------------------------------------------------------- 文件工具
def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def ann_path(name):
    return ANN_DIR / (Path(name).stem + ".json")


def load_annotation(name):
    """读取已有标注；没有就新建一个空白标注"""
    p = ann_path(name)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    img = RAW_DIR / name
    w, h = image_size(img)

    # 兼容：demo_one_image.py 生成的草稿（records/drafts）可以直接拿来用
    draft = ROOT / "records" / "drafts" / (Path(name).stem + ".json")
    if draft.exists():
        try:
            with open(draft, encoding="utf-8") as f:
                d = json.load(f)
            objs = [{
                "id": f"d{k}",
                "class_id": o["class_id"],
                "polygon": o["polygon"],
                "conf": o.get("conf", 1),
                "source": "model",
            } for k, o in enumerate(d.get("objects", []))]
            return {"image": name, "width": w, "height": h, "objects": objs}
        except Exception:
            pass

    return {"image": name, "width": w, "height": h, "objects": []}


def save_annotation(data):
    p = ann_path(data["image"])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p


def log_fixes(fixes):
    """修正记录：一条一行，追加写入，永不覆盖"""
    if not fixes:
        return
    FIX_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FIX_LOG, "a", encoding="utf-8") as f:
        for item in fixes:
            item["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def count_fixes():
    if not FIX_LOG.exists():
        return 0
    with open(FIX_LOG, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------- 照片类别（客厅/卧室/餐厅/书房/其他）
CAT_FILE = ROOT / "records" / "categories.json"
CAT_ZH = ["客厅", "卧室", "餐厅", "书房", "其他"]
_cats = None


def _guess_category(objs):
    """没人工指定时，按图里有什么物体猜一个类别（只用于照片墙分类，不影响标注）"""
    ids = {o.get("class_id") for o in objs}
    if 9 in ids:                 # bed
        return "卧室"
    if 2 in ids:                 # dining table
        return "餐厅"
    if 5 in ids and 1 in ids:    # laptop + chair
        return "书房"
    if 3 in ids or 4 in ids:     # couch / tv
        return "客厅"
    return "其他"


def load_categories():
    """文件名 -> 类别。第一次运行时按标注内容猜一遍，之后以 json 为准。"""
    global _cats
    if _cats is not None:
        return _cats
    if CAT_FILE.exists():
        try:
            _cats = json.loads(CAT_FILE.read_text(encoding="utf-8"))
            return _cats
        except Exception:
            pass
    _cats = {}
    for p in ANN_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        _cats[d.get("image") or (p.stem + ".jpg")] = _guess_category(d.get("objects", []))
    save_categories()
    return _cats


def save_categories():
    CAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAT_FILE.write_text(json.dumps(_cats or {}, ensure_ascii=False, indent=2), encoding="utf-8")


def set_category(name, cat):
    load_categories()
    if cat in CAT_ZH:
        _cats[name] = cat
        save_categories()
    return _cats.get(name, "其他")


# ---------------------------------------------------------------- 把标注画到照片上
# 和画布 app.js 里的 PALETTE 保持一致，导出的图和界面看起来是一套
PALETTE = [(226, 75, 74), (55, 138, 221), (99, 153, 34), (239, 159, 39), (127, 119, 221),
           (29, 158, 117), (212, 83, 126), (186, 117, 23), (136, 135, 128), (12, 68, 124)]

_FONTS = {}


def _font(size):
    from PIL import ImageFont
    if size in _FONTS:
        return _FONTS[size]
    ft = None
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf"):
        try:
            ft = ImageFont.truetype(name, size)
            break
        except Exception:
            continue
    _FONTS[size] = ft or ImageFont.load_default()
    return _FONTS[size]


def render_annotated(img_path, ann, max_w=None):
    """把多边形轮廓 + 类别标签画到照片上，返回 JPEG 字节（不落盘）"""
    import io
    from PIL import Image, ImageDraw

    im = Image.open(img_path).convert("RGB")
    if max_w and im.width > max_w:
        im = im.resize((max_w, max(1, round(im.height * max_w / im.width))), Image.LANCZOS)

    base_w = ann.get("width") or im.width
    s = im.width / float(base_w)

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    fs = max(11, round(13 * min(1.0, s) * 1.2))
    font = _font(fs)

    for ob in ann.get("objects", []):
        poly = ob.get("polygon") or []
        if len(poly) < 3:
            continue
        cid = int(ob.get("class_id", 0)) % len(PALETTE)
        r, g, b = PALETTE[cid]
        pts = [(p[0] * s, p[1] * s) for p in poly]
        d.polygon(pts, fill=(r, g, b, 44))
        d.line(pts + [pts[0]], fill=(r, g, b, 255), width=max(2, round(2 * max(s, .6))), joint="curve")

        # 类别标签（贴在框的左上角）
        label = P.ZH[cid] if cid < len(P.ZH) else str(cid)
        x1 = min(p[0] for p in pts)
        y1 = min(p[1] for p in pts)
        try:
            tw = d.textlength(label, font=font)
        except Exception:
            tw = fs * len(label)
        th = fs + 6
        ty = max(0, y1 - th)
        d.rectangle([x1, ty, x1 + tw + 10, ty + th], fill=(r, g, b, 235))
        d.text((x1 + 5, ty + 2), label, font=font, fill=(255, 255, 255, 255))

    out = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=86)
    return buf.getvalue()


# ---------------------------------------------------------------- 页面
@app.route("/")
def index():
    """封面页：介绍 + 上传入口"""
    return send_from_directory(app.static_folder, "cover.html")


@app.route("/workbench")
def workbench():
    """标注工作台（原来的单页界面）"""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/gallery")
def gallery():
    """照片墙：看所有标过的照片，可导出"""
    return send_from_directory(app.static_folder, "gallery.html")


@app.route("/api/config")
def api_config():
    return jsonify({"names": P.NAMES, "zh": P.ZH})


def count_objects(name):
    """统计一张图已有多少物体 / 其中人工改过多少（含尚未保存的草稿）"""
    for p in (ann_path(name), ROOT / "records" / "drafts" / (Path(name).stem + ".json")):
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    objs = json.load(f).get("objects", [])
                return len(objs), sum(1 for o in objs if o.get("source") == "human")
            except Exception:
                continue
    return 0, 0


@app.route("/api/images")
def api_images():
    """图片列表 + 每张的标注状态 + 照片类别"""
    cats = load_categories()
    items = []
    for p in sorted(RAW_DIR.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        n_obj, n_human = count_objects(p.name)
        items.append({"name": p.name, "objects": n_obj, "human": n_human,
                      "category": cats.get(p.name, "其他")})
    return jsonify({"images": items, "fixes": count_fixes()})


@app.route("/api/stats")
def api_stats():
    """封面页用的一排数字"""
    cats = load_categories()
    n_img = n_obj = n_human_img = 0
    for p in sorted(RAW_DIR.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        n_img += 1
        o, h = count_objects(p.name)
        n_obj += o
        if h > 0:
            n_human_img += 1
    return jsonify({"images": n_img, "objects": n_obj,
                    "human_images": n_human_img, "fixes": count_fixes(),
                    "categories": cats and len(cats) or 0})


@app.route("/api/category", methods=["POST"])
def api_category():
    """改某张照片的类别（工作台右上角那个下拉框）"""
    body = request.get_json(force=True) or {}
    name, cat = body.get("name"), body.get("category")
    if not name or not cat:
        return jsonify({"ok": False, "message": "缺少参数"}), 400
    return jsonify({"ok": True, "category": set_category(name, cat)})


@app.route("/api/img/<path:name>")
def api_img(name):
    return send_from_directory(RAW_DIR, name)


@app.route("/api/annotation/<path:name>")
def api_get_ann(name):
    d = load_annotation(name)
    d["category"] = load_categories().get(name, "其他")
    return jsonify(d)


@app.route("/api/annotation/<path:name>", methods=["POST"])
def api_post_ann(name):
    data = request.get_json(force=True)
    data["image"] = name
    fixes = data.pop("_fixes", [])      # 先把修正记录摘出来，别写进标注文件
    cat = data.pop("category", None)    # 类别单独存，不混进标注文件
    save_annotation(data)
    if cat:
        set_category(name, cat)
    log_fixes(fixes)
    return jsonify({"ok": True, "fixes": count_fixes()})


# ---------------------------------------------------------------- 预标注（后台异步 + 进度）
def _prelabel_worker(names):
    try:
        model = get_model()
        classes, need_map = P.model_classes(model)   # 自训权重不用做 COCO 编号翻译
        set_progress(total=len(names), done=0)
        for i, name in enumerate(names, 1):
            set_progress(current=name, message=f"正在处理 {name}")
            img = RAW_DIR / name
            w, h = image_size(img)
            try:
                r = model.predict(str(img), conf=0.25,
                                  classes=classes, verbose=False)[0]
                objs = [{
                    "id": f"o{i}_{k}",
                    "class_id": o["class_id"],
                    "polygon": o["polygon"],
                    "conf": o["conf"],
                    "source": "model",
                } for k, o in enumerate(P.objects_from_result(r, need_map))]
                save_annotation({"image": name, "width": w, "height": h, "objects": objs})
            except Exception as e:
                set_progress(message=f"{name} 失败：{e}")
            set_progress(done=i)
        set_progress(running=False, current="", message="预标注完成")
    except Exception as e:
        set_progress(running=False, message=f"预标注出错：{e}")


@app.route("/api/prelabel", methods=["POST"])
def api_prelabel():
    body = request.get_json(force=True) or {}
    names = body.get("names") or []
    if not names:
        names = [p.name for p in sorted(RAW_DIR.iterdir())
                 if p.suffix.lower() in IMG_EXTS
                 and not any(o.get("source") == "human"
                             for o in load_annotation(p.name)["objects"])]
    if PROGRESS["running"]:
        return jsonify({"ok": False, "message": "已经有一批在跑了"})
    set_progress(running=True, total=len(names), done=0, current="", message="准备中")
    threading.Thread(target=_prelabel_worker, args=(names,), daemon=True).start()
    return jsonify({"ok": True, "total": len(names)})


@app.route("/compare")
@app.route("/report")
def compare_page():
    """前后对比报告（make_report.py 生成）：改造前 vs 改造后的精度、第二轮回标修正量"""
    if not REPORT.exists():
        return ("<meta charset='utf-8'><p style='font:16px/1.8 sans-serif;padding:40px'>"
                "对比报告还没生成。<br>在 Miniforge Prompt 里运行：<br>"
                "<code>python eval_model.py --model model/best.pt --tag trained</code><br>"
                "<code>python round2_compare.py</code><br>"
                "<code>python make_report.py</code></p>"), 404
    return send_file(REPORT)


@app.route("/api/report")
def api_report():
    """界面用：报告是否已经生成"""
    return jsonify({"ready": REPORT.exists(),
                    "path": str(REPORT),
                    "mtime": time.strftime("%Y-%m-%d %H:%M",
                                           time.localtime(REPORT.stat().st_mtime))
                    if REPORT.exists() else None})

@app.route("/api/progress")
def api_progress():
    return jsonify(PROGRESS)


# ---------------------------------------------------------------- 看标注结果 / 导出照片
@app.route("/api/thumb/<path:name>")
def api_thumb(name):
    """返回「画好标注的照片」缩略图，照片墙和预览都用它"""
    img = RAW_DIR / name
    if not img.exists():
        return ("没有这张图", 404)
    w = request.args.get("w", type=int) or 640
    ann = load_annotation(name)
    try:
        data = render_annotated(img, ann, max_w=w)
    except Exception as e:
        return (f"渲染失败：{type(e).__name__}: {e}", 500)
    return send_file(__import__("io").BytesIO(data), mimetype="image/jpeg")


@app.route("/api/export_images", methods=["POST"])
def api_export_images():
    """把已标注的照片烧上框和标签，连同名 json 一起打包成 zip 下载"""
    import io
    import zipfile

    names = []
    for p in sorted(RAW_DIR.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        ann = load_annotation(p.name)
        if ann.get("objects"):
            names.append((p, ann))
    if not names:
        return jsonify({"ok": False, "message": "还没有标注过的照片"}), 400

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p, ann in names:
            try:
                z.writestr(f"标注照片/{p.stem}.jpg", render_annotated(p, ann))
            except Exception:
                continue
            z.writestr(f"标注照片/{p.stem}.json",
                       json.dumps(ann, ensure_ascii=False, indent=2))
        z.writestr("说明.txt",
                   f"人机协同图像标注系统 · 标注照片导出\n"
                   f"共 {len(names)} 张，时间 {time.strftime('%Y-%m-%d %H:%M')}\n"
                   f"每张 jpg 是把标注画在照片上的样子，同名 json 是原始标注数据。\n"
                   f"类别：{'/'.join(P.ZH)}\n")

    buf.seek(0)
    resp = send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name="标注照片.zip")
    resp.headers["X-Image-Count"] = str(len(names))
    return resp


# ---------------------------------------------------------------- 导出数据集
@app.route("/api/export", methods=["POST"])
def api_export():
    """把当前所有标好的图导成标准数据集（YOLO txt + 实例掩码 + 语义图）"""
    body = request.get_json(silent=True) or {}
    try:
        import export as E
        report = E.export(val_ratio=float(body.get("val_ratio", 0.2)))
    except Exception as e:
        return jsonify({"ok": False, "message": f"导出失败：{type(e).__name__}: {e}"})
    return jsonify(report)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """现场上传一张新照片（答辩演示用）：压缩到长边 960，存进 data/raw，返回文件名"""
    import io
    import re
    from PIL import Image

    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"ok": False, "message": "没有收到文件"}), 400

    stem = Path(f.filename).stem
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5_-]+", "_", stem)[:40] or "upload"
    dst = RAW_DIR / f"{stem}.jpg"
    i = 1
    while dst.exists():
        dst = RAW_DIR / f"{stem}_{i}.jpg"
        i += 1

    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception as e:
        return jsonify({"ok": False, "message": f"这张图打不开：{type(e).__name__}"}), 400

    w, h = img.size
    mx = max(w, h)
    if mx > 960:                       # 和数据集规格保持一致
        img = img.resize((round(w * 960 / mx), round(h * 960 / mx)))
    img.save(dst, "JPEG", quality=85)

    cat = request.form.get("category") or "其他"
    set_category(dst.name, cat)

    return jsonify({"ok": True, "name": dst.name, "category": cat,
                    "width": img.size[0], "height": img.size[1],
                    "message": f"已上传 {dst.name}"})


# ---------------------------------------------------------------- 启动
def open_browser_later(port=5000):
    import webbrowser
    time.sleep(1.2)
    webbrowser.open(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="预标注用的权重，默认 yolo26n-seg.pt；第二轮填 model/best.pt")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--lan", action="store_true",
                    help="允许同一 WiFi 下的手机/别的电脑访问（默认只本机）")
    args = ap.parse_args()
    if args.model:
        MODEL_PATH = (ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
        if not MODEL_PATH.exists():
            raise SystemExit(f"找不到模型文件：{MODEL_PATH}")

    host = "0.0.0.0" if args.lan else "127.0.0.1"

    print("=" * 56)
    print("  人机协同图像标注系统  已启动")
    print(f"  预标注模型：{MODEL_PATH.name}")
    print(f"  本机地址：http://127.0.0.1:{args.port}")
    if args.lan:
        import socket
        try:
            s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s_.connect(("8.8.8.8", 80))
            ip = s_.getsockname()[0]
            s_.close()
        except Exception:
            ip = "（查不到，用 ipconfig 看 IPv4 地址）"
        print(f"  局域网地址：http://{ip}:{args.port}   ← 同一 WiFi 下的手机/电脑用这个")
        print("  首次可能弹出「Windows 安全中心」允许访问，点允许")
    print("  关闭这个黑窗口即停止运行")
    print("=" * 56)
    threading.Thread(target=lambda: open_browser_later(args.port), daemon=True).start()
    app.run(host=host, port=args.port, debug=False, threaded=True)
