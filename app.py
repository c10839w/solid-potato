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
import track_engine as T

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

# 预标注的置信度阈值。设太低会冒出一堆误检（人得逐个删），设太高会漏标。
# 具体取多少是用 records/sweep_conf.json 的扫描结果定出来的，可用 --conf 覆盖
CONF = 0.25

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
    """文件名 -> 类别。第一次运行时按标注内容猜一遍，之后以 json 为准。
    新加进来的图（比如又采了一批）自动按内容补猜，不用手点。"""
    global _cats
    if _cats is not None:
        return _cats
    if CAT_FILE.exists():
        try:
            _cats = json.loads(CAT_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cats = {}
    else:
        _cats = {}
    added = 0
    for p in ANN_DIR.glob("*.json"):
        name = p.stem + ".jpg"
        if name in _cats:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = d.get("image") or name
        _cats[name] = _guess_category(d.get("objects", []))
        added += 1
    if added:
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


@app.route("/tracking")
def tracking_page():
    """人员识别打卡 / 同场景追踪（新加的验证页）"""
    resp = send_from_directory(app.static_folder, "track.html")
    # 打卡页改动频繁，禁用缓存，避免用户一直看到旧版脚本
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


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
                r = model.predict(str(img), conf=CONF,
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
# ---------------------------------------------------------------- 人员识别打卡 / 同场景追踪
_track_model = None
_track_lock = threading.Lock()
_video_cache = {}


def get_track_model():
    """追踪必须用 COCO 预训练权重（含 person 类）。

    注意：--model model/best.pt 是十类家具自训权重，虽然 0 号也是 person，
    但它是为室内家具微调的，场馆画面召回率差，所以这里固定用通用权重。
    """
    global _track_model
    with _track_lock:
        if _track_model is None:
            from ultralytics import YOLO
            _track_model = YOLO(str(ROOT / "yolo26n-seg.pt"))
        return _track_model


def list_track_videos():
    """列出可分析的视频：项目内的演示视频 + D:/sports_checkin/videos 下采集的视频。"""
    import cv2
    out = []
    for d in T.VIDEO_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.mp4")):
            if p.name.startswith("result"):
                continue
            key = str(p)
            if key not in _video_cache:
                cap = cv2.VideoCapture(key)
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                _video_cache[key] = {
                    "fps": round(fps, 1), "frames": n,
                    "seconds": round(n / fps, 1) if fps else 0,
                    "width": w, "height": h,
                    "size_mb": round(p.stat().st_size / 1048576, 1),
                }
            out.append({"name": p.name, "path": key, **_video_cache[key]})
    return out


@app.route("/api/track/upload", methods=["POST"])
def api_track_upload():
    """上传一段自己的视频（小文件用；4K 大文件建议直接放到 videos 目录）。"""
    import re
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"ok": False, "message": "没有收到文件"}), 400
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5_-]+", "_", Path(f.filename).stem)[:40]
    if stem.lower().startswith("result"):
        stem = "upload_" + stem        # 结果视频在列表里会被过滤掉，避免上传后"看不到"
    dst = T.OUT_ROOT / (stem + ".mp4")
    i = 1
    while dst.exists():
        dst = T.OUT_ROOT / f"{stem}_{i}.mp4"
        i += 1
    f.save(str(dst))
    _video_cache.pop(str(dst), None)
    # 直接把最新列表一起返回，前端不用再等一次 /info 就能看到并选中刚上传的视频
    return jsonify({"ok": True, "name": dst.name,
                    "message": f"已上传 {dst.name}",
                    "videos": list_track_videos()})


def _gallery_view():
    """注册库展示数据：photos = 真正进人脸模板的张数，saved = 磁盘上该人照片总数。

    之前只显示 photos，用户传了 3 张（正/背/侧）却只看到 1 张，会误以为是延迟。
    现在额外给前端「正面照 / 全部照片 / 角度 / 是否进模板」，
    点一下人员编号就能看到这个人的照片，不用去磁盘目录里翻。
    """
    def angle_of(fn):
        return _guess_register_angle(Path(fn).stem) or "other"

    g = T.load_gallery()
    view = []
    for x in g:
        used = list(x.get("files") or [])
        all_files = list(x.get("all_files") or used)
        anchor = x.get("anchor") or (used[0] if used else
                                     (all_files[0] if all_files else ""))
        fronts = [f for f in all_files if angle_of(f) == "front"]
        # 优先「已进模板的正面照」；没有就用任意正面照；再没有就用建库锚点
        front = (next((f for f in fronts if f in used), None)
                 or (fronts[0] if fronts else anchor))
        order = {"front": 0, "side": 1, "back": 2, "other": 3}
        shots = [{"name": f, "angle": angle_of(f), "used": f in used,
                  "anchor": f == anchor}
                 for f in sorted(all_files, key=lambda n: (order[angle_of(n)], n))]
        view.append({"name": x["name"], "photos": len(used),
                     "saved": len(all_files), "anchor": anchor,
                     "front": front, "shots": shots})
    folder = track_register_dir()
    disk = sum(1 for p in folder.glob("*") if p.suffix.lower() in REGISTER_IMG_EXTS)
    return view, disk


@app.route("/api/track/info")
def api_track_info():
    view, disk = _gallery_view()
    return jsonify({
        "mode": T.identity_mode(),
        "threshold": T.default_threshold(),
        "model_ready": T.identity_mode() == "face",
        "gallery": view,
        "disk_photos": disk,
        "templates": sum(x["photos"] for x in view),
        "videos": list_track_videos(),
    })


@app.route("/api/track/photo/<path:name>")
def api_track_photo(name):
    """点人员编号看注册照：只放行注册目录里的图片，?w=520 可缩到指定宽度。"""
    from flask import Response, abort
    folder = track_register_dir().resolve()
    p = (folder / Path(name).name).resolve()      # 只取文件名，挡住 ../ 穿越
    if p.parent != folder or not p.is_file() or p.suffix.lower() not in REGISTER_IMG_EXTS:
        abort(404)
    w = request.args.get("w", type=int)
    if not w:
        return send_file(str(p), conditional=True)
    w = max(64, min(1600, w))
    import cv2
    import numpy as np
    # 中文路径 cv2.imread 会静默失败，统一用 imdecode 读字节
    bgr = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        abort(404)
    if bgr.shape[1] > w:
        s = w / bgr.shape[1]
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        abort(404)
    return Response(buf.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "max-age=60"})


# ---------------------------------------------------------------- 注册照上传
# 注册照默认沿用之前解包出来的素材目录；别人 clone 仓库时没有 D 盘，
# 就自动落到项目内的 data/registered（已在 .gitignore 忽略，照片不进仓库）。
DEFAULT_REGISTER_DIR = Path("D:/sports_checkin/registered")
REGISTER_ANGLES = (
    ("front", ("front", "正面", "正脸")),
    ("back", ("back", "背面", "后脑", "背影")),
    ("side", ("side", "profile", "侧面", "侧脸")),
)
REGISTER_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GENERIC_NAMES = {"img", "dsc", "dscn", "photo", "image", "mmexport",
                 "screenshot", "微信图片", "屏幕截图"}


def track_register_dir(override=None):
    """注册照目录；目录不存在时自动建，保证 clone 后也能直接上传。"""
    if override:
        p = Path(override)
    elif DEFAULT_REGISTER_DIR.is_dir():
        p = DEFAULT_REGISTER_DIR
    else:
        p = ROOT / "data" / "registered"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _guess_register_angle(stem):
    s = (stem or "").lower()
    for angle, keys in REGISTER_ANGLES:
        if any(k in s for k in keys):
            return angle
    return None


def _clean_person(name):
    import re
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5_-]+", "_", name or "").strip("_-")
    return s[:24]


@app.route("/api/track/register_upload", methods=["POST"])
def api_track_register_upload():
    """上传注册人员照片（可多选），自动按「人员_角度」落盘并重建人脸库。

    文件名含 front/back/side 或 正面/背面/侧面 会被识别成角度；
    没写角度时按这个人还缺哪张补哪张（优先补正面，保证能建出人脸模板）。
    可传 person 字段显式指定人员编号；不传则从文件名公共前缀推断，
    推断不出来自动分配 s05、s06……与现有注册库命名保持一致。
    照片统一存成 jpg，避免 webp / 中文名 / 透明通道导致建库时漏读。
    """
    import re
    import cv2
    import numpy as np

    files = [f for f in (request.files.getlist("files") or request.files.getlist("file"))
             if f and f.filename]
    if not files:
        return jsonify({"ok": False, "message": "没有收到照片"}), 400

    folder = track_register_dir(request.form.get("dir"))
    person = _clean_person(request.form.get("person"))
    if not person:
        guesses = []
        for f in files:
            base = Path(f.filename).stem
            for _, keys in REGISTER_ANGLES:
                for k in keys:
                    base = re.sub(re.escape(k), "", base, flags=re.I)
            base = re.split(r"[_\-\s]", base.strip("_- "), maxsplit=1)[0]
            guesses.append(_clean_person(base))
        if (len(set(guesses)) == 1 and guesses[0]
                and len(guesses[0]) <= 12
                and guesses[0].lower() not in GENERIC_NAMES):
            person = guesses[0]
    if not person:
        used = [int(m.group(1)) for p in folder.glob("s[0-9][0-9]_*")
                for m in [re.match(r"s(\d+)_", p.stem)] if m]
        person = "s%02d" % (max(used + [4]) + 1)

    saved, skipped = [], []
    for f in files[:20]:
        stem = Path(f.filename).stem
        angle = _guess_register_angle(stem)
        if angle is None:
            for a in ("front", "back", "side"):
                if not (folder / f"{person}_{a}.jpg").exists():
                    angle = a
                    break
            angle = angle or "front"
        if Path(f.filename).suffix.lower() not in REGISTER_IMG_EXTS:
            skipped.append(f.filename + "（不是图片）")
            continue
        raw = f.read()
        if len(raw) > 25 * 1024 * 1024:
            skipped.append(f.filename + "（超过 25MB）")
            continue
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            skipped.append(f.filename + "（打不开）")
            continue
        dst = folder / f"{person}_{angle}.jpg"
        i = 2
        while dst.exists():
            dst = folder / f"{person}_{angle}_{i}.jpg"
            i += 1
        cv2.imwrite(str(dst), arr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append({"name": dst.name, "angle": angle})

    if not saved:
        return jsonify({"ok": False, "message": "没有有效照片；" + "、".join(skipped)}), 400

    info = T.build_gallery(str(folder), get_track_model())
    view, disk = _gallery_view()
    rej = info.get("rejected", [])
    msg = (f"已保存 {len(saved)} 张（{person}）。注册库现有 {info['count']} 人、"
           f"{info['photos']} 张有效模板（磁盘共 {disk} 张照片）")
    if rej:
        msg += f"；{len(rej)} 张未进模板，原因见下方"
    if person not in info["names"]:
        msg += "。注意：这个人还没进人脸模板，请补一张清晰的正面照"
    if skipped:
        msg += "。跳过：" + "、".join(skipped)
    if len(saved) > 1 and not request.form.get("person"):
        msg += "。若这几张不是同一个人，在下方「人工修正分组」里把照片拖到不同编号"
    if T.get_state()["running"]:
        msg += "。当前有分析在跑，新注册库从下一次分析开始生效"
    return jsonify({"ok": True, "message": msg, "person": person,
                    "saved": saved, "skipped": skipped,
                    "gallery": view, "disk_photos": disk, **info})


@app.route("/api/track/register", methods=["POST"])
def api_track_register():
    d = request.get_json(force=True) or {}
    folder = track_register_dir(d.get("dir"))
    try:
        info = T.build_gallery(str(folder), get_track_model())
        view, disk = _gallery_view()
        return jsonify({"ok": True, "gallery": view,
                        "disk_photos": disk, **info})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


# ---------------------------------------------------------------- 人工修正分组
# 注册照的归属直接由「文件名前缀」决定，所以拖拽修正 = 改前缀 + 重建注册库。
# 不做全自动人脸聚类：背面 / 远景照检到的多是背景假脸，实测跨人假脸相似度
# 可达 0.65（比真人还高），自动拆会把同一个人的正背面拆开。
_register_lock = threading.Lock()
_photo_meta_cache = {"key": None, "value": None}


def _next_person(folder):
    """给「新建编号」用：找下一个没被占用的 sNN。"""
    import re
    used = set()
    for p in folder.glob("s[0-9][0-9]_*"):
        m = re.match(r"s(\d+)_", p.stem)
        if m:
            used.add(int(m.group(1)))
    for g in T.load_gallery():
        m = re.fullmatch(r"s(\d+)", g["name"])
        if m:
            used.add(int(m.group(1)))
    return "s%02d" % (max(list(used) + [0]) + 1)


def _register_state(folder, with_photos=True):
    """面板要用的完整状态；照片诊断较慢，按「文件清单 + 注册库 mtime」缓存。"""
    view, disk = _gallery_view()
    state = {"gallery": view, "disk_photos": disk,
             "next_person": _next_person(folder)}
    if with_photos:
        gf = T.GALLERY_FILE
        # 用户可能正在拖照片改文件名，glob 到 stat 之间文件会被移走；
        # 单个文件读不到就跳过，不能让整个面板 500。
        sig = []
        for p in folder.glob("*"):
            if p.suffix.lower() not in REGISTER_IMG_EXTS:
                continue
            try:
                st = p.stat()
                sig.append((p.name, st.st_size, st.st_mtime_ns))
            except OSError:
                continue
        try:
            gsig = (gf.stat().st_mtime_ns, gf.stat().st_size)
        except OSError:
            gsig = (0, 0)
        key = (tuple(sorted(sig)), gsig)
        if _photo_meta_cache["key"] != key:
            _photo_meta_cache["value"] = T.photo_diagnostics(str(folder))
            _photo_meta_cache["key"] = key
        state["photos"] = _photo_meta_cache["value"]
        state["suspects"] = [x["name"] for x in state["photos"] if x["suspect"]]
        state["duplicates"] = [x["name"] for x in state["photos"]
                               if x["dup"] and not x["unsorted"]]
        state["templates"] = sum(g["photos"] for g in view)
    return state


@app.route("/api/track/register_photos")
def api_track_register_photos():
    """人工修正面板：每张注册照的编号、角度、人脸质量、与本人相似度、是否可疑。"""
    # 与移动操作互斥，否则会读到「改到一半」的目录
    with _register_lock:
        return jsonify({"ok": True, **_register_state(track_register_dir())})


@app.route("/api/track/register_move", methods=["POST"])
def api_track_register_move():
    """把照片拖到另一个编号：物理重命名前缀 → 重建人脸库 → 返回最新状态。

    to 可以是现成编号、"__new__"（自动开 sNN）、"__unsorted__"（移出到待确认）。
    只处理注册目录内的合法图片，挡住 ../ 穿越；每次移动写 records/photo_moves.jsonl 便于回溯。
    """
    d = request.get_json(force=True) or {}
    files = d.get("files") or []
    if isinstance(files, str):
        files = [files]
    to = str(d.get("to") or "").strip()
    if not files or not to:
        return jsonify({"ok": False, "message": "缺少照片或目标编号"}), 400
    folder = track_register_dir().resolve()
    if to == "__new__":
        to = _next_person(folder)
    elif to == "__unsorted__":
        to = T.UNSORTED_KEY
    else:
        to = _clean_person(to)
    if not to:
        return jsonify({"ok": False, "message": "目标编号不合法"}), 400

    with _register_lock:
        valid = []
        for name in files[:30]:
            p = (folder / Path(str(name)).name).resolve()
            if (p.parent == folder and p.is_file()
                    and p.suffix.lower() in REGISTER_IMG_EXTS):
                valid.append(p)
        if not valid:
            return jsonify({"ok": False, "message": "没有找到可移动的照片"}), 404
        todo = [p for p in valid if p.stem.split("_")[0] != to]
        if not todo:
            label = "待确认" if to == T.UNSORTED_KEY else to
            return jsonify({"ok": True, "moved": [], "to": to,
                            "message": f"这些照片已经在「{label}」里，无需移动",
                            **_register_state(folder)})

        log = ROOT / "records" / "photo_moves.jsonl"
        moved = []
        for p in todo:
            angle = _guess_register_angle(p.stem) or "photo"
            ext = p.suffix.lower() if p.suffix.lower() in REGISTER_IMG_EXTS else ".jpg"
            dst = folder / f"{to}_{angle}{ext}"
            i = 2
            while dst.exists():
                dst = folder / f"{to}_{angle}_{i}{ext}"
                i += 1
            old = p.name
            try:
                p.rename(dst)
            except OSError:
                continue          # 并发拖动时文件已被别的请求移走，跳过即可
            moved.append({"from": old, "to": dst.name})
            try:
                with log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                         "from": old, "to": dst.name, "person": to},
                                        ensure_ascii=False) + "\n")
            except OSError:
                pass
        # 人脸模式不需要 YOLO，别为了一次拖拽去加载几十 MB 的分割模型
        info = T.build_gallery(str(folder))
        _photo_meta_cache["key"] = None
        state = _register_state(folder)
        state["rejected"] = info.get("rejected", [])

    label = "待确认" if to == T.UNSORTED_KEY else to
    return jsonify({"ok": True, "moved": moved, "to": to,
                    "message": f"已移动 {len(moved)} 张到「{label}」，注册库重建为 "
                               f"{info['count']} 人 / {info['photos']} 张有效模板",
                    **state})


@app.route("/api/track/start", methods=["POST"])
def api_track_start():
    if T.get_state()["running"]:
        return jsonify({"ok": False, "message": "已有分析任务在跑，等它结束"}), 409
    d = request.get_json(force=True) or {}
    name = Path(d.get("video", "")).name
    video = next((v["path"] for v in list_track_videos() if v["name"] == name), None)
    if not video:
        return jsonify({"ok": False, "message": f"找不到视频 {name}"}), 404
    job = "job_" + time.strftime("%H%M%S")
    conf = float(d.get("conf", 0.35))
    stride = max(1, int(d.get("stride", 8)))
    seconds = max(2, min(60, int(d.get("seconds", 10))))
    use_id = bool(d.get("use_id", True))

    def worker():
        try:
            T.process_video(
                video, job, get_track_model(),
                T.load_gallery() if use_id else None,
                stride=stride, conf=conf, max_width=1280,
                max_seconds=seconds, id_threshold=T.default_threshold(),
            )
        except Exception as e:
            T.set_state(running=False, error=str(e))

    # 先把状态置为「运行中」再起线程，否则页面第一次轮询会看到空闲、误判任务结束
    T.set_state(running=True, done=0, total=0, message="正在加载模型……",
                job=job, error="", result=None)
    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job": job})


@app.route("/api/track/status")
def api_track_status():
    s = T.get_state()
    s.pop("result", None)          # 结果单独取，别把整份塞进轮询
    return jsonify(s)


@app.route("/api/track/result")
def api_track_result():
    r = T.get_state().get("result")
    if not r:
        js = sorted(T.OUT_ROOT.glob("*/result.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True)
        if js:
            r = json.loads(js[0].read_text(encoding="utf-8"))
    return jsonify({"ok": bool(r), "result": r})


@app.route("/api/track/frame/<job>/<int:n>")
def api_track_frame(job, n):
    d = T.OUT_ROOT / job / "frames"
    files = sorted(d.glob("*.jpg"))
    if n < 0 or n >= len(files):
        return "", 404
    return send_file(str(files[n]), mimetype="image/jpeg")


@app.route("/api/track/download/<job>")
def api_track_download(job):
    p = T.OUT_ROOT / job / "result.mp4"
    if not p.exists():
        return "", 404
    return send_file(str(p), as_attachment=True,
                     download_name=f"打卡追踪_{job}.mp4")


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
    ap.add_argument("--conf", type=float, default=0.25,
                    help="预标注置信度阈值，默认 0.25；误检多就调高到 0.4 左右")
    ap.add_argument("--lan", action="store_true",
                    help="允许同一 WiFi 下的手机/别的电脑访问（默认只本机）")
    args = ap.parse_args()
    CONF = args.conf
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
