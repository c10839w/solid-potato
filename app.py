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

from flask import Flask, jsonify, request, send_from_directory

import prelabel as P

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PREVIEW_DIR = ROOT / "data" / "preview"
ANN_DIR = ROOT / "records" / "annotations"
FIX_LOG = ROOT / "records" / "fixes.jsonl"

for d in (RAW_DIR, PREVIEW_DIR, ANN_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(ROOT / "canvas"), static_url_path="/static")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------- 模型（全局只加载一次）
_model = None
_model_lock = threading.Lock()


def get_model():
    """第一次调用时加载模型，之后复用。加载一次约 10 秒，不能每张图都重载。"""
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            _model = YOLO(str(ROOT / "yolo26n-seg.pt"))
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


# ---------------------------------------------------------------- 页面
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


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
    """图片列表 + 每张的标注状态"""
    items = []
    for p in sorted(RAW_DIR.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        n_obj, n_human = count_objects(p.name)
        items.append({"name": p.name, "objects": n_obj, "human": n_human})
    return jsonify({"images": items, "fixes": count_fixes()})


@app.route("/api/img/<path:name>")
def api_img(name):
    return send_from_directory(RAW_DIR, name)


@app.route("/api/annotation/<path:name>")
def api_get_ann(name):
    return jsonify(load_annotation(name))


@app.route("/api/annotation/<path:name>", methods=["POST"])
def api_post_ann(name):
    data = request.get_json(force=True)
    data["image"] = name
    fixes = data.pop("_fixes", [])      # 先把修正记录摘出来，别写进标注文件
    save_annotation(data)
    log_fixes(fixes)
    return jsonify({"ok": True, "fixes": count_fixes()})


# ---------------------------------------------------------------- 预标注（后台异步 + 进度）
def _prelabel_worker(names):
    try:
        model = get_model()
        set_progress(total=len(names), done=0)
        for i, name in enumerate(names, 1):
            set_progress(current=name, message=f"正在处理 {name}")
            img = RAW_DIR / name
            w, h = image_size(img)
            try:
                r = model.predict(str(img), conf=0.25,
                                  classes=P.COCO_IDS, verbose=False)[0]
                objs = []
                if r.masks is not None:
                    for poly, cls, score in zip(r.masks.xy, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                        if len(poly) < 3:
                            continue
                        pts = [[float(x), float(y)] for x, y in poly]
                        objs.append({
                            "id": f"o{i}_{k}",
                            "class_id": P.COCO_TO_OURS[int(cls)],
                            "polygon": pts,
                            "conf": round(float(score), 3),
                            "source": "model",
                        })
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


@app.route("/api/progress")
def api_progress():
    return jsonify(PROGRESS)


# ---------------------------------------------------------------- 启动
def open_browser_later():
    import webbrowser
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("=" * 56)
    print("  人机协同图像标注系统  已启动")
    print("  浏览器地址：http://127.0.0.1:5000")
    print("  关闭这个黑窗口即停止运行")
    print("=" * 56)
    threading.Thread(target=open_browser_later, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
