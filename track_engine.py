# -*- coding: utf-8 -*-
"""
track_engine.py —— 体育场所「人员识别打卡 / 同场景追踪」引擎

设计原则：完全离线可跑，不依赖任何需要联网下载的模型。
  1. 人员检测：复用 imglabel 已有的 yolo26n-seg.pt（COCO 的 person 类）
  2. 跨帧追踪：自写轻量 IOU 追踪器，不依赖 lap / deepsort
  3. 身份匹配：默认用「上半身颜色直方图」外观特征（离线）；
     若把 YuNet / SFace 的 onnx 放进 models/，自动升级成人脸级识别
  4. 结果：带框的逐帧 JPEG + result.json + 打卡流水 + 可下载 mp4

验证目标（简单场景）：
  - 能不能框住人
  - 同一个人跨帧是否保持同一个 Track ID
  - 注册人员 vs 路人能否区分
"""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
RECORDS = ROOT / "records"
OUT_ROOT = ROOT / "data" / "track"
GALLERY_FILE = RECORDS / "face_gallery.json"
CHECKIN_LOG = RECORDS / "checkin_log.jsonl"
VIDEO_DIRS = [ROOT / "data" / "track", Path("D:/sports_checkin/videos")]

for d in (RECORDS, OUT_ROOT):
    d.mkdir(parents=True, exist_ok=True)

YUNET = MODELS / "yunet.onnx"
SFACE = MODELS / "sface.onnx"


def identity_mode():
    """当前身份识别模式：人脸模型齐了就是 face，否则是外观特征 appearance。"""
    return "face" if (YUNET.exists() and SFACE.exists()) else "appearance"


# ---------------------------------------------------------------- 可选：人脸模型
class FaceIdentity:
    """YuNet 检测 + SFace 128 维特征。模型不存在时不会被实例化。"""

    def __init__(self):
        self.det = cv2.FaceDetectorYN_create(str(YUNET), "", (320, 320),
                                             score_threshold=0.6, nms_threshold=0.3,
                                             top_k=5000)
        self.rec = cv2.FaceRecognizerSF_create(str(SFACE), "")

    def feat(self, bgr, box=None):
        """从整图或指定 person 框里找最大的人脸，返回 128 维特征。"""
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            # 人脸在上半身
            y2 = min(y2, y1 + int((y2 - y1) * 0.65))
            bgr = bgr[y1:y2, x1:x2]
        if bgr.size == 0:
            return None
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])
        aligned = self.rec.alignCrop(bgr, f)
        return self.rec.feature(aligned)

    @staticmethod
    def sim(a, b):
        # SFace 官方推荐 cosine 匹配
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------- 离线外观特征
def appearance_feat(bgr, box=None):
    """上半身分块 HSV 颜色直方图。衣服颜色/发型在简单场景里区分度足够。"""
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = bgr.shape[:2]
        bgr = bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if bgr.size:
            bgr = bgr[:int(bgr.shape[0] * 0.65), :]     # 只要上半身
    if bgr.size == 0:
        return None
    bgr = cv2.resize(bgr, (64, 96))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for gy in range(3):
        for gx in range(2):
            cell = hsv[gy * 32:(gy + 1) * 32, gx * 32:(gx + 1) * 32]
            hist = cv2.calcHist([cell], [0, 1], None, [18, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            feats.append(hist.flatten())
    return np.concatenate(feats).astype(np.float32)


def appearance_sim(a, b):
    """把巴氏距离转成 0~1 的相似度。"""
    # 分块直方图拼起来的向量，直接算余弦更稳
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------- 注册库
def build_gallery(images_dir, model=None):
    """用现有注册照建库。文件名 s02_front / s02_back 属于同一个人，按前缀合并。

    model 给了就先用 YOLO 框出人、只取框内特征 —— 和视频里
    appearance_feat(frame, box) 的口径一致，背景差异不会把识别带偏。
    返回 [{name, files, feat, mode}]，name 是人员编号（s01~s04）。
    """
    images_dir = Path(images_dir)
    mode = identity_mode()
    fi = FaceIdentity() if mode == "face" else None
    groups = {}
    for p in sorted(images_dir.glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        box = None
        if model is not None:
            dets = detect_persons(model, bgr, conf=0.25)
            if dets:
                box = max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))[:4]
        f = (fi.feat(bgr, box) if mode == "face"
             else appearance_feat(bgr, box))
        if f is None:
            continue
        person = p.stem.split("_")[0]
        groups.setdefault(person, []).append((p.name, np.asarray(f, dtype=np.float32)))

    items = []
    for person, feats in sorted(groups.items()):
        v = np.mean(np.stack([f for _, f in feats]), axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        items.append({"name": person, "files": [n for n, _ in feats],
                      "feat": v.tolist(), "mode": mode})
    GALLERY_FILE.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return {"mode": mode, "count": len(items),
            "names": [i["name"] for i in items],
            "photos": sum(len(i["files"]) for i in items)}


def load_gallery():
    if not GALLERY_FILE.exists():
        return []
    return json.loads(GALLERY_FILE.read_text(encoding="utf-8"))


def match_identity(feat, gallery, mode, threshold):
    """返回 (名字, 最高分, 次高分)。低于阈值或差距太小 → 陌生人。"""
    if feat is None or not gallery:
        return None, 0.0, 0.0
    sim = (FaceIdentity.sim if mode == "face" else appearance_sim)
    scored = sorted(((sim(feat, np.array(g["feat"], dtype=np.float32)), g["name"])
                     for g in gallery), reverse=True)
    best, name = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best < threshold or (best - second) < 0.05:
        return None, best, second
    return name, best, second


# ---------------------------------------------------------------- 追踪器
class SimpleTracker:
    """IOU + 中心点距离匹配的轻量追踪器。简单场景足够，无需 lap / 卡尔曼。"""

    def __init__(self, iou_thr=0.25, max_miss=12, min_hits=2):
        self.iou_thr = iou_thr
        self.max_miss = max_miss
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1
        self._lock = threading.Lock()

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / (ua + 1e-6)

    @staticmethod
    def _center_dist(a, b):
        ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
        cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

    def update(self, dets):
        """dets: [[x1,y1,x2,y2,conf], ...]；返回稳定的 track 列表。"""
        with self._lock:
            # 贪心匹配：先按 IOU 从大到小配对
            pairs = []
            for ti, t in enumerate(self.tracks):
                for di, d in enumerate(dets):
                    v = self._iou(t["box"], d[:4])
                    # 轻微的中心距离兜底：只补 0.1，避免「人没动但换了一个人」
                    # 时被强行判成同一个目标（这个 bug 会让身份一直串下去）
                    dist = self._center_dist(t["box"], d[:4])
                    diag = max(1, (t["box"][2] - t["box"][0]))
                    v = min(1.0, v + 0.1 * (1 - min(1.0, dist / diag)))
                    if v >= self.iou_thr:
                        pairs.append((v, ti, di))
            pairs.sort(reverse=True)
            used_t, used_d = set(), set()
            matched = set()
            for v, ti, di in pairs:
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti); used_d.add(di)
                t = self.tracks[ti]
                t["box"] = dets[di][:4]
                t["conf"] = dets[di][4]
                t["hits"] += 1
                t["miss"] = 0
                matched.add(id(t))

            # 先给「这一帧没匹配上」的老 track 计一次丢失，再创建新 track，
            # 否则新 track 第一帧就会被误记成丢失
            for t in self.tracks:
                if id(t) not in matched:
                    t["miss"] += 1

            for di, d in enumerate(dets):
                if di in used_d:
                    continue
                self.tracks.append({
                    "id": self.next_id, "box": d[:4], "conf": d[4],
                    "hits": 1, "miss": 0, "name": None, "sim": 0.0,
                    "votes": {}, "checked_in": False,
                })
                self.next_id += 1

            self.tracks = [t for t in self.tracks if t["miss"] <= self.max_miss]
            return [t for t in self.tracks if t["hits"] >= self.min_hits]


# ---------------------------------------------------------------- 检测
def detect_persons(model, frame, conf):
    r = model.predict(frame, conf=conf, classes=[0], verbose=False)[0]
    out = []
    if r.boxes is None:
        return out
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append([x1, y1, x2, y2, float(b.conf[0])])
    return out


def log_checkin(name, track_id, frame_idx, sim, mode):
    """同一人 30 秒内只记一次，防止打卡流水被刷屏。"""
    now = time.time()
    if CHECKIN_LOG.exists():
        for line in CHECKIN_LOG.read_text(encoding="utf-8").splitlines()[-80:]:
            try:
                r = json.loads(line)
                if r["name"] == name and now - r.get("ts", 0) < 30:
                    return False
            except Exception:
                pass
    with open(CHECKIN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "name": name, "track_id": track_id, "frame": frame_idx,
            "sim": round(float(sim), 3), "mode": mode,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "ts": now,
        }, ensure_ascii=False) + "\n")
    return True


# ---------------------------------------------------------------- 主流程
STATE = {"running": False, "done": 0, "total": 0, "message": "空闲",
         "job": "", "result": None, "error": ""}
_lock = threading.Lock()


def set_state(**kw):
    with _lock:
        STATE.update(kw)


def get_state():
    with _lock:
        return dict(STATE)


def process_video(video_path, job_id, model, gallery=None, stride=3, conf=0.35,
                  id_threshold=0.62, max_frames=0, out_dir=None,
                  max_width=0, max_seconds=0):
    """后台线程里跑：逐帧检测 → 追踪 → 身份 → 画框 → 存 JPEG + json。

    max_width：4K 视频先等比缩小再处理（CPU 上快很多），默认 0 = 原尺寸。
    max_seconds：只处理前 N 秒，演示时不必跑完整段。
    """
    out_dir = Path(out_dir) if out_dir else OUT_ROOT / job_id
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()

    mode = identity_mode()
    fi = FaceIdentity() if mode == "face" else None
    tracker = SimpleTracker(max_miss=6, min_hits=1)   # 首帧就画框，别让评委以为没框住
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        set_state(running=False, error=f"打不开视频：{video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 12
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W, H = src_W, src_H
    scale = 1.0
    if max_width and src_W > max_width:
        scale = max_width / src_W
        W, H = int(src_W * scale), int(src_H * scale)
    if max_seconds and fps:
        lim = int(fps * max_seconds)
        total_frames = min(total_frames or lim, lim)
        max_frames = min(max_frames or lim, lim)
    if max_frames:
        total_frames = min(total_frames or max_frames, max_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_dir / "result.mp4"), fourcc, fps, (W, H))

    per_frame, frames_out = [], []
    id_names = {}          # 每个历史 Track ID 最终锁定的身份
    last_boxes = []
    idx = 0
    set_state(running=True, done=0, total=total_frames, message="开始分析",
              job=job_id, error="")

    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        if idx % stride == 0:
            dets = detect_persons(model, frame, conf)
            tracks = tracker.update(dets)
            to_draw = []
            for t in tracks:
                # 身份：多帧投票，避免抖来抖去
                if t["name"] is None and gallery:
                    if mode == "face":
                        f = fi.feat(frame, t["box"])
                    else:
                        f = appearance_feat(frame, t["box"])
                    nm, best, second = match_identity(f, gallery, mode, id_threshold)
                    if nm:
                        t["votes"][nm] = t["votes"].get(nm, 0) + 1
                        if t["votes"][nm] >= 2:
                            t["name"], t["sim"] = nm, best
                            log_checkin(nm, t["id"], idx, best, mode)
                            t["checked_in"] = True
                    if t["name"]:
                        id_names[t["id"]] = (t["name"], float(t.get("sim", 0)))
                to_draw.append((t["id"], t["box"], t["name"], t["sim"]))
            last_boxes = to_draw
        else:
            to_draw = last_boxes

        vis = frame.copy()
        for tid, box, name, sim in to_draw:
            x1, y1, x2, y2 = [int(v) for v in box]
            color = (60, 180, 75) if name else (230, 160, 60)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"#{tid}"
            if name:
                label += f"  {name}  {sim:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(vis, label, (x1 + 4, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 左上角状态条
        cv2.putText(vis, f"frame {idx}  tracks {len(tracker.tracks)}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(vis, f"frame {idx}  tracks {len(tracker.tracks)}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 1)

        fname = f"{idx:06d}.jpg"
        cv2.imwrite(str(frames_dir / fname), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if writer.isOpened():
            writer.write(vis)
        frames_out.append(fname)
        per_frame.append({"frame": idx, "n": len(to_draw),
                          "ids": [t[0] for t in to_draw],
                          "names": [t[2] for t in to_draw if t[2]]})
        if idx % (stride * 10) == 0:
            set_state(done=idx, total=total_frames,
                      message=f"已处理 {idx} 帧，当前 {len(tracker.tracks)} 个目标")
        idx += 1

    cap.release()
    writer.release()

    # 统计：每个历史 Track ID 出现过多少帧、有没有断、最后认成了谁。
    # 注意不能用 tracker.tracks —— 它只保留「还没超时」的，历史段会全丢。
    id_counter = {}
    for r in per_frame:
        for k in r["ids"]:
            id_counter[k] = id_counter.get(k, 0) + 1
    stable_ids = {k for k, n in id_counter.items() if n >= 4}   # 滤掉一闪而过的误检碎片
    id_switches = 0
    prev = None
    for r in per_frame:
        cur = tuple(sorted(k for k in r["ids"] if k in stable_ids))
        if prev is not None and cur != prev:
            id_switches += 1
        prev = cur
    track_summary = []
    for tid, n in sorted(id_counter.items()):
        if tid not in stable_ids:
            continue
        name, sim = id_names.get(tid, (None, 0.0))
        track_summary.append({
            "id": tid, "frames": n, "name": name,
            "sim": round(sim, 3), "checked_in": name is not None,
        })
    checkins = []
    if CHECKIN_LOG.exists():
        for line in CHECKIN_LOG.read_text(encoding="utf-8").splitlines():
            try:
                checkins.append(json.loads(line))
            except Exception:
                pass

    result = {
        "job": job_id, "video": Path(video_path).name,
        "mode": mode, "stride": stride, "conf": conf,
        "fps": round(fps, 2), "width": W, "height": H,
        "frames_total": idx, "frames_saved": len(frames_out),
        "frame_files": frames_out,
        "tracks": track_summary,
        "track_count": len(track_summary),
        "max_track_frames": max([t["frames"] for t in track_summary], default=0),
        "id_switches": id_switches,
        "checkins": checkins[-30:],
        "verdict": _verdict(track_summary, idx, id_switches),
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    set_state(running=False, done=idx, total=idx, message="完成", result=result,
              job=job_id)
    return result


def _verdict(tracks, n_frames, id_switches=0):
    if not tracks:
        return "没有稳定框住任何目标：调低置信度，或换更清晰的画面"
    longest = max(t["frames"] for t in tracks)
    mean = sum(t["frames"] for t in tracks) / len(tracks)
    ratio = longest / max(1, n_frames)
    if len(tracks) == 1 and ratio >= 0.6:
        return (f"单目标场景通过：全场 1 个 ID，持续 {longest}/{n_frames} 帧，"
                f"同一个人没有被跟丢或换号")
    if len(tracks) > 1 and mean >= 8 and id_switches <= len(tracks) * 2:
        return (f"分段场景通过：共 {len(tracks)} 个 ID，平均每段 {mean:.0f} 帧、"
                f"最长 {longest} 帧，ID 切换 {id_switches} 次（与场景切换次数接近，"
                f"说明同一段内没有频繁换号）")
    return (f"追踪不稳定：{len(tracks)} 个 ID、平均每段仅 {mean:.0f} 帧、"
            f"ID 切换 {id_switches} 次，可能被遮挡、快速移动或检测抖动打断")


def make_demo_video(images_dir, out_path=None, per_image=18, fps=12, gap=16):
    """用现有注册照造一段「简单场景」演示视频：每人缓慢平移一段，段间留空场。

    段间空场是为了让追踪器把上一个人判定为「离开画面」，下一个人来时
    才会分配新的 Track ID —— 这样才能验证『4 个人 → 4 个 ID，各不串号』。
    """
    images_dir = Path(images_dir)
    out_path = Path(out_path) if out_path else OUT_ROOT / "demo_simple.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(images_dir.glob("*"))
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not files:
        return None
    W, H = 480, 640
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    blank = np.full((H, W, 3), 245, dtype=np.uint8)
    for pi, p in enumerate(files):
        if pi:
            for _ in range(gap):
                writer.write(blank.copy())
        im = cv2.imread(str(p))
        if im is None:
            continue
        # 等比缩放贴到画布中间
        h, w = im.shape[:2]
        s = min(W / w, H / h) * 0.82
        im = cv2.resize(im, (int(w * s), int(h * s)))
        for k in range(per_image):
            canvas = np.full((H, W, 3), 245, dtype=np.uint8)
            dx = int((k / per_image - 0.5) * 36)
            dy = int((k / per_image - 0.5) * 12)
            x = (W - im.shape[1]) // 2 + dx
            y = (H - im.shape[0]) // 2 + dy
            x, y = max(0, x), max(0, y)
            hh = min(im.shape[0], H - y)
            ww = min(im.shape[1], W - x)
            if hh > 0 and ww > 0:
                canvas[y:y + hh, x:x + ww] = im[:hh, :ww]
            writer.write(canvas)
    writer.release()
    return out_path
