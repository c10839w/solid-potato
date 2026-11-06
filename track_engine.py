# -*- coding: utf-8 -*-
"""
track_engine.py —— 体育场所「人员识别打卡 / 同场景追踪」引擎

  1. 人员检测：yolo26n-seg.pt 的 COCO person 类
  2. 跨帧追踪：自写轻量 IOU 追踪器（不依赖 lap / deepsort）
  3. 身份识别：YuNet 人脸检测 + SFace 128 维特征，与注册库做 cosine 比对
     - 只认「提供的数据」里的注册人员，严格阈值 + 多人 gap + 多帧投票
     - 没匹配上一律视为路人，绝不猜测
     - 模型缺失时退回外观直方图（会认错人，页面会如实标注）
  4. 画框：注册人员 = 红色，路人 = 蓝色
  5. 结果：带框逐帧 JPEG + result.json + 打卡流水 + 可下载 mp4
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

# SFace 官方同人 cosine 阈值是 0.363；这里提到 0.55 更严格：
# 实测注册人员命中 0.83~0.93，真实视频路人最高 0.28，中间留了很宽的安全带。
FACE_THRESHOLD = 0.55
FACE_MARGIN = 0.10          # 第一名必须比第二名高出这么多，避免"像谁都像一点"
APPEARANCE_THRESHOLD = 0.75
VOTE_MIN = 2                # 累计命中 2 帧才锁定；阈值 0.55 + gap 0.10 已足够严，2 帧可兼顾抽帧场景

# 注册照建库：SFace 官方同人 cosine 阈值 0.363，低于此值的照片极可能是
# 误检的假脸、另一个人、或姿态/光线差太大的照片，混进模板会把本人特征带偏。
REGISTER_MIN_SCORE = 0.45   # 注册照人脸置信度（视频帧用 0.40，注册照略严）
SAME_PERSON_MIN = 0.36      # 与本人锚点的最低相似度
REGISTER_MIN_FACE = 40      # 人脸短边最低像素；低于此值的"模板"其实是背景纹理
LOW_QUALITY_PENALTY = 0.10  # 小脸模板要求比正常阈值再高这么多，防止误认
BACK_KEYS = ("back", "背面", "后脑", "背影")
# 「待确认」栏里的照片用这个前缀，只存档、不建模板、不参与识别
UNSORTED_KEY = "unsorted"

# OpenCV 是 BGR，不是 RGB
COLOR_REGISTERED = (60, 60, 235)    # 红：数据里提供的注册人员
COLOR_PASSERBY = (235, 150, 60)     # 蓝：路人 / 未匹配上


def identity_mode():
    """人脸模型齐了就是 face，否则退回外观特征 appearance。"""
    return "face" if (YUNET.exists() and SFACE.exists()) else "appearance"


def default_threshold():
    return FACE_THRESHOLD if identity_mode() == "face" else APPEARANCE_THRESHOLD


# ---------------------------------------------------------------- 人脸模型
class FaceIdentity:
    """YuNet 检测 + SFace 128 维特征。feature() 返回 (1,128)，一律 ravel 成 (128,)。

    这个 shape 细节很关键：直接 np.dot((1,128),(1,128)) 会抛
    "shapes not aligned"，之前外观模式下不会暴露，切到人脸模式立刻崩。
    """

    def __init__(self):
        self.det = cv2.FaceDetectorYN_create(str(YUNET), "", (320, 320),
                                             score_threshold=0.40, nms_threshold=0.3,
                                             top_k=5000)
        self.rec = cv2.FaceRecognizerSF_create(str(SFACE), "")
        self._size = None

    def feat(self, bgr, box=None, up=1.0, min_score=0.40,
             return_score=False, return_meta=False):
        """从整图或 person 框上半身里找最可信的人脸，返回 128 维特征或 None。

        up > 1 时先把局部放大 —— 远景里 20~40px 的小脸放大后 YuNet 才检得出。
        return_score=True  → (特征, 人脸置信度)
        return_meta=True   → (特征, 人脸置信度, 脸宽, 脸高)，建库时判断模板质量。
        """
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            y2 = min(y2, y1 + max(1, int((y2 - y1) * 0.65)))   # 人脸只在上半身
            bgr = bgr[y1:y2, x1:x2]
        if return_meta:
            empty = (None, 0.0, 0.0, 0.0)
        elif return_score:
            empty = (None, 0.0)
        else:
            empty = None
        if bgr is None or bgr.size == 0:
            return empty
        if up != 1.0:
            bgr = cv2.resize(bgr, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        h, w = bgr.shape[:2]
        if h < 24 or w < 24:
            return empty
        if self._size != (w, h):
            self.det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return empty
        f = max(faces, key=lambda r: r[2] * r[3])
        score = float(f[-1])
        if score < min_score:
            return empty
        aligned = self.rec.alignCrop(bgr, f)
        v = np.asarray(self.rec.feature(aligned), dtype=np.float32).ravel()
        if return_meta:
            return v, score, float(f[2]), float(f[3])
        return (v, score) if return_score else v

    @staticmethod
    def sim(a, b):
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------- 离线外观特征（兜底）
def appearance_feat(bgr, box=None):
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = bgr.shape[:2]
        bgr = bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if bgr.size:
            bgr = bgr[:int(bgr.shape[0] * 0.65), :]
    if bgr is None or bgr.size == 0:
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
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------- 注册库
def build_gallery(images_dir, model=None):
    """用注册照建库：多角度都尝试，但必须通过与本人锚点的一致性校验。

    为什么改：旧版只认文件名里的 front，用户按提示传的侧面/其他角度全被丢；
    而且误检的"假脸"和不属于本人的照片会混进模板再取均值，把本人特征带偏 ——
    实测同一人的两张正面照互相只有 0.15~0.23（同人阈值 0.363），
    一张背面误检照甚至更像另一个人（0.688）。

    现在的规则：
      ① 背面照一律排除（没有脸，YuNet 检到的都是背景误检），仅存档
      ② 每人挑人脸置信度最高的照片作锚点
      ③ 其余照片与锚点 cosine >= SAME_PERSON_MIN 才收为独立模板
      ④ 多模板比对取最高分，绝不取均值（均值会让不同姿态互相抵消）
    """
    images_dir = Path(images_dir)
    mode = identity_mode()
    fi = FaceIdentity() if mode == "face" else None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in exts)
    rejected = []
    all_by_person = {}
    candidates = {}

    for p in files:
        stem = p.stem.lower()
        person = p.stem.split("_")[0]
        all_by_person.setdefault(person, []).append(p.name)
        if person.lower() == UNSORTED_KEY:
            # 人工修正时先移出的照片：还没确定是谁，绝不能拿去建模板
            rejected.append({"file": p.name, "reason": "待确认分组（不参与识别）"})
            continue
        is_back = any(k in stem for k in BACK_KEYS)
        if mode == "face" and is_back:
            rejected.append({"file": p.name, "reason": "背面照仅存档（没有人脸）"})
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            rejected.append({"file": p.name, "reason": "图片打不开"})
            continue
        if mode == "face":
            f, score, fw, fh = fi.feat(bgr, None, up=1.0,
                                       min_score=REGISTER_MIN_SCORE,
                                       return_meta=True)
            if f is None:   # 小脸放大再试一次，尺寸要还原回原图
                f, score, fw, fh = fi.feat(bgr, None, up=2.0,
                                           min_score=0.40, return_meta=True)
                fw, fh = fw / 2.0, fh / 2.0
            if f is None:
                rejected.append({"file": p.name, "reason": "没检出清晰人脸"})
                continue
            short = min(fw, fh)
            if short < REGISTER_MIN_FACE:
                rejected.append({
                    "file": p.name,
                    "reason": f"人脸仅 {int(short)}px（需 ≥{REGISTER_MIN_FACE}px），"
                              f"请重拍平视近景正面照"})
                continue
            # 远景小脸的"特征"往往只是背景纹理，和别人的假脸会虚高相似，
            # 这种模板必须要求更高的相似度才敢判本人。
            quality = "high" if (score >= 0.85 and short >= 60) else "low"
        else:
            box = None
            if model is not None:
                dets = detect_persons(model, bgr, conf=0.25)
                if dets:
                    box = max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))[:4]
            f = appearance_feat(bgr, box)
            score, quality = 1.0, "low"
            if f is None:
                rejected.append({"file": p.name, "reason": "提不到特征"})
                continue
        candidates.setdefault(person, []).append(
            {"file": p.name, "feat": np.asarray(f, dtype=np.float32).ravel(),
             "score": float(score), "quality": quality,
             "short": float(short) if mode == "face" else 0.0,
             "is_front": any(k in stem for k in ("front", "正面", "正脸"))})

    items = []
    for person, cands in sorted(candidates.items()):
        # 同一编号里混了人时，先按「人脸最可信」选锚点；置信度相同再优先
        # 文件名写明 front 的、脸更大的，别让远景假脸抢走主模板位置。
        cands.sort(key=lambda c: (c["score"], int(c["is_front"]), c["short"]),
                   reverse=True)
        anchor = cands[0]
        kept = [anchor]
        for c in cands[1:]:
            s = (FaceIdentity.sim(anchor["feat"], c["feat"]) if mode == "face"
                 else appearance_sim(anchor["feat"], c["feat"]))
            if s >= SAME_PERSON_MIN:
                kept.append(c)
            else:
                rejected.append({"file": c["file"],
                                 "reason": f"与 {person} 正面照不像（{s:.2f}），"
                                           f"可能是别人、假脸或角度过大"})
        arrs = [c["feat"] for c in kept]
        items.append({
            "name": person,
            "files": [c["file"] for c in kept],       # 真正进模板的照片
            "all_files": all_by_person.get(person, []),  # 磁盘上该人全部照片
            "anchor": anchor["file"],
            "quality": ("high" if any(c["quality"] == "high" for c in kept)
                        else "low"),
            "feat": anchor["feat"].tolist(),           # 主模板 = 最清晰那张，不取均值
            "feats": [a.tolist() for a in arrs],
            "mode": mode,
        })

    GALLERY_FILE.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return {
        "mode": mode, "count": len(items),
        "names": [i["name"] for i in items],
        "photos": sum(len(i["files"]) for i in items),   # 有效模板数
        "disk_photos": len(files),                       # 磁盘实际照片数
        "rejected": rejected,
    }


def load_gallery():
    if not GALLERY_FILE.exists():
        return []
    try:
        items = json.loads(GALLERY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    # 旧版本是 appearance 的颜色直方图，和 SFace 特征一样都是 128 维，
    # 但含义完全不同；模型切换后若混用会算出无意义的相似度甚至误认。
    # 宁可当作「还没建库」，让页面上点重建 / 上传注册照重新生成。
    mode = identity_mode()
    if items and any(i.get("mode", mode) != mode for i in items):
        return []
    return items


def photo_diagnostics(images_dir):
    """给「人工修正分组」面板用：逐张报告人脸质量 / 与本人锚点相似度 / 是否可疑。

    只把「清晰正面脸 + 与本人锚点 cosine < SAME_PERSON_MIN」标为可疑。
    背面照、小脸、检不到脸一律只标「仅存档」，不自动判成别人 —— 这些图上
    YuNet 检到的多是背景假脸，实测跨编号假脸相似度能到 0.65，比真人还高，
    拿它们做自动聚类会把同一个人的正/背面拆开、把不同人的背景聚到一起。
    """
    images_dir = Path(images_dir)
    gallery = load_gallery()
    used = {f for g in gallery for f in g.get("files", [])}
    anchors = {g["name"]: (np.asarray(g["feat"], dtype=np.float32), g.get("anchor"))
               for g in gallery}
    fi = FaceIdentity() if identity_mode() == "face" else None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    for p in sorted(x for x in images_dir.glob("*") if x.suffix.lower() in exts):
        stem = p.stem.lower()
        person = p.stem.split("_")[0]
        is_back = any(k in stem for k in BACK_KEYS)
        rec = {"name": p.name, "person": person, "face": False, "short": 0,
               "score": 0.0, "sim": None, "used": p.name in used,
               "anchor": anchors.get(person, (None, None))[1] == p.name,
               "unsorted": person.lower() == UNSORTED_KEY,
               "is_back": is_back, "suspect": False, "label": "仅存档",
               "suggest": None, "suggest_sim": None, "dup": False,
               "best": None, "best_sim": None}
        try:
            raw = p.read_bytes() if p.is_file() else None
        except OSError:      # 拖动改名过程中文件可能刚好被移走
            raw = None
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
        if bgr is not None and fi is not None:
            f, sc, fw, fh = fi.feat(bgr, None, up=1.0,
                                    min_score=REGISTER_MIN_SCORE, return_meta=True)
            if f is None:   # 小脸放大再试一次，尺寸还原回原图
                f, sc, fw, fh = fi.feat(bgr, None, up=2.0,
                                        min_score=0.40, return_meta=True)
                fw, fh = fw / 2.0, fh / 2.0
            rec["short"] = round(float(min(fw, fh)), 1)
            rec["score"] = round(float(sc), 3)
            rec["face"] = f is not None
            if f is not None and rec["short"] >= REGISTER_MIN_FACE:
                # 与每个人的主模板都比一遍：给出「更像谁」的拖拽建议
                sims = {n: FaceIdentity.sim(a[0], f) for n, a in anchors.items()}
                if sims:
                    best = max(sims, key=sims.get)
                    rec["sim"] = round(sims.get(person, sims[best]), 3)
                    rec["best"] = best
                    rec["best_sim"] = round(sims[best], 3)
                    if best != person and sims[best] >= 0.45:
                        rec["suggest"] = best
                        rec["suggest_sim"] = round(sims[best], 3)
                        rec["dup"] = sims[best] >= 0.98
                # 只有「脸够大、检测够可信、又不是主模板」的照片才敢标可疑，
                # 避免把远景假脸 / 小脸误判成另一个人
                if (not is_back and not rec["anchor"] and rec["score"] >= 0.60
                        and rec["short"] >= 60 and person in anchors
                        and sims.get(person, 1.0) < SAME_PERSON_MIN):
                    rec["suspect"] = True
        if rec["unsorted"]:
            rec["label"] = "待确认"
        elif rec["dup"]:
            rec["label"] = "与 " + rec["suggest"] + " 的照片重复"
        elif rec["anchor"]:
            rec["label"] = "主模板"
        elif rec["used"]:
            rec["label"] = "有效模板"
        elif rec["suspect"]:
            rec["label"] = "可疑 · 可能不是本人"
        elif not rec["face"]:
            rec["label"] = "仅存档 · 没检到脸"
        elif rec["short"] < REGISTER_MIN_FACE:
            rec["label"] = "仅存档 · 脸太小"
        elif is_back:
            rec["label"] = "仅存档 · 背面"
        else:
            rec["label"] = "仅存档 · 与本人不像"
        out.append(rec)
    return out


def match_identity(feat, gallery, mode, threshold, margin=FACE_MARGIN):
    """返回 (名字, 最高分, 次高分)。低于阈值、或与第二名太接近 → 路人。"""
    if feat is None or not gallery:
        return None, 0.0, 0.0
    sim = (FaceIdentity.sim if mode == "face" else appearance_sim)
    scored = []
    for g in gallery:
        templates = g.get("feats") or [g["feat"]]
        s = max(sim(feat, np.asarray(t, dtype=np.float32)) for t in templates)
        # 远景小脸模板要与假脸拉开更大距离才敢认，否则宁可判路人
        req = threshold + (LOW_QUALITY_PENALTY
                           if g.get("quality") == "low" else 0.0)
        scored.append((s, g["name"], s >= req))
    scored.sort(key=lambda x: x[0], reverse=True)
    best, name, eligible = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if not eligible or best < threshold or (best - second) < margin:
        return None, best, second
    return name, best, second


# ---------------------------------------------------------------- 框重叠处理
def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-6)


def contain_ratio(a, b):
    """a 被 b 包住的比例：一个人被切成「大框 + 小框」时用来去掉小框。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def dedup_boxes(dets, iou_thr=0.55, contain_thr=0.65):
    """YOLO 偶尔把同一个人切成两个重叠框，先在这里清一次。"""
    out = []
    for d in sorted(dets, key=lambda x: x[4], reverse=True):
        if any(box_iou(d[:4], k[:4]) > iou_thr or
               contain_ratio(d[:4], k[:4]) > contain_thr for k in out):
            continue
        out.append(d)
    return out


# ---------------------------------------------------------------- 追踪器
class SimpleTracker:
    """IOU + 中心点距离匹配的轻量追踪器。

    防重框三道闸：
      ① detect_persons 出来的框先 dedup_boxes
      ② 新建 track 前，若与任何现存 track 高重叠 → 判为重复框，不分配新 ID
      ③ 返回画框列表前再做一次轨迹级去重，保留 hits 大的那个
    """

    def __init__(self, iou_thr=0.25, max_miss=6, min_hits=1, dup_iou=0.50):
        self.iou_thr = iou_thr
        self.max_miss = max_miss
        self.min_hits = min_hits
        self.dup_iou = dup_iou
        self.tracks = []
        self.next_id = 1
        self._lock = threading.Lock()

    _iou = staticmethod(box_iou)

    @staticmethod
    def _center_dist(a, b):
        ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
        cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

    def update(self, dets, dt=1):
        """dt = 距上次检测隔了多少帧。miss 按真实帧数累积，
        否则抽帧间隔一变，「人已离开画面多久」的判定就跟着变，容易把两个人接成同一个 ID。"""
        with self._lock:
            pairs = []
            for ti, t in enumerate(self.tracks):
                for di, d in enumerate(dets):
                    v = self._iou(t["box"], d[:4])
                    dist = self._center_dist(t["box"], d[:4])
                    diag = max(1, (t["box"][2] - t["box"][0]))
                    v = min(1.0, v + 0.1 * (1 - min(1.0, dist / diag)))
                    if v >= self.iou_thr:
                        pairs.append((v, ti, di))
            pairs.sort(reverse=True)
            used_t, used_d, matched = set(), set(), set()
            for v, ti, di in pairs:
                if ti in used_t or di in used_d:   # 一个框只给一个 ID，一个 ID 只占一个框
                    continue
                used_t.add(ti); used_d.add(di)
                t = self.tracks[ti]
                t["box"] = dets[di][:4]
                t["conf"] = dets[di][4]
                t["contour"] = dets[di][5] if len(dets[di]) > 5 else None
                t["hits"] += 1
                t["miss"] = 0
                matched.add(id(t))

            for t in self.tracks:
                if id(t) not in matched:
                    t["miss"] += dt

            for di, d in enumerate(dets):
                if di in used_d:
                    continue
                # ② 与现存轨迹高度重叠 → 同一个人的重复框，不再开新 ID
                if any(self._iou(t["box"], d[:4]) > self.dup_iou or
                       contain_ratio(d[:4], t["box"]) > 0.70 for t in self.tracks):
                    continue
                self.tracks.append({
                    "id": self.next_id, "box": d[:4], "conf": d[4],
                    "contour": d[5] if len(d) > 5 else None,
                    "hits": 1, "miss": 0, "name": None, "sim": 0.0,
                    "best_sim": 0.0, "votes": {}, "checked_in": False,
                })
                self.next_id += 1

            self.tracks = [t for t in self.tracks if t["miss"] <= self.max_miss]

            # ③ 输出前去重：老轨迹还没超时、新轨迹又起来时，只留更可信的那个
            active = sorted((t for t in self.tracks if t["hits"] >= self.min_hits),
                            key=lambda t: (t["hits"], t["conf"]), reverse=True)
            out = []
            for t in active:
                if any(self._iou(t["box"], o["box"]) > self.dup_iou for o in out):
                    continue
                out.append(t)
            return out


# ---------------------------------------------------------------- 检测
def _mask_geometry(mask, box, frame_shape):
    """由实例掩码算「贴轮廓的收紧框 + 轮廓点」；不可靠时返回 None 回退原框。

    YOLO 的检测框是矩形，常把人身边的背景、旁边的人一起包进去。
    分割掩码才是人的实际形状，用它的最大外轮廓收紧能明显减少多余区域。
    保护：掩码太碎（只框到半个人）或收紧过头时不采用，宁可用原框。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    area_box = max(1.0, (x2 - x1) * (y2 - y1))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < area_box * 0.12:
        return None
    bx, by, bw, bh = cv2.boundingRect(c)
    if bw < (x2 - x1) * 0.35 or bh < (y2 - y1) * 0.35:
        return None
    pad = 2
    H, W = frame_shape[:2]
    # 收紧框只能是原框的子集；允许向外扩会把框撑大，反而包进更多背景
    nx1 = max(x1, max(0, bx - pad))
    ny1 = max(y1, max(0, by - pad))
    nx2 = min(x2, min(W, bx + bw + pad))
    ny2 = min(y2, min(H, by + bh + pad))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    if (nx2 - nx1) * (ny2 - ny1) > area_box:
        return None
    contour = cv2.approxPolyDP(c, 2.0, True)
    if len(contour) < 3:
        return None
    return (nx1, ny1, nx2, ny2, contour)


def _person_masks(result, frame_shape):
    """把 masks.data（模型输入分辨率）缩回原图坐标。

    这一步不能省：masks.xy / masks.data 都是 640 输入尺寸下的坐标，
    直接拿来画会错位（旧项目踩过同一个坑）。
    """
    if getattr(result, "masks", None) is None or result.masks.data is None:
        return []
    try:
        from ultralytics.utils import ops
        m = result.masks.data[None].float()
        scaled = ops.scale_masks(m, frame_shape[:2])[0]
        out = []
        for i in range(scaled.shape[0]):
            t = scaled[i]
            arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            out.append((arr > 0.5).astype(np.uint8))
        return out
    except Exception:
        return []


def detect_persons(model, frame, conf, tight=True):
    """返回 [x1,y1,x2,y2,score,contour]；contour 可能是 None。

    tight=True 时用分割掩码收紧框，并附带轮廓点供画框阶段描边。
    """
    r = model.predict(frame, conf=conf, classes=[0], iou=0.5, verbose=False)[0]
    if r.boxes is None:
        return []
    boxes = [b.tolist() for b in r.boxes.xyxy]
    confs = [float(c) for c in r.boxes.conf]
    masks = _person_masks(r, frame.shape) if tight else []
    out = []
    for i, b in enumerate(boxes):
        contour = None
        x1, y1, x2, y2 = b
        if i < len(masks):
            geo = _mask_geometry(masks[i], b, frame.shape)
            if geo is not None:
                x1, y1, x2, y2, contour = geo
        out.append([x1, y1, x2, y2, confs[i], contour])
    return dedup_boxes(out)


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
                  id_threshold=None, max_frames=0, out_dir=None,
                  max_width=0, max_seconds=0):
    """后台线程里跑：逐帧检测 → 追踪 → 人脸比对 → 分色画框 → 存 JPEG + json。"""
    if id_threshold is None:
        id_threshold = default_threshold()
    out_dir = Path(out_dir) if out_dir else OUT_ROOT / job_id
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()

    mode = identity_mode()
    fi = FaceIdentity() if mode == "face" else None
    margin = FACE_MARGIN if mode == "face" else 0.05
    tracker = SimpleTracker(max_miss=max(2, 6 * stride), min_hits=1)
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

    writer = cv2.VideoWriter(str(out_dir / "result.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    per_frame, frames_out = [], []
    id_names = {}
    last_boxes = []
    overlap_pairs = 0          # 同帧任意两框 IOU>0.5 的次数，用来量化「重框」有没有修掉
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
            tracks = tracker.update(dets, dt=stride)
            if not dets:
                tracks = []          # 这一帧确实没检到人，不残留上一帧的框
            to_draw = []
            for t in tracks:
                # 身份：只在还没锁定时比对；锁定后即使某帧没检到脸也保持身份
                if gallery and t["name"] is None:
                    # 远景小目标常常连脸都检不到，连续 2 次后降频，别把 CPU 耗在空跑上；
                    # 一旦检到脸就恢复每帧比对，保证正面段能在 2 帧内锁定。
                    skip = (t.get("no_face", 0) >= 2 and (idx // stride) % 3 != 0)
                    f = None
                    if not skip:
                        f = (fi.feat(frame, t["box"], up=2.0) if mode == "face"
                             else appearance_feat(frame, t["box"]))
                        if f is None:
                            t["no_face"] = t.get("no_face", 0) + 1
                        else:
                            t["no_face"] = 0
                    if f is not None:
                        nm, best, _ = match_identity(f, gallery, mode,
                                                     id_threshold, margin)
                        t["best_sim"] = max(t.get("best_sim", 0.0), best)
                        if nm:
                            t["votes"][nm] = t["votes"].get(nm, 0) + 1
                            if t["votes"][nm] >= VOTE_MIN:
                                t["name"], t["sim"] = nm, t["best_sim"]
                                log_checkin(nm, t["id"], idx, t["best_sim"], mode)
                                t["checked_in"] = True
                if t["name"]:
                    id_names[t["id"]] = (t["name"], float(t.get("sim", 0)))
                to_draw.append((t["id"], t["box"], t["name"], t["sim"],
                                t.get("contour")))
            last_boxes = to_draw
        else:
            to_draw = last_boxes

        for i in range(len(to_draw)):
            for j in range(i + 1, len(to_draw)):
                if box_iou(to_draw[i][1], to_draw[j][1]) > 0.50:
                    overlap_pairs += 1

        vis = frame.copy()
        # 只给人形轮廓内部上淡色：矩形四角的背景不再被"框进去"，
        # 视觉上框就是贴着人的形状，多余区域一眼能看出不属于目标。
        overlay = vis.copy()
        has_contour = False
        for item in to_draw:
            if item[4] is not None:
                cv2.fillPoly(overlay, [item[4]],
                             COLOR_REGISTERED if item[2] else COLOR_PASSERBY)
                has_contour = True
        if has_contour:
            vis = cv2.addWeighted(overlay, 0.14, vis, 0.86, 0)

        for tid, box, name, sim, contour in to_draw:
            x1, y1, x2, y2 = [int(v) for v in box]
            color = COLOR_REGISTERED if name else COLOR_PASSERBY
            if contour is not None:
                cv2.polylines(vis, [contour], True, color, 2, cv2.LINE_AA)
                # 用四角标记代替完整矩形：保留"框"的定位感，
                # 又不会把人形轮廓以外的背景（桌脚、旁边的人）圈进框里
                L = max(10, min(26, int((x2 - x1) * 0.18)))
                for cx, cy, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                       (x1, y2, 1, -1), (x2, y2, -1, -1)):
                    cv2.line(vis, (cx, cy), (cx + sx * L, cy), color, 2, cv2.LINE_AA)
                    cv2.line(vis, (cx, cy), (cx, cy + sy * L), color, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"#{tid}  {name} {sim:.2f}" if name else f"#{tid}  路人"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(vis, label, (x1 + 4, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

    id_counter = {}
    for r in per_frame:
        for k in r["ids"]:
            id_counter[k] = id_counter.get(k, 0) + 1
    stable_ids = {k for k, n in id_counter.items() if n >= 4}
    id_switches, prev = 0, None
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
            "kind": "注册人员" if name else "路人",
            "sim": round(sim, 3), "checked_in": name is not None,
        })
    registered = sorted({t["name"] for t in track_summary if t["name"]})
    checkins = []
    if CHECKIN_LOG.exists():
        for line in CHECKIN_LOG.read_text(encoding="utf-8").splitlines():
            try:
                checkins.append(json.loads(line))
            except Exception:
                pass

    # 首屏默认帧：挑「框最多、且优先含已识别人员」的一帧，
    # 避免固定 35% 落在段间空场或没人的画面上（演示第一眼很重要）
    best_frame = 0
    if per_frame:
        mx = max(r["n"] for r in per_frame)
        cands = [i for i, r in enumerate(per_frame) if r["n"] == mx]
        named = [i for i in cands if per_frame[i]["names"]]
        pick = named or cands
        best_frame = pick[len(pick) // 2]

    result = {
        "job": job_id, "video": Path(video_path).name,
        "best_frame": best_frame,
        "mode": mode, "threshold": id_threshold, "margin": margin,
        "stride": stride, "conf": conf,
        "fps": round(fps, 2), "width": W, "height": H,
        "frames_total": idx, "frames_saved": len(frames_out),
        "frame_files": frames_out,
        "tracks": track_summary,
        "track_count": len(track_summary),
        "registered": registered,
        "registered_count": len(registered),
        "passerby_count": len(track_summary) - len({t["id"] for t in track_summary if t["name"]}),
        "max_track_frames": max([t["frames"] for t in track_summary], default=0),
        "id_switches": id_switches,
        "overlap_pairs": overlap_pairs,
        "checkins": checkins[-30:],
        "verdict": _verdict(track_summary, idx, id_switches, mode, registered),
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    set_state(running=False, done=idx, total=idx, message="完成", result=result,
              job=job_id)
    return result


def _verdict(tracks, n_frames, id_switches=0, mode="face", registered=None):
    registered = registered or []
    if not tracks:
        return "没有稳定框住任何目标：调低置信度，或换更清晰的画面"
    longest = max(t["frames"] for t in tracks)
    mean = sum(t["frames"] for t in tracks) / len(tracks)
    n_reg = len({t["id"] for t in tracks if t["name"]})
    who = (f"注册人员 {len(registered)} 人（{', '.join(registered)}），"
           f"其余 {len(tracks) - n_reg} 个 ID 判为路人"
           if registered else
           f"未发现注册库中的人员，{len(tracks)} 个 ID 全部判为路人（蓝框）") if mode == "face" \
        else "外观模式，身份结论不可靠"
    if len(tracks) == 1 and longest / max(1, n_frames) >= 0.6:
        return f"单目标场景：1 个 ID 持续 {longest}/{n_frames} 帧不换号；{who}"
    if len(tracks) > 1 and mean >= 8 and id_switches <= len(tracks) * 2:
        return (f"多目标场景：{len(tracks)} 个 ID、平均 {mean:.0f} 帧、最长 {longest} 帧，"
                f"ID 切换 {id_switches} 次；{who}")
    return (f"追踪不稳定：{len(tracks)} 个 ID、平均 {mean:.0f} 帧、"
            f"ID 切换 {id_switches} 次；{who}")


def make_demo_video(images_dir, out_path=None, per_image=32, fps=12, gap=60):
    """用现有注册照造一段「简单场景」演示视频：每人一段，段间留空场。

    gap 必须明显大于 6*stride 帧，否则追踪器会把「上一个人离开」当成短暂丢失，
    下一个人进来时接着用旧 ID —— 身份也会被旧 ID 带错。
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
