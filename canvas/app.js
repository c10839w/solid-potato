/* 人机协同图像标注系统 —— 前端交互
   纯原生 JS + Canvas，不依赖任何外部库，断网也能用 */

const PALETTE = ['#E24B4A', '#378ADD', '#639922', '#EF9F27', '#7F77DD',
                 '#1D9E75', '#D4537E', '#BA7517', '#888780', '#0C447C'];

const S = {
  names: [], zh: [],
  images: [], idx: -1,
  cur: null,          // 当前图的标注 {image,width,height,objects[]}
  mode: 'seg',        // det 目标检测 / seg 实例分割 / sem 语义分割
  sel: -1,            // 选中的物体下标
  img: null,
  scale: 1,           // 图片像素 -> 屏幕像素
  drag: null,
  adding: false,
  newClass: 0,
  fixes: [],          // 还没提交的修正记录
  dirty: false,
  cat: 'all',         // 左侧照片类别筛选：all / 客厅 / 卧室 ...
};

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const stage = document.getElementById('stage');

/* ---------------------------------------------------------------- 小工具 */
const $ = id => document.getElementById(id);

function bboxOf(ob) {
  let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
  for (const [x, y] of ob.polygon) {
    if (x < x1) x1 = x; if (y < y1) y1 = y;
    if (x > x2) x2 = x; if (y > y2) y2 = y;
  }
  return [x1, y1, x2, y2];
}

function pointInPoly(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p.y) !== (yj > p.y) && p.x < ((xj - xi) * (p.y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function status(msg) { $('status').textContent = msg; }

/* ---------------------------------------------------------------- 启动 */
async function init() {
  const cfg = await (await fetch('/api/config')).json();
  S.names = cfg.names; S.zh = cfg.zh;

  const sel = $('classSel');
  sel.innerHTML = S.names.map((n, i) => `<option value="${i}">${cfg.zh[i]}（${n}）</option>`).join('');

  bindUI();
  await loadImages();
  refreshTips();

  // 从封面页上传后跳过来：?pick=文件名 → 自动选中并跑一遍模型
  const pick = new URLSearchParams(location.search).get('pick');
  if (pick) {
    const i = S.images.findIndex(it => it.name === pick);
    if (i >= 0) {
      await selectImage(i);
      status(`${pick} 已就绪，正在识别……`);
      prelabel([pick]);
    }
  }
}

function bindUI() {
  $('modes').addEventListener('click', e => {
    const b = e.target.closest('.mode');
    if (!b) return;
    S.mode = b.dataset.mode;
    document.querySelectorAll('.mode').forEach(x => x.classList.toggle('active', x === b));
    refreshTips();
    draw();
  });

  cv.addEventListener('mousedown', onDown);
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
  window.addEventListener('keydown', onKey);
  window.addEventListener('resize', () => { fitCanvas(); draw(); });

  $('classSel').addEventListener('change', e => {
    const ob = curObj(); if (!ob) return;
    const before = S.names[ob.class_id], after = S.names[+e.target.value];
    ob.class_id = +e.target.value;
    ob.source = 'human';
    record('class', ob, before, after);
    renderObjectList(); draw(); markDirty();
  });

  $('btnDel').addEventListener('click', delSelected);
  $('btnAdd').addEventListener('click', () => {
    S.adding = true;
    $('btnAdd').classList.add('primary');
    status('新增模式：在图上按住拖出一个框（Esc 取消）');
  });
  $('btnSave').addEventListener('click', () => save(true));
  $('btnPrelabel').addEventListener('click', () => prelabel([S.cur && S.cur.image].filter(Boolean)));
  $('btnPrelabelAll').addEventListener('click', () => prelabel([]));
  $('btnExport').addEventListener('click', doExport);
  $('btnUpload').addEventListener('click', () => $('fileUpload').click());
  $('fileUpload').addEventListener('change', uploadImage);
  $('btnExportImg').addEventListener('click', exportImages);

  // 左侧类别筛选
  $('chips').addEventListener('click', e => {
    const b = e.target.closest('.chipbtn');
    if (!b) return;
    S.cat = b.dataset.cat;
    document.querySelectorAll('#chips .chipbtn').forEach(x => x.classList.toggle('on', x === b));
    renderImageList();
  });

  // 改这张照片的类别（客厅/卧室/…），立刻存到服务端，照片墙的筛选就靠它
  $('catSel').addEventListener('change', async e => {
    if (!S.cur) return;
    const cat = e.target.value;
    S.cur.category = cat;
    if (S.idx >= 0 && S.images[S.idx]) S.images[S.idx].category = cat;
    renderImageList();
    await fetch('/api/category', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: S.cur.image, category: cat }),
    });
    status(`已把这张归到「${cat}」`);
  });
}

function refreshTips() {
  const map = {
    det: ['框由轮廓自动算出，拖动四个角可缩放',
          '拖动框内部可整体移动',
          '改轮廓请切到「实例分割」'],
    seg: ['点物体选中，拖动轮廓顶点调整',
          '拖动物体内部整体移动',
          '点「新增物体」可手动补漏标的'],
    sem: ['按类别合并显示，仅用于查看效果',
          '要修改请切回「实例分割」'],
  };
  $('tips').innerHTML = map[S.mode].map(t => `<li>${t}</li>`).join('');
}


/* 修正记录显示：带上任务书要求的 60 处目标，随时知道还差多少 */
const FIX_TARGET = 60;
function renderFixInfo(n, pending) {
  const left = Math.max(0, FIX_TARGET - n);
  const tail = n >= FIX_TARGET ? '已达标' : `还差 ${left} 处`;
  const extra = pending ? `（未保存 ${pending}）` : '';
  const el = $('fixInfo');
  el.textContent = `修正 ${n}/${FIX_TARGET} 处 · ${tail}${extra}`;
  el.className = n >= FIX_TARGET ? 'done' : 'muted';
}

/* ---------------------------------------------------------------- 图片列表 */
async function loadImages() {
  const data = await (await fetch('/api/images')).json();
  S.images = data.images;
  S.fixTotal = data.fixes;
  renderImageList();
  renderFixInfo(data.fixes, 0);
  if (S.images.length && S.idx < 0) selectImage(0);
}

function renderImageList() {
  const ul = $('imgList');
  const rows = S.images
    .map((it, i) => ({ it, i }))
    .filter(({ it }) => S.cat === 'all' || (it.category || '其他') === S.cat);

  $('imgCount').textContent = rows.length + ' / ' + S.images.length + ' 张';
  ul.innerHTML = rows.map(({ it, i }) => {
    const cls = it.human > 0 ? 'human' : (it.objects > 0 ? 'has' : '');
    const meta = it.objects > 0
      ? `${it.category || '其他'} · ${it.objects} 个物体`
      : `${it.category || '其他'} · 未标注`;
    return `<li data-i="${i}" class="${i === S.idx ? 'active' : ''}">
      <img src="/api/img/${encodeURIComponent(it.name)}" loading="lazy" alt="">
      <div style="min-width:0">
        <div class="img-name" title="${it.name}">${it.name}</div>
        <div class="img-meta">${meta}</div>
      </div>
      <span class="dot ${cls}"></span>
    </li>`;
  }).join('');
  ul.querySelectorAll('li').forEach(li => li.addEventListener('click', () => selectImage(+li.dataset.i)));
}

async function selectImage(i) {
  if (i === S.idx && S.cur) return;
  if (S.dirty) await save(false);
  S.idx = i;
  S.sel = -1;
  const name = S.images[i].name;
  S.cur = await (await fetch('/api/annotation/' + encodeURIComponent(name))).json();
  S.img = await loadImg('/api/img/' + encodeURIComponent(name));
  $('catSel').value = S.cur.category || '其他';
  document.querySelectorAll('#imgList li').forEach(li =>
    li.classList.toggle('active', +li.dataset.i === i));
  $('hint').style.display = 'none';
  fitCanvas(); draw(); renderObjectList(); updateObjBox();
  status(`${name}　${S.cur.width}×${S.cur.height}　${S.cur.objects.length} 个物体`);
}

function loadImg(src) {
  return new Promise(res => { const im = new Image(); im.onload = () => res(im); im.src = src; });
}

/* ---------------------------------------------------------------- 画布 */
function fitCanvas() {
  if (!S.img) return;
  const pad = 24;
  const w = stage.clientWidth - pad, h = stage.clientHeight - pad;
  S.scale = Math.min(w / S.img.naturalWidth, h / S.img.naturalHeight, 1);
  const dpr = window.devicePixelRatio || 1;
  const cssW = S.img.naturalWidth * S.scale, cssH = S.img.naturalHeight * S.scale;
  cv.style.width = cssW + 'px'; cv.style.height = cssH + 'px';
  cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr * S.scale, 0, 0, dpr * S.scale, 0, 0);
}

function draw() {
  if (!S.img || !S.cur) return;
  const W = S.cur.width, H = S.cur.height;
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(S.img, 0, 0, W, H);

  const lw = 1.6 / S.scale;
  const fs = 12 / S.scale;

  S.cur.objects.forEach((ob, i) => {
    const color = PALETTE[ob.class_id % PALETTE.length];
    const sel = i === S.sel;

    if (S.mode === 'sem') {
      ctx.globalAlpha = 0.42;
      ctx.fillStyle = color;
      ctx.beginPath();
      ob.polygon.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 1;
      return;
    }

    ctx.beginPath();
    ob.polygon.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath();

    if (S.mode === 'seg') {
      ctx.globalAlpha = sel ? 0.26 : 0.14;
      ctx.fillStyle = color; ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color; ctx.lineWidth = sel ? lw * 2 : lw; ctx.stroke();
    } else {
      const [x1, y1, x2, y2] = bboxOf(ob);
      ctx.strokeStyle = color;
      ctx.lineWidth = sel ? lw * 2 : lw;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    // 类别标签
    const [bx1, by1] = bboxOf(ob);
    const label = S.zh[ob.class_id] + (ob.source === 'human' ? ' *' : '');
    ctx.font = `500 ${fs}px "Microsoft YaHei", sans-serif`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(bx1, Math.max(0, by1 - fs * 1.35), tw + fs * 0.5, fs * 1.3);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, bx1 + fs * 0.25, Math.max(fs, by1 - fs * 0.28));

    // 选中的物体：画出可拖动的控制点
    if (sel) {
      const r = 4 / S.scale;
      ctx.fillStyle = '#fff'; ctx.strokeStyle = color; ctx.lineWidth = lw * 1.2;
      if (S.mode === 'seg') {
        ob.polygon.forEach(([x, y]) => {
          ctx.beginPath(); ctx.rect(x - r, y - r, r * 2, r * 2); ctx.fill(); ctx.stroke();
        });
      } else {
        const [x1_, y1_, x2_, y2_] = bboxOf(ob);
        [[x1_, y1_], [x2_, y1_], [x1_, y2_], [x2_, y2_]].forEach(([x, y]) => {
          ctx.beginPath(); ctx.rect(x - r, y - r, r * 2, r * 2); ctx.fill(); ctx.stroke();
        });
      }
    }
  });

  // 新增时的临时框
  if (S.drag && S.drag.type === 'new') {
    const { a, b } = S.drag;
    ctx.strokeStyle = '#185FA5'; ctx.lineWidth = lw * 1.5;
    ctx.setLineDash([6 / S.scale, 4 / S.scale]);
    ctx.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(a.x - b.x), Math.abs(a.y - b.y));
    ctx.setLineDash([]);
  }
}

/* ---------------------------------------------------------------- 命中检测 */
function toImg(evt) {
  const r = cv.getBoundingClientRect();
  return { x: (evt.clientX - r.left) / S.scale, y: (evt.clientY - r.top) / S.scale };
}

function hitTest(p) {
  const tol = 8 / S.scale;
  // 先看选中物体的控制点（优先于物体本体，方便拖）
  const ob = curObj();
  if (ob) {
    if (S.mode === 'seg') {
      for (let k = 0; k < ob.polygon.length; k++) {
        const [x, y] = ob.polygon[k];
        if (Math.abs(p.x - x) < tol && Math.abs(p.y - y) < tol) return { kind: 'vertex', vi: k };
      }
    } else {
      const [x1, y1, x2, y2] = bboxOf(ob);
      const corners = [[x1, y1], [x2, y1], [x1, y2], [x2, y2]];
      for (let k = 0; k < 4; k++) {
        if (Math.abs(p.x - corners[k][0]) < tol && Math.abs(p.y - corners[k][1]) < tol)
          return { kind: 'corner', ci: k };
      }
    }
  }
  // 再看是哪个物体（后画的在上层，所以从后往前找）
  for (let i = S.cur.objects.length - 1; i >= 0; i--) {
    const o = S.cur.objects[i];
    if (S.mode === 'det') {
      const [x1, y1, x2, y2] = bboxOf(o);
      if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) return { kind: 'obj', i };
    } else if (pointInPoly(p, o.polygon)) return { kind: 'obj', i };
  }
  return { kind: 'empty' };
}

/* ---------------------------------------------------------------- 鼠标 */
function onDown(evt) {
  if (!S.cur || !S.img) return;
  const p = toImg(evt);

  if (S.adding) {
    S.drag = { type: 'new', a: p, b: p };
    return;
  }

  const hit = hitTest(p);

  if (hit.kind === 'vertex') {
    const ob = curObj();
    S.drag = { type: 'vertex', vi: hit.vi, before: ob.polygon.map(q => [...q]) };
    return;
  }
  if (hit.kind === 'corner') {
    const ob = curObj();
    S.drag = { type: 'corner', ci: hit.ci, before: ob.polygon.map(q => [...q]), oldBox: bboxOf(ob), start: p };
    return;
  }
  if (hit.kind === 'obj') {
    S.sel = hit.i;
    const ob = curObj();
    S.drag = { type: 'move', before: ob.polygon.map(q => [...q]), start: p };
    renderObjectList(); updateObjBox(); draw();
    return;
  }
  S.sel = -1;
  renderObjectList(); updateObjBox(); draw();
}

function onMove(evt) {
  if (!S.drag || !S.cur) return;
  const p = toImg(evt);
  const ob = curObj();

  if (S.drag.type === 'new') {
    S.drag.b = p; draw(); return;
  }
  if (S.drag.type === 'move') {
    const dx = p.x - S.drag.start.x, dy = p.y - S.drag.start.y;
    ob.polygon = S.drag.before.map(([x, y]) => [x + dx, y + dy]);
  } else if (S.drag.type === 'vertex') {
    ob.polygon[S.drag.vi] = [p.x, p.y];
  } else if (S.drag.type === 'corner') {
    const [x1, y1, x2, y2] = S.drag.oldBox;
    const nb = [x1, y1, x2, y2];
    if (S.drag.ci === 0) { nb[0] = p.x; nb[1] = p.y; }
    if (S.drag.ci === 1) { nb[2] = p.x; nb[1] = p.y; }
    if (S.drag.ci === 2) { nb[0] = p.x; nb[3] = p.y; }
    if (S.drag.ci === 3) { nb[2] = p.x; nb[3] = p.y; }
    const sx = (nb[2] - nb[0]) / Math.max(1e-6, x2 - x1);
    const sy = (nb[3] - nb[1]) / Math.max(1e-6, y2 - y1);
    ob.polygon = S.drag.before.map(([x, y]) => [nb[0] + (x - x1) * sx, nb[1] + (y - y1) * sy]);
  }
  draw();
}

function onUp(evt) {
  const d = S.drag;
  if (!d) return;
  S.drag = null;

  if (d.type === 'new') {
    const p = toImg(evt);
    const x1 = Math.min(d.a.x, p.x), y1 = Math.min(d.a.y, p.y);
    const x2 = Math.max(d.a.x, p.x), y2 = Math.max(d.a.y, p.y);
    S.adding = false; $('btnAdd').classList.remove('primary');
    if (x2 - x1 > 4 && y2 - y1 > 4) {
      const ob = {
        id: 'h' + Date.now(), class_id: S.newClass, source: 'human', conf: 1,
        polygon: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
      };
      S.cur.objects.push(ob);
      S.sel = S.cur.objects.length - 1;
      record('add', ob, '', S.names[ob.class_id]);
      renderObjectList(); updateObjBox(); markDirty();
    }
    status('已退出新增模式');
    draw();
    return;
  }

  const ob = curObj();
  if (ob && d.before) {
    const changed = ob.polygon.some(([x, y], k) =>
      Math.abs(x - d.before[k][0]) > 0.5 || Math.abs(y - d.before[k][1]) > 0.5);
    if (changed) {
      ob.source = 'human';
      record(d.type, ob, d.before.map(q => q.map(v => Math.round(v))), ob.polygon.map(q => q.map(v => Math.round(v))));
      markDirty(); renderObjectList();
    }
  }
  draw();
}

function onKey(e) {
  if (e.key === 'Escape') {
    S.adding = false; $('btnAdd').classList.remove('primary'); S.drag = null; draw();
    status('已取消');
  }
  if ((e.key === 'Delete' || e.key === 'Backspace') && S.sel >= 0 && document.activeElement.tagName !== 'SELECT') {
    e.preventDefault(); delSelected();
  }
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { if (S.idx < S.images.length - 1) selectImage(S.idx + 1); }
  if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { if (S.idx > 0) selectImage(S.idx - 1); }
}

function delSelected() {
  const ob = curObj(); if (!ob) return;
  record('delete', ob, S.names[ob.class_id], '');
  S.cur.objects.splice(S.sel, 1);
  S.sel = -1;
  renderObjectList(); updateObjBox(); markDirty(); draw();
}

/* ---------------------------------------------------------------- 修正记录 */
function curObj() { return (S.cur && S.sel >= 0) ? S.cur.objects[S.sel] : null; }

function record(action, ob, before, after) {
  S.fixes.push({
    image: S.cur.image, obj_id: ob.id || '', action,
    class_id: ob.class_id, before, after,
  });
  S.fixTotal += 1;
  renderFixInfo(S.fixTotal, S.fixes.length);
}

function markDirty() { S.dirty = true; }

/* ---------------------------------------------------------------- 保存 */
async function save(showMsg) {
  if (!S.cur) return;
  const payload = { ...S.cur, _fixes: S.fixes };
  const res = await (await fetch('/api/annotation/' + encodeURIComponent(S.cur.image), {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })).json();
  S.fixes = []; S.dirty = false;
  S.fixTotal = res.fixes;
  renderFixInfo(res.fixes, 0);
  if (showMsg) status('已保存　' + new Date().toLocaleTimeString());
}

/* ---------------------------------------------------------------- 右侧面板 */
function renderObjectList() {
  const ul = $('objList');
  if (!S.cur) { ul.innerHTML = ''; return; }
  ul.innerHTML = S.cur.objects.map((ob, i) => `
    <li data-i="${i}" class="${i === S.sel ? 'active' : ''}">
      <span class="chip" style="background:${PALETTE[ob.class_id % PALETTE.length]}"></span>
      <span>${S.zh[ob.class_id]}</span>
      <span class="muted">${ob.source === 'human' ? '人工改过' : '模型草稿'}</span>
    </li>`).join('');
  ul.querySelectorAll('li').forEach(li => li.addEventListener('click', () => {
    S.sel = +li.dataset.i;
    renderObjectList(); updateObjBox(); draw();
  }));
}

function updateObjBox() {
  const ob = curObj();
  $('classRow').style.display = ob ? 'flex' : 'none';
  $('delRow').style.display = ob ? 'flex' : 'none';
  if (!ob) { $('objBox').textContent = '没有选中物体'; return; }
  $('objBox').textContent = `${S.zh[ob.class_id]}　置信度 ${ob.conf ?? '-'}　${ob.source === 'human' ? '人工修改' : '模型草稿'}`;
  $('classSel').value = ob.class_id;
  S.newClass = ob.class_id;
}

/* ---------------------------------------------------------------- 上传识别 */
/* 答辩演示用：现场拍一张图，上传后立刻用模型标一遍 */
async function uploadImage(e) {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  status(`正在上传 ${f.name} ……`);
  const fd = new FormData();
  fd.append('file', f);
  fd.append('category', $('catSel').value);   // 照片类别（客厅/卧室/…）
  let res;
  try {
    res = await (await fetch('/api/upload', { method: 'POST', body: fd })).json();
  } catch (err) {
    status('上传失败：' + err); return;
  } finally {
    e.target.value = '';             // 允许同一张图再传一次
  }
  if (!res.ok) { status(res.message || '上传失败'); return; }

  await loadImages();
  const i = S.images.findIndex(it => it.name === res.name);
  if (i >= 0) await selectImage(i);
  status(`${res.name} 已上传，正在识别……`);
  prelabel([res.name]);              // 上传完直接跑模型
}

/* ---------------------------------------------------------------- 预标注 */
async function prelabel(names) {
  const res = await (await fetch('/api/prelabel', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ names }),
  })).json();
  if (!res.ok) { status(res.message); return; }
  status(`开始预标注 ${res.total} 张，请稍等……`);
  $('progress').style.display = 'flex';
  pollProgress();
}

/* ---------------------------------------------------------------- 导出数据集 */
function toast(text, isErr) {
  let el = document.getElementById('toast');
  if (!el) {
    el = document.createElement('div');
    el.id = 'toast';
    document.body.appendChild(el);
  }
  el.textContent = text;
  el.className = 'toast show' + (isErr ? ' err' : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove('show'), 8000);
}

async function doExport() {
  status('正在导出……');
  let res;
  try {
    res = await (await fetch('/api/export', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ val_ratio: 0.2 }),
    })).json();
  } catch (e) {
    status('导出失败：' + e);
    return;
  }
  if (!res.ok) {
    status(res.message || '导出失败');
    toast(res.message || '导出失败', true);
    return;
  }
  const hit = Object.entries(res.per_class).filter(([, c]) => c > 0)
    .map(([n, c]) => `${n} ${c}`).join('，');
  const miss = Object.entries(res.per_class).filter(([, c]) => c === 0).map(([n]) => n);
  toast(`导出完成　${res.images_total} 张（训练 ${res.train} / 验证 ${res.val}）\n`
    + `物体 ${res.objects} 个　用时 ${res.seconds} 秒\n${hit}`
    + (miss.length ? `\n还没标到：${miss.join('，')}` : '')
    + `\n→ D:\\imglabel\\dataset`);
  status(`导出完成：${res.images_total} 张 / ${res.objects} 个物体`);
}

/* ---------------------------------------------------------------- 导出标注照片 */
/* 把标过的图连框带标签烧进图片，打包成 zip 下载（给不装 Python 的人看结果用） */
async function exportImages() {
  const btn = $('btnExportImg');
  const old = btn.textContent;
  btn.textContent = '打包中……'; btn.disabled = true;
  status('正在把标注画到照片上并打包……');
  try {
    const r = await fetch('/api/export_images', { method: 'POST' });
    if (!r.ok) throw new Error('服务端 ' + r.status);
    const n = r.headers.get('X-Image-Count') || '?';
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = '标注照片.zip';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    toast(`已打包 ${n} 张标注照片\n打开 zip 就能看到画好框的图，旁边还有同名 .json 标注`);
    status(`已导出 ${n} 张标注照片`);
    btn.textContent = '已导出 ✓';
  } catch (e) {
    status('导出失败：' + e.message);
    toast('导出失败：' + e.message, true);
    btn.textContent = '导出失败';
  }
  setTimeout(() => { btn.textContent = old; btn.disabled = false; }, 2600);
}

async function pollProgress() {  const p = await (await fetch('/api/progress')).json();
  $('barFill').style.width = (p.total ? (p.done / p.total * 100) : 0) + '%';
  $('progressText').textContent = p.running
    ? `${p.done}/${p.total}　${p.current || ''}`
    : (p.message || '完成');

  if (p.running) { setTimeout(pollProgress, 900); return; }
  $('progress').style.display = 'none';
  await loadImages();
  if (S.idx >= 0) {
    const name = S.images[S.idx] && S.images[S.idx].name;
    if (name) {
      const fresh = await (await fetch('/api/annotation/' + encodeURIComponent(name))).json();
      if (fresh.image === S.cur.image) { S.cur = fresh; S.sel = -1; }
      draw(); renderObjectList(); updateObjBox();
    }
  }
}

init();
