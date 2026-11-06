/* 人员打卡 / 同场景追踪 —— 页面逻辑 */
const $ = id => document.getElementById(id);
const S = {
  job: '', frames: 0, idx: 0, timer: null, playing: false, info: null,
  gallery: [], preferredVideo: '', userPickedVideo: false, pollTries: 0,
  fix: null, selected: new Set(), dragFiles: [], suspectByPerson: {}, moving: false,
};

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function status(t) { $('statusText').textContent = t; }
function setBar(p) { $('barFill').style.width = Math.max(0, Math.min(100, p)) + '%'; }

/* ------------------------------------------------ 视频素材列表 */
function renderVideos(videos, prefer) {
  const sel = $('videoSel');
  const want = prefer || S.preferredVideo || sel.value;
  sel.innerHTML = videos.length
    ? videos.map(v => `<option value="${esc(v.name)}">${esc(v.name)} · ${v.width}×${v.height} · ${v.seconds}s · ${v.size_mb}MB</option>`).join('')
    : '<option value="">（没有找到视频，请先上传或把 mp4 放进 videos 目录）</option>';
  // 优先保留「用户刚上传 / 刚选的」那个；否则默认选中 *_ck 打卡机位
  if (want && [...sel.options].some(o => o.value === want)) sel.value = want;
  else {
    const ck = videos.find(v => v.name.includes('_ck'));
    if (ck) sel.value = ck.name;
  }
  S.preferredVideo = sel.value;
}

async function loadInfo(prefer) {
  const r = await (await fetch('/api/track/info')).json();
  S.info = r;
  renderVideos(r.videos || [], prefer);

  const face = r.mode === 'face';
  const modeTxt = face
    ? `人脸级识别（YuNet + SFace 已加载，阈值 ${r.threshold}）`
    : '外观特征（离线临时方案，身份结果不可靠）';
  $('modeNote').innerHTML = `<b>当前身份模式：${modeTxt}</b>　·　` +
    `<i class="swatch" style="background:#d2453c"></i>红框只给注册库里的人，` +
    `<i class="swatch" style="background:#3d7fbf"></i>蓝框是路人。` +
    `人员检测用 COCO 预训练的 yolo26n-seg，追踪用自写 IOU 追踪器；` +
    `重复框已在「检测 / 建轨 / 输出」三处去重。全程离线、无需联网。`;

  renderGallery(r.gallery, r.disk_photos);
}

/* ------------------------------------------------ 注册库：点编号看正面照 */
const ANGLE = { front: '正面', side: '侧面', back: '背面', other: '其他' };

function renderGallery(view, disk) {
  S.gallery = view || [];
  const n = S.gallery.length;
  const templates = S.gallery.reduce((a, g) => a + (g.photos || 0), 0);
  $('galCount').textContent = `（${n} 人 · 磁盘已保存 ${disk || 0} 张 · ${templates} 张有效模板）`;
  $('galList').innerHTML = n
    ? S.gallery.map(g => {
        const warn = S.suspectByPerson[g.name] || 0;
        return `<button class="chip" data-name="${esc(g.name)}" title="点开看 ${esc(g.name)} 的正面照">
        <b>${esc(g.name)}</b><small>${g.photos} 张模板 · 共 ${g.saved} 张</small>${warn ? `<small class="chip-warn" title="有 ${warn} 张照片疑似不是本人">⚠${warn}</small>` : ''}</button>`;
      }).join('')
    : '还没有注册人员。点「上传注册人员照片」，或用现有注册照重建注册库。';
  $('galTip').textContent = n
    ? '点蓝色编号 → 看这个人的正面照；卡片里可切换他 / 她的全部注册照。'
    : '上传后，这里会列出人员编号，点击即可查看正面照。';
}

function photoUrl(name, w) { return `/api/track/photo/${encodeURIComponent(name)}?w=${w || 520}`; }

function shotNote(g, file) {
  const isFront = /front|正面|正脸/i.test(file || '');
  return `正在显示：${file}${isFront ? '（正面照）' : '（非正面照，点右侧缩略图可切换）'}` +
         `。标「仅存档」的背面 / 不合格照片不参与识别；${g.name} 共 ${g.saved} 张照片、${g.photos} 张有效模板。`;
}

function openPerson(name) {
  const g = S.gallery.find(x => x.name === name);
  if (!g) return;
  const shots = g.shots || [];
  const front = g.front || g.anchor || (shots[0] && shots[0].name) || '';
  $('pmTitle').textContent = `已登记人员 ${g.name}`;
  $('pmMeta').textContent = `有效模板 ${g.photos} 张 · 共保存 ${g.saved} 张 · 主模板：${g.anchor || '—'}`;
  $('pmPhoto').src = front ? photoUrl(front, 640) : '';
  $('pmPhoto').alt = `${g.name} 的正面注册照`;
  $('pmShots').innerHTML = shots.length
    ? shots.map(s => `<button class="pm-shot${s.name === front ? ' on' : ''}" data-file="${esc(s.name)}" title="${esc(s.name)}">
        <img src="${photoUrl(s.name, 220)}" alt="${esc(s.name)}" loading="lazy">
        <span>${ANGLE[s.angle] || '照片'}${s.anchor ? ' · 主' : ''}${s.used ? ' · 有效' : ' · 仅存档'}</span></button>`).join('')
    : '<span class="muted">（这个人还没有可显示的照片）</span>';
  $('pmNote').textContent = front
    ? shotNote(g, front)
    : '没有找到照片，请重新上传或重建注册库。';
  $('personModal').hidden = false;
}
function closePerson() { $('personModal').hidden = true; }

/* ------------------------------------------------ 未进模板：分类摘要 + 折叠明细 */
const REJECT_KIND = {
  back: { label: '背面照 · 仅存档', tip: '背面照没有人脸，本来就只保存、不参与识别，属于正常情况。' },
  face: { label: '检不到合格的脸', tip: '人脸太小 / 低头 / 遮挡 / 模糊；补一张平视、光线均匀、人脸 ≥80×80px 的正面照。' },
  diff: { label: '与本人正面照不一致', tip: '可能是别人、背景假脸或偏转超过 30°；用打卡机位重拍 ±15° 内的正面照。' },
  other: { label: '其他', tip: '文件打不开或提不到特征，换一张 jpg / png 重传。' },
};
function rejectKind(reason) {
  const r = reason || '';
  if (r.includes('背面')) return 'back';
  if (r.includes('不像') || r.includes('不一致')) return 'diff';
  if (/px|人脸|检出|检不出|特征|打不开|没有人脸/.test(r)) return 'face';
  return 'other';
}
function shortReason(reason) {
  const r = reason || '';
  const m = r.match(/（([\d.]+)）/);
  if (m) return `与本人相似度 ${m[1]}，偏低`;
  const p = r.match(/仅\s*(\d+)px/);
  if (p) return `人脸只有 ${p[1]}px，太小`;
  return r.replace(/（需.*?）/, '').replace(/，请.*$/, '').replace(/，可能.*$/, '');
}
function renderRejects(rej) {
  const box = $('regDetail');
  if (!rej || !rej.length) { box.innerHTML = ''; return; }
  const groups = { back: [], face: [], diff: [], other: [] };
  rej.forEach(x => groups[rejectKind(x.reason)].push(x));
  const keys = Object.keys(REJECT_KIND).filter(k => groups[k].length);
  const parts = keys.map(k => `<i>${REJECT_KIND[k].label} ${groups[k].length} 张</i>`).join('');
  const rows = keys.map(k => `
    <div class="rej-group"><b>${REJECT_KIND[k].label}</b>　${REJECT_KIND[k].tip}</div>
    <table class="rej-table"><tbody>${groups[k].map(x =>
      `<tr><td>${esc(x.file)}</td><td>${esc(shortReason(x.reason))}</td></tr>`).join('')}</tbody></table>`).join('');
  box.innerHTML = `<div class="rej-sum">共 <b>${rej.length} 张</b>照片未进模板（多数是背面照，属正常）：<br>${parts}</div>
    <details class="fold"><summary>展开逐张原因（${rej.length} 条）</summary><div class="fold-body">${rows}</div></details>`;
}

/* ------------------------------------------------ 人工修正分组（拖拽） */
const UNSORTED = 'unsorted';
const byNum = (a, b) => String(a).localeCompare(String(b), 'zh-CN', { numeric: true });

function updateSelCount() {
  $('fixSelCount').textContent = S.selected.size;
  document.querySelectorAll('.fix-photo').forEach(el =>
    el.classList.toggle('sel', S.selected.has(el.dataset.file)));
}

function fixColumn(person, list, g, kind) {
  const isNew = kind === 'new', isUnsorted = kind === 'unsorted';
  const title = isNew ? '＋ 新建编号' : (isUnsorted ? '待确认' : person);
  const count = isNew ? '' :
    `${list.length} 张${g ? ` · ${g.photos} 模板` : ' · 未进模板'}`;
  const tiles = isNew
    ? `<div class="fix-empty" style="padding:34px 6px">把照片拖到这里<br>自动开一个<br>新人员编号<br><b>${esc((S.fix && S.fix.next_person) || '')}</b></div>`
    : (list.length ? list.map(p => {
        const cls = ['fix-photo'];
        if (p.suspect) cls.push('suspect');
        if (p.dup) cls.push('dup');
        const badge = p.unsorted ? ['', '待确认']
          : p.dup ? ['dup', '重复']
          : p.anchor ? ['ok', '主模板']
          : p.used ? ['ok', '有效']
          : p.suspect ? ['warn', '可疑']
          : ['', '仅存档'];
        const tip = [p.label, `人脸 ${p.short}px`, p.score ? `置信 ${p.score}` : '',
                     p.best ? `最像 ${p.best}（${p.best_sim}）` : '',
                     p.suggest ? `建议移到 ${p.suggest}` : ''].filter(Boolean).join('；');
        const label = p.suggest
          ? `→ ${p.suggest} ${p.suggest_sim}` : (p.suspect ? '⚠ 可能不是本人' : p.label);
        return `<button type="button" class="${cls.join(' ')}" draggable="true"
            data-file="${esc(p.name)}" title="${esc(p.name)}｜${esc(tip)}">
          <img src="${photoUrl(p.name, 200)}" alt="${esc(p.name)}" loading="lazy" draggable="false">
          <span class="badge ${badge[0]}">${badge[1]}</span>
          <span class="cap">${esc(p.name.replace(/\.[^.]+$/, ''))}<br>${esc(label)}</span>
        </button>`;
      }).join('') : '<div class="fix-empty">（暂无照片，可把别的编号拖过来）</div>');
  const look = (!isNew && !isUnsorted && g)
    ? `<button type="button" class="look" data-look="${esc(person)}">看正面照</button>` : '';
  return `<div class="fix-col${isNew || isUnsorted ? ' special' : ''}" data-person="${esc(isNew ? '__new__' : isUnsorted ? '__unsorted__' : person)}">
      <div class="fix-hd"><b>${esc(title)}</b><small>${esc(count)}</small>${look}</div>
      <div class="fix-tiles">${tiles}</div>
    </div>`;
}

function renderFix() {
  const st = S.fix;
  if (!st) return;
  const photos = st.photos || [], gal = st.gallery || [];
  const byPerson = {};
  photos.forEach(p => { (byPerson[p.person] = byPerson[p.person] || []).push(p); });
  const names = [...new Set([...gal.map(g => g.name), ...Object.keys(byPerson)])]
    .filter(n => n !== UNSORTED).sort(byNum);

  const prev = $('fixTarget').value;
  $('fixTarget').innerHTML = names.map(n =>
    `<option value="${esc(n)}">${esc(n)}</option>`).join('') +
    `<option value="__unsorted__">待确认（不参与识别）</option>` +
    `<option value="__new__">＋ 新建编号 ${esc(st.next_person || '')}</option>`;
  if (prev && [...$('fixTarget').options].some(o => o.value === prev)) $('fixTarget').value = prev;

  $('fixCols').innerHTML = names.map(n =>
    fixColumn(n, byPerson[n] || [], gal.find(g => g.name === n))).join('') +
    fixColumn(UNSORTED, byPerson[UNSORTED] || [], null, 'unsorted') +
    fixColumn('', [], null, 'new');
  updateSelCount();
}

function applyFixState(r) {
  S.fix = r;
  S.selected.clear();
  S.suspectByPerson = {};
  (r.photos || []).forEach(p => {
    if (p.suspect && !p.unsorted)
      S.suspectByPerson[p.person] = (S.suspectByPerson[p.person] || 0) + 1;
  });
  renderGallery(r.gallery || [], r.disk_photos);
  renderFix();
  const n = (r.photos || []).length;
  $('fixSum').textContent = `· ${n} 张照片 / ${(r.gallery || []).length} 人`;
}

function renderFixRejects(rej) {
  const box = $('fixReject');
  if (!rej || !rej.length) { box.innerHTML = ''; return; }
  const kinds = {};
  rej.forEach(x => { const k = rejectKind(x.reason); (kinds[k] = kinds[k] || []).push(x); });
  const bits = Object.keys(kinds).map(k =>
    `<i>${REJECT_KIND[k].label} ${kinds[k].length} 张</i>`).join('');
  box.innerHTML = `<div class="rej-sum">当前共 <b>${rej.length} 张</b>未进模板：${bits}
    <br><span class="muted">背面照 / 远景小脸本来就只存档，属正常；详细逐张原因见上方注册库卡片。</span></div>`;
}

async function refreshFix() {
  if (!$('foldFix').open) return;
  $('fixMsg').textContent = '正在逐张检测人脸，请稍候……';
  try {
    const r = await (await fetch('/api/track/register_photos')).json();
    if (!r.ok) throw new Error(r.message || '读取失败');
    applyFixState(r);
    $('fixMsg').textContent = `共 ${(r.photos || []).length} 张 · ` +
      `${(r.suspects || []).length} 张可疑 · ${(r.duplicates || []).length} 张跨编号重复` +
      `　（拖动或点选照片即可修正）`;
    renderFixRejects(null);
  } catch (e) { $('fixMsg').textContent = '读取失败：' + e.message; }
}

async function movePhotos(files, to) {
  // 面板可能残留上一次移动前的文件名，先过滤掉，别拿旧名字去请求
  const known = new Set(((S.fix && S.fix.photos) || []).map(p => p.name));
  files = [...new Set(files || [])].filter(f => known.has(f));
  if (!files.length) {
    $('fixMsg').textContent = '选中的照片已不在列表里，点「↻ 重新检测」后再操作';
    return;
  }
  if (!to) return;
  if (S.moving) return;
  S.moving = true;
  const label = to === '__new__' ? `新建编号 ${S.fix.next_person || ''}`
    : to === '__unsorted__' ? '待确认' : to;
  $('fixMsg').textContent = `正在把 ${files.length} 张移到「${label}」并重建注册库……`;
  $('btnFixMove').disabled = true;
  $('btnFixSuspect').disabled = true;
  try {
    const r = await (await fetch('/api/track/register_move', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files, to }),
    })).json();
    if (!r.ok) throw new Error(r.message || '移动失败');
    applyFixState(r);
    $('fixMsg').textContent = r.message + (r.moved && r.moved.length
      ? `　（移动记录已写入 records/photo_moves.jsonl，可再拖回来）` : '');
    renderFixRejects(r.rejected);
    renderRejects(r.rejected);
  } catch (e) {
    $('fixMsg').textContent = '移动失败：' + e.message + '（可点「↻ 重新检测」刷新）';
  } finally {
    $('btnFixMove').disabled = false;
    $('btnFixSuspect').disabled = false;
    S.moving = false;
  }
}

function bindFix() {
  $('btnOpenFix').addEventListener('click', () => {
    $('foldFix').open = true;
    $('fixCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  $('foldFix').addEventListener('toggle', () => { if ($('foldFix').open) refreshFix(); });
  $('btnFixReload').addEventListener('click', refreshFix);
  $('btnFixClear').addEventListener('click', () => { S.selected.clear(); updateSelCount(); });
  $('btnFixMove').addEventListener('click', () =>
    movePhotos([...S.selected], $('fixTarget').value));
  $('btnFixSuspect').addEventListener('click', () => {
    const st = S.fix;
    if (!st) return;
    const bad = [...new Set([...(st.suspects || []), ...(st.duplicates || [])])];
    if (!bad.length) { $('fixMsg').textContent = '没有检出可疑或重复照片，无需移出。'; return; }
    movePhotos(bad, '__unsorted__');
  });

  const cols = $('fixCols');
  cols.addEventListener('click', e => {
    const look = e.target.closest('[data-look]');
    if (look) { openPerson(look.dataset.look); return; }
    const tile = e.target.closest('.fix-photo');
    if (!tile) return;
    const f = tile.dataset.file;
    if (S.selected.has(f)) S.selected.delete(f); else S.selected.add(f);
    updateSelCount();
  });
  cols.addEventListener('dragstart', e => {
    const tile = e.target.closest('.fix-photo');
    if (!tile) return;
    const f = tile.dataset.file;
    S.dragFiles = S.selected.has(f) && S.selected.size > 1 ? [...S.selected] : [f];
    e.dataTransfer.effectAllowed = 'move';
    try { e.dataTransfer.setData('text/plain', S.dragFiles.join('\n')); } catch (_) {}
  });
  cols.addEventListener('dragover', e => {
    const col = e.target.closest('.fix-col');
    if (!col) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    cols.querySelectorAll('.fix-col.over').forEach(x => { if (x !== col) x.classList.remove('over'); });
    col.classList.add('over');
  });
  cols.addEventListener('dragleave', e => {
    const col = e.target.closest('.fix-col');
    if (col && !col.contains(e.relatedTarget)) col.classList.remove('over');
  });
  cols.addEventListener('drop', e => {
    const col = e.target.closest('.fix-col');
    if (!col) return;
    e.preventDefault();
    cols.querySelectorAll('.fix-col.over').forEach(x => x.classList.remove('over'));
    let files = S.dragFiles;
    try {
      const raw = e.dataTransfer.getData('text/plain');
      if (raw) files = raw.split('\n').filter(Boolean);
    } catch (_) {}
    S.dragFiles = [];
    movePhotos(files, col.dataset.person);
  });
  cols.addEventListener('dragend', () => {
    cols.querySelectorAll('.fix-col.over').forEach(x => x.classList.remove('over'));
    S.dragFiles = [];
  });
}

/* ------------------------------------------------ 上传 / 重建 */
async function uploadVideo(e) {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  status(`正在上传 ${f.name} ……`);
  const fd = new FormData(); fd.append('file', f);
  try {
    const r = await (await fetch('/api/track/upload', { method: 'POST', body: fd })).json();
    if (!r.ok) throw new Error(r.message);
    S.userPickedVideo = true;
    S.preferredVideo = r.name;
    // 新版后端直接回传最新列表，不用再等一次 /info；旧后端走 loadInfo 兜底
    if (r.videos) renderVideos(r.videos, r.name);
    else await loadInfo(r.name);
    $('videoSel').value = r.name;
    $('resultVideo').innerHTML = `<b>素材已切换：</b>下拉框现在是 <b>${esc(r.name)}</b>；` +
      `右侧画面仍是上一次分析的结果，点「开始分析」后才会换成新视频。`;
    status(`已上传 ${r.name}，已自动选中；点「开始分析」查看新视频`);
  } catch (err) { status('上传失败：' + err.message); }
  e.target.value = '';
}

async function uploadReg(e) {
  const fs = Array.from(e.target.files || []);
  if (!fs.length) return;
  const btn = $('btnUploadReg');
  const msg = $('regMsg');
  btn.disabled = true;
  msg.textContent = `正在上传 ${fs.length} 张注册照，逐张检测人脸并重建注册库……`;
  $('galCount').textContent = `（正在保存 ${fs.length} 张……）`;
  const fd = new FormData();
  fs.forEach(f => fd.append('files', f));
  const who = ($('regPerson').value || '').trim();
  if (who) fd.append('person', who);   // 一次多选同一人时显式指定，避免被按文件名拆成多个人
  try {
    const r = await (await fetch('/api/track/register_upload', {
      method: 'POST', body: fd,
    })).json();
    if (!r.ok) throw new Error(r.message);
    // 直接用返回值渲染，不再等一次 /info（视频列表不用变，避免计数「延迟」）
    renderGallery(r.gallery || [], r.disk_photos);
    const bits = [`已保存 ${(r.saved || []).length} 张（${r.person || '新人员'}）`,
                  `注册库 ${r.count ?? '—'} 人 / ${r.photos ?? '—'} 张有效模板`,
                  `磁盘共 ${r.disk_photos ?? 0} 张照片`];
    if (r.person && (r.names || []).length && !r.names.includes(r.person))
      bits.push('注意：这个人还没进模板，请补一张清晰正面照');
    if ((r.skipped || []).length) bits.push(`跳过 ${r.skipped.length} 个非图片文件`);
    if ((r.message || '').includes('当前有分析在跑')) bits.push('新注册库从下一次分析开始生效');
    msg.textContent = bits.join('　·　');
    renderRejects(r.rejected);
    if ($('foldFix').open) refreshFix();   // 面板开着就同步最新分组，不用手动刷新
  } catch (err) {
    msg.textContent = '上传失败：' + err.message;
    await loadInfo();
  } finally {
    btn.disabled = false;
    e.target.value = '';
  }
}

async function rebuild() {
  status('正在重建注册库……');
  const r = await (await fetch('/api/track/register', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dir: 'D:/sports_checkin/registered' }),
  })).json();
  if (!r.ok) { status('重建失败：' + r.message); return; }
  renderGallery(r.gallery || [], r.disk_photos);
  $('regMsg').textContent = `注册库已重建：${r.count} 人 / ${r.photos} 张有效模板（磁盘共 ${r.disk_photos} 张照片）`;
  renderRejects(r.rejected);
  if ($('foldFix').open) refreshFix();
  status('注册库已重建。');
}

/* ------------------------------------------------ 分析任务 */
async function poll() {
  const s = await (await fetch('/api/track/status')).json();
  if (s.error) { status('出错：' + s.error); setBar(0); $('btnStart').disabled = false; return; }
  const pct = s.total ? Math.round(s.done / s.total * 100) : 0;
  setBar(pct);
  status(s.running ? `${s.message}　${pct}%` : s.message);
  if (s.running) { setTimeout(poll, 700); return; }
  // 刚点开始时线程可能还没真正跑起来，前几轮「空闲」不算结束
  S.pollTries = (S.pollTries || 0) + 1;
  if (s.done === 0 && S.pollTries < 10) { setTimeout(poll, 700); return; }
  if (s.done > 0) { await loadResult(); $('btnStart').disabled = false; }
  else status('任务结束但没有产出，检查视频能否打开');
}

async function start() {
  const name = $('videoSel').value;
  if (!name) { status('请先选一段视频'); return; }
  const body = {
    video: name,
    seconds: +$('inpSeconds').value || 10,
    stride: +$('inpStride').value || 8,
    conf: +$('inpConf').value || 0.35,
    use_id: $('chkId').checked,
  };
  $('btnStart').disabled = true;
  stopPlay();
  setBar(0);
  status('正在启动……');
  const r = await (await fetch('/api/track/start', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })).json();
  if (!r.ok) { status('启动失败：' + r.message); $('btnStart').disabled = false; return; }
  S.job = r.job;
  S.preferredVideo = name;
  S.userPickedVideo = true;
  S.pollTries = 0;
  setTimeout(poll, 500);
}

async function loadResult() {
  const r = await (await fetch('/api/track/result')).json();
  if (!r.ok || !r.result) { status('没有结果文件'); return; }
  const d = r.result;
  S.job = d.job || S.job;
  S.frames = d.frames_saved || 0;
  S.idx = 0;
  $('verdict').textContent = d.verdict;
  $('idHint').innerHTML = d.mode === 'face'
    ? (d.registered_count > 0
        ? `<b style="color:#c0392b">红框 ${d.registered_count} 人</b>：${(d.registered || []).join('、')}（人脸 cosine ≥ ${d.threshold}、领先次高 ≥ ${d.margin}）　·　<b style="color:#3d7fbf">蓝框 ${d.passerby_count} 个 ID</b> 为路人或没检到脸。同帧重叠框：${d.overlap_pairs ?? 0} 对。`
        : `这段视频里没有人与注册库匹配上，<b>全部按路人画蓝框</b> —— 这是「宁可判路人也不误认」的正确结果，不是识别失败。想看红框效果请选 <b>demo_simple.mp4</b>（用注册照生成的演示视频）。同帧重叠框：${d.overlap_pairs ?? 0} 对。`)
    : '当前是外观特征兜底模式，身份结论不可靠；把 yunet.onnx / sface.onnx 放进 models/ 即自动切换人脸模式。';
  $('resultVideo').innerHTML = d.video
    ? `本次结果来自：<b>${esc(d.video)}</b>　·　下方「分析明细」可折叠`
    : '';
  $('kvFrames').textContent = d.frames_total;
  $('kvIds').textContent = d.track_count;
  $('kvLong').textContent = d.max_track_frames;
  $('kvReg').textContent = d.registered_count ?? (d.registered ? d.registered.length : 0);
  $('kvPass').textContent = d.passerby_count ?? '—';

  const range = $('frameRange');
  range.max = Math.max(0, S.frames - 1);
  range.value = 0;
  // 停在「框最多、优先含注册人员」的那一帧，别落在空场/无人画面上
  showFrame(Math.min(S.frames - 1,
                     Math.max(0, d.best_frame ?? Math.floor(S.frames * 0.35))));
  // 只在用户没自己选过视频时同步，避免旧任务的结果把刚上传的新视频「顶掉」
  if (d.video && !S.userPickedVideo &&
      [...$('videoSel').options].some(o => o.value === d.video)) {
    $('videoSel').value = d.video;
    S.preferredVideo = d.video;
  }

  $('trackSum').textContent = `· ${(d.tracks || []).length} 个 ID`;
  $('checkinSum').textContent = `· ${(d.checkins || []).length} 条`;
  $('trackRows').innerHTML = d.tracks.length
    ? d.tracks.map(t => {
        const reg = !!t.name;
        const kind = reg ? '<span class="pill red">注册人员</span>'
                         : '<span class="pill blue">路人</span>';
        return `<tr>
          <td><span class="pill">#${t.id}</span></td>
          <td>${t.frames} 帧</td>
          <td>${kind}</td>
          <td>${reg ? `<b style="color:#c0392b">${t.name}</b>` : '<span class="muted">—</span>'}</td>
          <td>${t.sim ? t.sim.toFixed(3) : '—'}</td></tr>`;
      }).join('')
    : '<tr><td colspan="5" class="muted">没有稳定目标</td></tr>';

  $('checkinRows').innerHTML = d.checkins && d.checkins.length
    ? d.checkins.slice().reverse().map(c => `<tr>
        <td><span class="pill ok">${c.name}</span></td>
        <td>${c.time}</td><td>第 ${c.frame} 帧</td><td>${c.sim}</td></tr>`).join('')
    : '<tr><td colspan="4" class="muted">没有打卡记录（未启用身份比对，或没有人通过阈值）</td></tr>';

  const dl = $('btnDownload');
  dl.style.display = 'inline-block';
  dl.href = '/api/track/download/' + S.job;
  status('完成：' + d.verdict);
}

function showFrame(i) {
  if (!S.frames) return;
  S.idx = Math.max(0, Math.min(S.frames - 1, i));
  $('frameImg').src = `/api/track/frame/${S.job}/${S.idx}?t=${Date.now()}`;
  $('frameRange').value = S.idx;
  $('frameNo').textContent = `${S.idx + 1} / ${S.frames}`;
}

function stopPlay() {
  if (S.timer) clearInterval(S.timer);
  S.timer = null; S.playing = false;
  $('btnPlay').textContent = '播放';
}

function togglePlay() {
  if (S.playing) { stopPlay(); return; }
  if (!S.frames) return;
  S.playing = true;
  $('btnPlay').textContent = '暂停';
  S.timer = setInterval(() => {
    if (S.idx >= S.frames - 1) { stopPlay(); return; }
    showFrame(S.idx + 1);
  }, 85);
}

/* ------------------------------------------------ 事件绑定 */
function bind() {
  bindFix();
  $('btnStart').addEventListener('click', start);
  $('btnUpload').addEventListener('click', () => $('fileVideo').click());
  $('fileVideo').addEventListener('change', uploadVideo);
  $('btnPlay').addEventListener('click', togglePlay);
  $('btnRebuild').addEventListener('click', rebuild);
  $('btnUploadReg').addEventListener('click', () => $('fileReg').click());
  $('fileReg').addEventListener('change', uploadReg);
  $('frameRange').addEventListener('input', e => { stopPlay(); showFrame(+e.target.value); });

  $('videoSel').addEventListener('change', e => {
    S.userPickedVideo = true; S.preferredVideo = e.target.value;
  });
  $('galList').addEventListener('click', e => {
    const b = e.target.closest('.chip');
    if (b) openPerson(b.dataset.name);
  });
  $('pmShots').addEventListener('click', e => {
    const b = e.target.closest('.pm-shot');
    if (!b) return;
    const f = b.dataset.file;
    $('pmPhoto').src = photoUrl(f, 640);
    [...$('pmShots').querySelectorAll('.pm-shot')].forEach(x => x.classList.toggle('on', x === b));
    const g = S.gallery.find(x => (x.shots || []).some(s => s.name === f));
    if (g) $('pmNote').textContent = shotNote(g, f);
  });
  $('pmClose').addEventListener('click', closePerson);
  $('personModal').addEventListener('click', e => {
    if (e.target === $('personModal')) closePerson();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !$('personModal').hidden) closePerson();
  });
}

bind();
loadInfo();
loadResult();     // 打开页面就显示上一次的结果，答辩不用重新跑
