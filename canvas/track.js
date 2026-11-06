/* 人员打卡 / 同场景追踪 —— 页面逻辑 */
const $ = id => document.getElementById(id);
const S = {
  job: '', frames: 0, idx: 0, timer: null, playing: false, info: null,
  gallery: [], preferredVideo: '', userPickedVideo: false, pollTries: 0,
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
    ? S.gallery.map(g => `<button class="chip" data-name="${esc(g.name)}" title="点开看 ${esc(g.name)} 的正面照">
        <b>${esc(g.name)}</b><small>${g.photos} 张模板 · 共 ${g.saved} 张</small></button>`).join('')
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
