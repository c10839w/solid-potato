/* 人员打卡 / 同场景追踪 —— 页面逻辑 */
const $ = id => document.getElementById(id);
const S = { job: '', frames: 0, idx: 0, timer: null, playing: false, info: null };

function status(t) { $('statusText').textContent = t; }
function setBar(p) { $('barFill').style.width = Math.max(0, Math.min(100, p)) + '%'; }

async function loadInfo() {
  const r = await (await fetch('/api/track/info')).json();
  S.info = r;
  const sel = $('videoSel');
  sel.innerHTML = r.videos.length
    ? r.videos.map(v => `<option value="${v.name}">${v.name} · ${v.width}×${v.height} · ${v.seconds}s · ${v.size_mb}MB</option>`).join('')
    : '<option value="">（没有找到视频，请先上传或把 mp4 放进 videos 目录）</option>';
  // 默认选 *_ck 打卡机位；没有就选第一个
  const ck = r.videos.find(v => v.name.includes('_ck'));
  if (ck) sel.value = ck.name;

  const modeTxt = r.mode === 'face'
    ? '人脸级识别（YuNet + SFace 已加载）'
    : '外观特征（离线临时方案，身份结果不可靠）';
  $('modeNote').innerHTML = `<b>当前身份模式：${modeTxt}</b>　·　` +
    `验证重点是「框住 + 同一个人不换号」；识别是谁这一步目前仅供参考。` +
    `人员检测用 COCO 预训练的 yolo26n-seg，追踪用自写 IOU 追踪器，全程离线、无需联网。`;

  $('galCount').textContent = `（${r.gallery.length} 人）`;
  $('galList').innerHTML = r.gallery.length
    ? r.gallery.map(g => `<span class="pill" style="margin:0 6px 6px 0">${g.name} · ${g.photos} 张</span>`).join('')
    : '还没有注册人员。点下面按钮，用现有 7 张注册照建库。';
}

async function uploadVideo(e) {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  status(`正在上传 ${f.name} ……`);
  const fd = new FormData(); fd.append('file', f);
  try {
    const r = await (await fetch('/api/track/upload', { method: 'POST', body: fd })).json();
    if (!r.ok) throw new Error(r.message);
    await loadInfo();
    $('videoSel').value = r.name;
    status(`已上传 ${r.name}，可以开始分析了`);
  } catch (err) { status('上传失败：' + err.message); }
  e.target.value = '';
}

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
  $('kvFrames').textContent = d.frames_total;
  $('kvIds').textContent = d.track_count;
  $('kvLong').textContent = d.max_track_frames;
  $('kvSwitch').textContent = d.id_switches ?? '—';

  const range = $('frameRange');
  range.max = Math.max(0, S.frames - 1);
  range.value = 0;
  showFrame(Math.floor(S.frames * 0.35));   // 停在中段，画面比首帧干净
  // 下拉同步到这份结果对应的视频，避免「选的是 A、看到的是 B」
  if (d.video && [...$('videoSel').options].some(o => o.value === d.video)) {
    $('videoSel').value = d.video;
  }

  $('trackRows').innerHTML = d.tracks.length
    ? d.tracks.map(t => `<tr>
        <td><span class="pill">#${t.id}</span></td>
        <td>${t.frames} 帧</td>
        <td>${t.name ? `<span class="pill ok">${t.name}</span>` : '<span class="muted">未识别 / 未启用</span>'}</td>
        <td>${t.sim ? t.sim.toFixed(3) : '—'}</td></tr>`).join('')
    : '<tr><td colspan="4" class="muted">没有稳定目标</td></tr>';

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

async function rebuild() {
  status('正在重建注册库……');
  const r = await (await fetch('/api/track/register', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dir: 'D:/sports_checkin/registered' }),
  })).json();
  if (!r.ok) { status('重建失败：' + r.message); return; }
  status(`注册库已重建：${r.count} 人 / ${r.photos} 张照片（${r.mode}）`);
  await loadInfo();
}

function bind() {
  $('btnStart').addEventListener('click', start);
  $('btnUpload').addEventListener('click', () => $('fileVideo').click());
  $('fileVideo').addEventListener('change', uploadVideo);
  $('btnPlay').addEventListener('click', togglePlay);
  $('btnRebuild').addEventListener('click', rebuild);
  $('frameRange').addEventListener('input', e => { stopPlay(); showFrame(+e.target.value); });
}

bind();
loadInfo();
loadResult();     // 打开页面就显示上一次的结果，答辩不用重新跑
