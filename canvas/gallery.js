/* 照片墙：按类别筛、点开看大图、导出已标注照片 */
const $ = id => document.getElementById(id);

const G = { all: [], cat: 'all', onlyAnnotated: false };

async function load() {
  const d = await (await fetch('/api/images')).json();
  G.all = d.images;
  render();
}

function render() {
  const list = G.all.filter(it => {
    if (G.cat !== 'all' && (it.category || '其他') !== G.cat) return false;
    if (G.onlyAnnotated && !(it.objects > 0)) return false;
    return true;
  });

  const annotated = G.all.filter(it => it.objects > 0).length;
  $('wallSub').textContent =
    `${G.all.length} 张图，其中 ${annotated} 张有标注；共 ${G.all.reduce((a, b) => a + b.objects, 0)} 个物体`;

  $('grid').innerHTML = list.map(it => {
    const human = it.human > 0;
    return `<div class="shot" data-name="${it.name}">
      <img src="/api/thumb/${encodeURIComponent(it.name)}" loading="lazy" alt="">
      <span class="badge ${human ? 'human' : ''}">${it.objects} 个物体${human ? ' · 已修正' : ''}</span>
      <div class="cap"><b>${it.name.replace(/\.(jpg|jpeg|png|webp)$/i, '')}</b><span>${it.category || '其他'}</span></div>
    </div>`;
  }).join('');

  $('empty').style.display = list.length ? 'none' : 'block';

  $('grid').querySelectorAll('.shot').forEach(el => {
    el.addEventListener('click', () => openLB(el.dataset.name));
  });
}

function openLB(name) {
  const it = G.all.find(x => x.name === name);
  $('lbImg').src = '/api/thumb/' + encodeURIComponent(name) + '?w=1200';
  $('lbName').textContent = name;
  $('lbMeta').textContent = it ? `${it.objects} 个物体 · ${it.human} 处人工改过 · ${it.category || '其他'}` : '';
  $('lightbox').classList.add('show');
}

$('btnClose').addEventListener('click', () => $('lightbox').classList.remove('show'));
$('lightbox').addEventListener('click', e => {
  if (e.target === $('lightbox')) $('lightbox').classList.remove('show');
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') $('lightbox').classList.remove('show'); });

$('filters').addEventListener('click', e => {
  const b = e.target.closest('.chipbtn[data-cat]');
  if (!b) return;
  G.cat = b.dataset.cat;
  document.querySelectorAll('.chipbtn[data-cat]').forEach(x => x.classList.toggle('on', x === b));
  render();
});

$('btnAnnotated').addEventListener('click', () => {
  G.onlyAnnotated = !G.onlyAnnotated;
  $('btnAnnotated').classList.toggle('on', G.onlyAnnotated);
  render();
});

$('btnExportImg').addEventListener('click', async () => {
  const btn = $('btnExportImg');
  const old = btn.textContent;
  btn.textContent = '打包中……';
  btn.disabled = true;
  try {
    const r = await fetch('/api/export_images', { method: 'POST' });
    if (!r.ok) throw new Error('服务端返回 ' + r.status);
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = '标注照片.zip';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    btn.textContent = '已开始下载 ✓';
  } catch (err) {
    btn.textContent = '导出失败：' + err.message;
  }
  setTimeout(() => { btn.textContent = old; btn.disabled = false; }, 2600);
});

load();
