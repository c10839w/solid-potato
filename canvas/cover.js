/* 封面页：读统计数字 + 上传（先选类别再选文件） */
const $ = id => document.getElementById(id);

(async () => {
  try {
    const s = await (await fetch('/api/stats')).json();
    $('sImg').textContent = s.images;
    $('sObj').textContent = s.objects;
    $('sFix').textContent = s.fixes;
  } catch (e) { /* 打不开也照样能用，不阻断页面 */ }
})();

$('btnOpenUpload').addEventListener('click', () => $('sheet').classList.add('show'));
$('btnCancel').addEventListener('click', () => $('sheet').classList.remove('show'));
$('sheet').addEventListener('click', e => { if (e.target === $('sheet')) $('sheet').classList.remove('show'); });
$('btnPick').addEventListener('click', () => $('coverFile').click());

$('coverFile').addEventListener('change', async e => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  const cat = (document.querySelector('input[name=cat]:checked') || {}).value || '其他';
  const msg = $('upMsg');
  msg.textContent = `正在上传 ${f.name} ……`;
  msg.style.color = '';

  const fd = new FormData();
  fd.append('file', f);
  fd.append('category', cat);

  let res;
  try {
    res = await (await fetch('/api/upload', { method: 'POST', body: fd })).json();
  } catch (err) {
    msg.textContent = '上传失败：' + err;
    msg.style.color = '#c0555a';
    return;
  } finally { e.target.value = ''; }

  if (!res.ok) {
    msg.textContent = res.message || '上传失败';
    msg.style.color = '#c0555a';
    return;
  }
  msg.textContent = '上传好了，正在打开工作台……';
  // 带上文件名跳过去，工作台会自动选中并跑一遍模型
  location.href = '/workbench?pick=' + encodeURIComponent(res.name);
});
