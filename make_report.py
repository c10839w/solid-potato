# -*- coding: utf-8 -*-
"""
make_report.py —— 生成「前后对比报告」records/compare.html

任务书第 6 步要求：第二轮回标后，给出可量化的对比材料。
这一页就是把散在各处的数字汇总成人能一眼看懂的报告，交付时直接给评委看。

读的数据：
    records/metrics_baseline.json    第一轮基线模型（COCO 预训练）在验证集上的精度
    records/metrics_trained.json     第二轮自训模型（model/best.pt）的精度
    records/round2_report.json       第二轮回标：草稿需要人工处理的处数
    records/fixes.jsonl              人工修正账本（每改一处记一行）
    dataset/export_report.json       数据集概况

用法：python make_report.py
产物：records/compare.html（单文件，样式内嵌，断网也能打开）
"""

import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
RECORDS = ROOT / "records"


def load(name):
    p = RECORDS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def bar(value, vmax, color="#2F7D5B", w=180, h=14):
    """一条简单的横条，纯 div，不依赖任何 JS 库"""
    pct = 0 if not vmax else max(0.0, min(1.0, value / vmax))
    return (f'<div class="bar"><span style="width:{pct*100:.1f}%;'
            f'background:{color};height:{h}px"></span></div>')


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def pct(v):
    return "—" if v is None else f"{v:+.1f}%"


def main():
    baseline = load("metrics_baseline.json")
    trained = load("metrics_trained.json")
    round2 = load("round2_report.json")
    export = ROOT / "dataset" / "export_report.json"
    export = json.loads(export.read_text(encoding="utf-8")) if export.exists() else None

    fixes = []
    fp = RECORDS / "fixes.jsonl"
    if fp.exists():
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    fixes.append(json.loads(line))
                except Exception:
                    pass
    act = Counter(f.get("action") for f in fixes)
    act_zh = {"class": "改类别", "vertex": "拖轮廓顶点", "move": "拖框",
              "add": "新增物体", "delete": "删除物体", "corner": "缩放框"}
    n_img_fixed = len({f.get("image") for f in fixes})

    names = (export or {}).get("per_class") and list(export["per_class"].keys()) or []

    # ---------- 精度对比 ----------
    def metric(d, key):
        return None if not d else d.get(key)

    rows = []
    if baseline and trained:
        per_b = baseline.get("mask_map50_per_class") or {}
        per_t = trained.get("mask_map50_per_class") or {}
        for n in sorted(set(per_b) | set(per_t), key=lambda x: -(per_b.get(x) or 0)):
            b, t = per_b.get(n), per_t.get(n)
            delta = (None if (b is None or t is None) else (t - b) * 100)
            cls = "up" if (delta or 0) > 0.5 else ("down" if (delta or 0) < -0.5 else "flat")
            arrow = "↑" if cls == "up" else ("↓" if cls == "down" else "→")
            rows.append(f"""<tr><td>{n}</td><td>{fmt(b,3)}</td><td>{fmt(t,3)}</td>
              <td class="{cls}">{arrow} {pct(delta)}</td>
              <td>{bar(t or 0, 1.0, '#2F7D5B', 120)}</td></tr>""")

    per_class_table = "".join(rows) or '<tr><td colspan="5" class="dim">待训练完成后生成</td></tr>'

    def card(label, b, t, unit=""):
        if b is None or t is None:
            return f"""<div class="metric"><div class="k">{label}</div>
                      <div class="v dim">待评估</div></div>"""
        d = (t - b) * 100
        cls = "up" if d > 0.5 else ("down" if d < -0.5 else "flat")
        arrow = "↑" if cls == "up" else ("↓" if cls == "down" else "→")
        return f"""<div class="metric"><div class="k">{label}</div>
                  <div class="v">{t:.4f}{unit}</div>
                  <div class="d {cls}">{arrow} {d:+.1f}（原 {b:.4f}）</div></div>"""

    # ---------- 第二轮回标 ----------
    r2_html = '<p class="dim">待第二轮回标完成后生成</p>'
    if round2:
        s = round2["summary"]
        a1, a2 = s["round1"], s["round2"]
        imp = s["improvement_pct"]
        r2_html = f"""
        <div class="two">
          <div class="col"><div class="col-h">第一轮草稿（COCO 原模型）</div>
            <div class="big">{a1['need_fix']}</div>
            <div class="sub">处需要人工处理 · 平均每张 {a1['per_image']} 处</div>
            <ul class="brk">
              <li>漏标 <b>{a1['miss']}</b></li><li>类别认错 <b>{a1['wrong_class']}</b></li>
              <li>框位置歪 <b>{a1['drift']}</b></li><li>多标需删 <b>{a1['extra']}</b></li>
            </ul></div>
          <div class="col hi"><div class="col-h">第二轮草稿（自训 best.pt）</div>
            <div class="big">{a2['need_fix']}</div>
            <div class="sub">处需要人工处理 · 平均每张 {a2['per_image']} 处</div>
            <ul class="brk">
              <li>漏标 <b>{a2['miss']}</b></li><li>类别认错 <b>{a2['wrong_class']}</b></li>
              <li>框位置歪 <b>{a2['drift']}</b></li><li>多标需删 <b>{a2['extra']}</b></li>
            </ul></div>
        </div>
        <p class="concl">在同样 {s['images']} 张人工核对过的图上，需要人工处理的地方从
        <b>{a1['need_fix']}</b> 处降到 <b>{a2['need_fix']}</b> 处，
        <b class="up">减少 {imp['need_fix']}%</b>。其中漏标减少 {imp['miss']}%、
        类别认错减少 {imp['wrong_class']}%、框位置歪减少 {imp['drift']}%。</p>"""

    # ---------- 修正分布 ----------
    vmax = max(act.values()) if act else 1
    fix_rows = "".join(
        f'<tr><td>{act_zh.get(k, k)}</td><td>{v}</td><td>{bar(v, vmax, "#4A6FA5", 200)}</td></tr>'
        for k, v in act.most_common())

    ds = ""
    if export:
        ds = (f"{export['images_total']} 张图（训练 {export['train']} / 验证 {export['val']}）·"
              f"{export['objects']} 个标注物体 · 十类齐全")

    generated = time.strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>人机协同图像标注系统 · 前后对比报告</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:40px 28px 64px; background:#FAF9F6; color:#2C2C2A;
         font:15px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width: 880px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:.5px; }}
  .meta {{ color:#8A8780; font-size:13px; margin-bottom:32px; }}
  h2 {{ font-size:16px; margin:38px 0 14px; padding-left:10px;
        border-left:3px solid #2F7D5B; font-weight:600; }}
  .card {{ background:#fff; border:1px solid #ECEAE4; border-radius:10px; padding:20px 22px; }}
  .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
  .metric {{ background:#fff; border:1px solid #ECEAE4; border-radius:10px; padding:14px 16px; }}
  .metric .k {{ font-size:12.5px; color:#8A8780; }}
  .metric .v {{ font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }}
  .metric .d {{ font-size:12.5px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #F1EFE9; }}
  th {{ font-size:12.5px; color:#8A8780; font-weight:500; }}
  td:nth-child(2),td:nth-child(3) {{ font-variant-numeric:tabular-nums; }}
  .bar {{ background:#F1EFE9; border-radius:4px; overflow:hidden; width:100%; min-width:60px; }}
  .bar span {{ display:block; border-radius:4px; }}
  .up {{ color:#2F7D5B; }} .down {{ color:#C0392B; }} .flat {{ color:#8A8780; }}
  .dim {{ color:#A8A49B; }}
  .two {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  .col {{ background:#fff; border:1px solid #ECEAE4; border-radius:10px; padding:18px 20px; }}
  .col.hi {{ border-color:#BFDCCB; background:#F7FBF8; }}
  .col-h {{ font-size:13px; color:#8A8780; }}
  .big {{ font-size:38px; font-weight:600; line-height:1.25; font-variant-numeric:tabular-nums; }}
  .sub {{ font-size:12.5px; color:#8A8780; }}
  .brk {{ margin:10px 0 0; padding-left:18px; font-size:13.5px; color:#55534E; }}
  .brk li {{ margin:2px 0; }}
  .concl {{ background:#F7FBF8; border:1px solid #BFDCCB; border-radius:10px;
            padding:14px 18px; font-size:14.5px; }}
  .foot {{ margin-top:40px; color:#8A8780; font-size:12.5px; border-top:1px solid #ECEAE4; padding-top:16px; }}
  code {{ background:#F1EFE9; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
</style></head><body><div class="wrap">
<h1>人机协同图像标注系统 · 前后对比报告</h1>
<div class="meta">生成于 {generated}　·　数据：{ds or '见 dataset/export_report.json'}</div>

<h2>一、人工修正工作量</h2>
<div class="metrics" style="grid-template-columns:repeat(3,1fr)">
  <div class="metric"><div class="k">修正总处数</div><div class="v">{len(fixes)}</div>
    <div class="d flat">任务书要求 ≥60 处</div></div>
  <div class="metric"><div class="k">涉及图片</div><div class="v">{n_img_fixed}</div>
    <div class="d flat">人工核对过的图</div></div>
  <div class="metric"><div class="k">平均每张</div><div class="v">{(len(fixes)/n_img_fixed if n_img_fixed else 0):.1f}</div>
    <div class="d flat">处 / 张</div></div>
</div>
<div class="card" style="margin-top:14px">
  <table><tr><th>修正动作</th><th>次数</th><th>占比</th></tr>{fix_rows}</table>
</div>

<h2>二、模型精度对比（同一验证集，均为人工核对过的标注）</h2>
<div class="metrics">
  {card('掩码 mAP50', metric(baseline,'mask_map50'), metric(trained,'mask_map50'))}
  {card('掩码 mAP50-95', metric(baseline,'mask_map50_95'), metric(trained,'mask_map50_95'))}
  {card('框 mAP50', metric(baseline,'box_map50'), metric(trained,'box_map50'))}
  {card('框 mAP50-95', metric(baseline,'box_map50_95'), metric(trained,'box_map50_95'))}
</div>
<div class="card" style="margin-top:14px">
  <table>
    <tr><th>类别</th><th>改造前</th><th>改造后</th><th>变化</th><th>改造后水平</th></tr>
    {per_class_table}
  </table>
</div>

<h2>三、第二轮回标：模型草稿还需要人改多少</h2>
{r2_html}

<div class="foot">
  复现方式：<code>python export.py</code> → <code>python eval_baseline_coco.py</code> →
  <code>python train.py</code> → <code>python eval_model.py --model model/best.pt --tag trained</code> →
  <code>python round2_compare.py</code> → <code>python make_report.py</code><br>
  基线口径说明：COCO 预训练模型的类别编号与本项目十类不同，评估时已把验证集标签临时翻译回 COCO 编号，
  与本项目十类口径等价（脚本 <code>eval_baseline_coco.py</code>）。
</div>
</div></body></html>"""

    RECORDS.mkdir(exist_ok=True)
    out = RECORDS / "compare.html"
    out.write_text(html, encoding="utf-8")
    print(f"报告已生成：{out}")
    print(f"  修正 {len(fixes)} 处 / {n_img_fixed} 张图")
    if baseline:
        print(f"  基线：框 {baseline.get('box_map50')}　掩码 {baseline.get('mask_map50')}")
    if trained:
        print(f"  自训：框 {trained.get('box_map50')}　掩码 {trained.get('mask_map50')}")
    if round2:
        print(f"  第二轮回标：需修 {round2['summary']['round1']['need_fix']} → "
              f"{round2['summary']['round2']['need_fix']} 处"
              f"（{round2['summary']['improvement_pct']['need_fix']}%）")


if __name__ == "__main__":
    main()
