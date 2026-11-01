# 人机协同图像标注系统

一个本地运行的网页工具：模型先画出标注草稿，人用鼠标把草稿改准，改完的数据导出成标准数据集，
再拿去训练，训练好的模型回来重新预标注 —— 一轮比一轮准。

十类物体（顺序固定，不得更改）：
`person` 人 / `chair` 椅子 / `dining table` 桌子 / `couch` 沙发 / `tv` 电视 /
`laptop` 笔记本 / `cup` 杯子 / `bottle` 瓶子 / `potted plant` 盆栽 / `bed` 床

---

## 一、怎么运行

**方式 A：双击（推荐）**

双击 `启动.bat`。它会自动激活 `imglabel` 环境并打开浏览器 `http://127.0.0.1:5000`。
第一次运行若弹出「Windows 已保护你的电脑」，点「更多信息 → 仍要运行」。

**方式 B：命令行**

```
conda activate imglabel
cd /d D:\imglabel
python app.py
```

浏览器打开 `http://127.0.0.1:5000`。

## 二、环境依赖

```
conda create -n imglabel python=3.10 -y
conda activate imglabel
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

无独立显卡亦可运行（CPU 训练），首次启动会自动下载模型权重。

## 三、目录说明

| 路径 | 作用 |
|---|---|
| `app.py` | Flask 服务端：图片列表、标注读写、预标注、修正记账、导出 |
| `prelabel.py` | 模型预标注：图片 → 类别 + 轮廓（框由轮廓外接矩形派生） |
| `export.py` | 导出标准数据集（det / seg 两套标签 + 两种掩码 + 训练配置） |
| `classes.yaml` | 十类词表 + COCO 编号映射（**全流程唯一的类别顺序来源**） |
| `canvas/` | 网页界面：`index.html` / `style.css` / `app.js` |
| `data/raw/` | 原始图片（只放图，不放别的） |
| `data/preview/` | 预标注结果可视化 |
| `records/drafts/` | 模型草稿（json） |
| `records/annotations/` | 人工确认后的标注 |
| `records/fixes.jsonl` | 修正流水账，一条一行，只追加不覆盖 |
| `model/` | 训练产物 best.pt |
| `启动.bat` | 双击启动 |

## 四、三种标注模式的关系

只维护**一份实例掩码数据**，另外两种模式由它派生：

- 目标检测 = 掩码的外接矩形
- 实例分割 = 掩码本身
- 语义分割 = 同类掩码按类别合并上色

因此界面上真正需要手工编辑的只有「轮廓」这一件事。

## 五、导出数据集

在界面上点 **「导出数据集」**，或在命令行执行 `python export.py`。产物在 `D:\imglabel\dataset\`：

```
dataset\
  images\train|val\        原图
  labels\train|val\        检测标签  cls cx cy w h（0~1）
  seg\images\train|val\    原图（与上面同一批图，训练时靠目录配对）
  seg\labels\train|val\    分割标签  cls x1 y1 x2 y2 ...
  masks\train|val\         实例掩码 PNG（0=背景，1~N=第几个物体）
  semantic\train|val\      语义掩码 PNG（0=背景，1~10=类别号+1）
  data.yaml                检测训练配置
  seg\data.yaml            分割训练配置
  classes.txt / instances.json / export_report.json
```

要点：

- **训练/验证固定 8:2**，随机种子写死（`--seed 42`），每次导出划分结果一致，结果可复现
- 图片少于 2 张时没有验证集，`val` 自动指向训练集，只适合快速跑通流程
- 同一份标注出两套标签：`labels/` 是外接框（检测用），`seg/labels/` 是多边形（分割用）
- 掩码图也可以肉眼看：`data/preview/export/` 里是按类别上色的语义图
- 导出会先清空旧产物，不会把上次的图混进来

命令行参数：`--val-ratio 0.15`（验证集比例）、`--only-human`（只导人工改过的）。

## 六、当前进度

- [x] 环境搭建（Miniforge + imglabel + ultralytics/torch/flask）
- [x] 模型预标注打通（yolo26n-seg / YOLO-World 均可运行）
- [x] 画布界面：三模式切换、点选物体、拖顶点、拖框、改类别、增删、修正记账
- [x] 数据集导出（YOLO txt / 实例掩码 PNG / 语义 PNG + 两份训练配置）
- [ ] 训练脚本 train.py，产出 model/best.pt
- [ ] 第二轮回标 + 前后对比（mAP、修正条数）
- [ ] 演示视频与交付文档

## 七、AI 工具使用说明

开发过程中使用了 AI 编程助手，时间、工具、用途逐条记录在 `ai_log.md`。
所有模型权重均为本机实际训练/实际运行产出，未使用任何伪造结果。
