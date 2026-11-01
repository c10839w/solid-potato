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
| `app.py` | Flask 服务端：图片列表、标注读写、预标注、修正记账 |
| `prelabel.py` | 模型预标注：图片 → 类别 + 轮廓（框由轮廓外接矩形派生） |
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

## 五、当前进度

- [x] 环境搭建（Miniforge + imglabel + ultralytics/torch/flask）
- [x] 模型预标注打通（yolo26n-seg / YOLO-World 均可运行）
- [x] 画布界面：三模式切换、点选物体、拖顶点、拖框、改类别、增删、修正记账
- [ ] 数据集导出（YOLO txt / 实例掩码 PNG / 语义 PNG）
- [ ] 训练脚本 train.py，产出 model/best.pt
- [ ] 第二轮回标 + 前后对比（mAP、修正条数）
- [ ] 演示视频与交付文档

## 六、AI 工具使用说明

开发过程中使用了 AI 编程助手，时间、工具、用途逐条记录在 `ai_log.md`。
所有模型权重均为本机实际训练/实际运行产出，未使用任何伪造结果。
