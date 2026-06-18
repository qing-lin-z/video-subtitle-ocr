# Video Subtitle OCR 视频字幕提取工具

自动提取视频中的字幕文本，生成 SRT 字幕文件。支持 GPU 加速 (CUDA/DirectML)，多线程并行 OCR，智能去重合并。

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## ✨ 特性

- **🎯 交互式框选** — OpenCV 窗口拖拽选择字幕区域，实时预览
- **⚡ 高性能 OCR** — 基于 RapidOCR (ONNX Runtime)，无需 PyTorch，安装包仅 ~50MB
- **🚀 GPU 加速** — 支持 CUDA (NVIDIA) / DirectML (AMD/Intel)，单帧 ~12ms
- **🧵 多线程并行** — 顺序读帧 + 多线程并行 OCR，充分利用 CPU/GPU
- **🎨 智能预处理** — 9 种预处理变体（去噪/锐化/自适应阈值/CLAHE/形态学），适配各种字幕风格
- **🏆 加权投票** — 多候选一致增强 + 高置信度早停，准确率大幅提升
- **🔗 智能合并** — 相似度去重 + 渐进式文本拼接，输出干净 SRT

## 📦 安装

```bash
# 1. 克隆仓库
git clone https://github.com/qing-lin-z/video-subtitle-ocr.git
cd video-subtitle-ocr

# 2. 安装依赖
pip install -r requirements.txt
```

### Windows 快捷启动

双击 `video_subtitle_ocr_gui.bat` 启动图形界面，或 `video_subtitle_ocr.bat` 运行命令行。

## 🚀 用法

### 图形界面（推荐）

```bash
python video_subtitle_ocr_gui.py
```

支持的参数：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu` | 自动 | 强制开启/关闭 GPU (`--no-gpu`) |
| `--chinese-lite` | - | 中文加速模式（关闭方向分类器） |
| `--box-thresh` | 0.3 | 文本框检出敏感度 |
| `--text-score` | 0.5 | 文字置信度阈值 |
| `--workers` | 4 | 并行 OCR 线程数 |

### 命令行

```bash
# 最简用法
python video_subtitle_ocr.py video.mp4

# 每 3 帧扫描（更高精度）
python video_subtitle_ocr.py video.mp4 -i 3

# GPU 加速
python video_subtitle_ocr.py video.mp4 --gpu

# 调整相似度阈值
python video_subtitle_ocr.py video.mp4 -t 0.9

# 保存/复用框选位置
python video_subtitle_ocr.py video.mp4 --save-roi
python video_subtitle_ocr.py video2.mp4 --no-select
```

#### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i / --interval` | 10 | 帧扫描间隔 |
| `-t / --threshold` | 0.85 | 文本相似度阈值 |
| `--min-duration` | 500 | 字幕最短持续时间(ms) |
| `--min-text-len` | 2 | 最短识别文本长度 |
| `--preprocess` | auto | 预处理模式 |
| `--gpu` | CPU | 启用 GPU 加速 |
| `--no-gpu` | - | 强制 CPU 模式 |
| `--chinese-lite` | - | 中文加速模式 |
| `--box-thresh` | 0.3 | 文本框检出阈值 |
| `--text-score` | 0.5 | 文字置信度阈值 |
| `--workers` | 4 | 并行 OCR 线程数 |

## 🧠 工作原理

### 三阶段流程

1. **框选 ROI** — 交互式选择字幕区域（自动跳过黑屏开场）
2. **OCR 识别** — 9 种预处理变体 → RapidOCR → 加权投票
3. **字幕合并** — 相似度去重 + 渐进式拼接 → 输出 SRT

### 准确率优化

| 优化项 | 说明 |
|--------|------|
| 9 种预处理 | 去噪 → 锐化 → 多尺度自适应阈值 → CLAHE → 形态学闭运算 |
| 智能投票 | 权重 = 置信度×0.7 + 文本长度因子×0.3 |
| 多数一致增强 | ≥2 个变体返回相同结果直接采用 |
| 高置信度早停 | 置信度≥0.90 且长度≥4 跳过后续变体 |
| 后处理 | 首尾标点清理、多余空格合并 |

### GPU 加速

支持三种后端，自动选择最优：
1. **CUDA** (NVIDIA) — 首选，~12ms/帧
2. **TensorRT** (NVIDIA) — 次选
3. **DirectML** (AMD/Intel) — 备选

## 🔧 GPU 配置（可选）

```bash
# NVIDIA CUDA
pip install onnxruntime-gpu

# 或者 AMD/Intel DirectML
pip install onnxruntime-directml
```

脚本会自动检测可用的 GPU 提供商。所有预处理（图像去噪/缩放）在 CPU 上并行执行，GPU 只做推理，充分发挥硬件性能。

## 📄 输出格式

生成 `.srt` 文件，与视频同目录：

```srt
1
00:00:01,234 --> 00:00:04,567
这是第一句字幕

2
00:00:05,678 --> 00:00:08,901
这是第二句字幕
```

## 📋 依赖

- Python 3.10+
- rapidocr-onnxruntime >= 1.4.4
- opencv-python >= 4.13.0
- numpy
- onnxruntime-gpu (可选，GPU 加速)

## 📜 许可

MIT License
