#!/usr/bin/env python3
"""
Video Subtitle OCR Extractor GUI (视频字幕OCR提取工具 - 图形界面版)
===================================================================
功能: GUI框选视频字幕区域 → GPU加速逐帧OCR → 输出SRT字幕文件

依赖: opencv-python, rapidocr-onnxruntime, onnxruntime-gpu, numpy, pillow, tkinter
"""

import sys
import os

# ═══ 启动时自动注入 NVIDIA CUDA DLL 到 PATH（必须在 import onnxruntime 之前）═══
def _setup_cuda_path():
    """扫描 pip nvidia wheel 安装的 CUDA DLL 并添加到 DLL 搜索路径"""
    nvidia_root = os.path.join(os.environ.get('APPDATA', ''),
                               'Python', 'Python313', 'site-packages', 'nvidia')
    if os.path.isdir(nvidia_root):
        bins = [os.path.join(nvidia_root, d, 'bin')
                for d in os.listdir(nvidia_root)
                if os.path.isdir(os.path.join(nvidia_root, d, 'bin'))]
        for b in bins:
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(b)
                except OSError:
                    pass
            if b not in os.environ.get('PATH', ''):
                os.environ['PATH'] = b + ';' + os.environ.get('PATH', '')

_setup_cuda_path()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
import time
import os
import sys
import re
import json
from pathlib import Path
from datetime import timedelta
from difflib import SequenceMatcher


# ═══════════════════════════════════════════════════════════════
# OCR 模型预设
# ═══════════════════════════════════════════════════════════════

MODEL_PRESETS = {
    'PP-OCRv3': {
        'det': 'ch_PP-OCRv3_det_infer.onnx',
        'rec': 'ch_PP-OCRv3_rec_infer.onnx',
        'label': 'PP-OCRv3（轻量快速)',
    },
    'PP-OCRv4': {
        'det': 'ch_PP-OCRv4_det_infer.onnx',
        'rec': 'ch_PP-OCRv4_rec_infer.onnx',
        'label': 'PP-OCRv4（高精度)',
    },
    'PP-OCRv4 Server': {
        'det': 'ch_PP-OCRv4_det_infer.onnx',
        'rec': 'ch_PP-OCRv4_rec_server_infer.onnx',
        'label': 'PP-OCRv4 Server（最高精度)',
    },
}
DEFAULT_MODEL = 'PP-OCRv3'

# ═══════════════════════════════════════════════════════════════
# GPU 检测与引擎管理
# ═══════════════════════════════════════════════════════════════

class GpuManager:
    """GPU 加速检测与 OCR 引擎管理（单例 + 引擎预热缓存）"""

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._detect_gpu()
        self._engines = {}           # 缓存 RapidOCR 实例 {'full': ..., 'lite': ...}
        self._engine_ready = False
        self._engine_lock = threading.Lock()

    def _detect_gpu(self):
        """检测可用的 GPU 加速方案"""
        import onnxruntime as ort
        self.providers = ort.get_available_providers()
        self.gpu_provider = None
        self.gpu_label = "CPU"

        # 优先级: CUDA/TensorRT(NVIDIA) > DirectML(AMD/Intel) > OpenVINO
        gpu_order = [
            ('CUDAExecutionProvider', 'CUDA'),
            ('TensorrtExecutionProvider', 'TensorRT'),
            ('DmlExecutionProvider', 'DirectML'),
            ('OpenVINOExecutionProvider', 'OpenVINO'),
        ]
        for prov, label in gpu_order:
            if prov in self.providers:
                self.gpu_provider = prov
                self.gpu_label = label
                break

        self.gpu_available = self.gpu_provider is not None

        # 构建传递给 RapidOCR 的 GPU kwargs
        self._ocr_gpu_kwargs = {}
        if self.gpu_available:
            if self.gpu_provider == 'DmlExecutionProvider':
                self._ocr_gpu_kwargs = dict(
                    det_use_dml=True, cls_use_dml=True, rec_use_dml=True,
                    det_model_path='', cls_model_path='', rec_model_path='',
                )
            else:
                # CUDA / TensorRT → rapidocr_onnxruntime 通过 use_cuda 控制
                self._ocr_gpu_kwargs = dict(
                    det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True,
                    det_model_path='', cls_model_path='', rec_model_path='',
                )

    def create_engine(self, use_gpu: bool = True, chinese_lite: bool = False,
                      model_preset: str = DEFAULT_MODEL):
        """创建 RapidOCR 引擎实例（底层工厂，不缓存）"""
        from rapidocr_onnxruntime import RapidOCR
        preset = MODEL_PRESETS.get(model_preset, MODEL_PRESETS[DEFAULT_MODEL])
        kwargs = {
            'use_angle_cls': True,
            'det_model_path': preset['det'],
            'rec_model_path': preset['rec'],
        }
        if use_gpu:
            kwargs.update(self._ocr_gpu_kwargs)
        if chinese_lite:
            # 中文加速：关闭方向分类器（字幕永远水平，省 30% 推理）
            kwargs['use_cls'] = False
        return RapidOCR(**kwargs)

    def get_engine(self, chinese_lite: bool = False,
                   model_preset: str = DEFAULT_MODEL):
        """获取缓存的 RapidOCR 引擎（若未初始化则同步加载）"""
        cache_key = f"{model_preset}_{'lite' if chinese_lite else 'full'}"
        if self._engines.get(cache_key) is not None:
            return self._engines[cache_key]
        with self._engine_lock:
            if self._engines.get(cache_key) is not None:
                return self._engines[cache_key]
            engine = self.create_engine(self.gpu_available, chinese_lite,
                                        model_preset)
            self._engines[cache_key] = engine
            self._engine_ready = True
            return engine

    def warmup_async(self, on_ready=None, model_preset: str = DEFAULT_MODEL):
        """后台异步预热引擎（默认加载标准引擎），完成后回调 on_ready()"""
        def _load():
            self.get_engine(chinese_lite=False, model_preset=model_preset)
            if on_ready:
                on_ready()
        t = threading.Thread(target=_load, daemon=True)
        t.start()

    @property
    def engine_ready(self) -> bool:
        return self._engine_ready

    def status_text(self) -> str:
        if self.gpu_available:
            return f"⚡ GPU: {self.gpu_label} ✓"
        return "💻 CPU 模式"

    def status_color(self) -> str:
        return "#2ecc71" if self.gpu_available else "#e67e22"


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    chars = (
        r"\s"
        r",，。、．.!！?？;；:："
        r"\u201c\u201d\u2018\u2019"
        r"\u300c\u300d\u300e\u300f"
        r"\u3010\u3011"
        r"\uff08\uff09"
        r"()"
        r"\u2026\u2014\u2013"
        r"_/\\|@#$%^&*+=~`"
    )
    return re.sub(f"[{chars}]", '', text)


def text_similarity_fn(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ca, cb = clean_text(a), clean_text(b)
    if not ca or not cb:
        return 0.0
    if ca in cb or cb in ca:
        return 0.95
    return SequenceMatcher(None, ca, cb).ratio()


def frame_to_time_str(frame: int, fps: float) -> str:
    seconds = frame / fps
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)
    ms = td.microseconds // 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ═══════════════════════════════════════════════════════════════
# 图像预处理
# ═══════════════════════════════════════════════════════════════

class ImagePreprocessor:
    """字幕区域图像预处理 — 多策略流水线，最大化 OCR 准确率"""

    @staticmethod
    def preprocess(image: np.ndarray, mode: str = "auto") -> list[tuple[str, np.ndarray]]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if mode != "auto":
            # 非 auto 模式保持快速路径
            from collections import OrderedDict
            pass

        variants = []

        # 1. 原始灰度
        variants.append(("original", gray))

        # 2. 双边滤波去噪 (保边缘)
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        variants.append(("denoised", denoised))

        if mode in ("auto", "sharpened"):
            # 3. 锐化 (unsharp mask)
            blur = cv2.GaussianBlur(denoised, (0, 0), 3.0)
            sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
            variants.append(("sharpened", sharpened))

        if mode in ("auto", "binary"):
            # 4. OTSU 二值化 (去噪后)
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(binary) < 128:
                binary = cv2.bitwise_not(binary)
            variants.append(("binary", binary))

        if mode in ("auto", "enhanced"):
            # 5. CLAHE 对比度增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            variants.append(("enhanced", clahe.apply(denoised)))

        if mode in ("auto", "adaptive"):
            # 6-8. 多尺度自适应阈值
            for bs, c_val in [(9, 2), (15, 3), (21, 4)]:
                adp = cv2.adaptiveThreshold(denoised, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, bs, c_val)
                variants.append((f"adaptive_{bs}", adp))

        if mode in ("auto",):
            # 9. 形态学闭运算 (连接断裂笔画)
            if len(variants) >= 5:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                morph = cv2.morphologyEx(variants[4][1], cv2.MORPH_CLOSE, kernel)
                variants.append(("morph_closed", morph))

        return variants


# ═══════════════════════════════════════════════════════════════
# OCR 引擎
# ═══════════════════════════════════════════════════════════════


def _ocr_cleanup(text: str) -> str:
    """OCR 后处理：修正常见识别错误"""
    if not text:
        return text
    text = text.strip(".,;:!?。，、；：！？·… ")
    text = re.sub(r" {2,}", " ", text)
    return text


class OCREngine:
    """RapidOCR 封装 — 复用 GpuManager 缓存的全局引擎实例"""

    def __init__(self, use_gpu: bool = True, chinese_lite: bool = False,
                 model_preset: str = DEFAULT_MODEL,
                 min_text_score: float = 0.5):
        self.use_gpu = use_gpu
        self.chinese_lite = chinese_lite
        self.model_preset = model_preset
        self.min_text_score = min_text_score
        self._engine = None

    def init(self):
        if self._engine is not None:
            return
        mgr = GpuManager.instance()
        self._engine = mgr.get_engine(self.chinese_lite, self.model_preset)
        label = MODEL_PRESETS.get(self.model_preset, {}).get('label', self.model_preset)
        flag = ("GPU" if self.use_gpu and mgr.gpu_available else "CPU")
        extra = []
        if self.chinese_lite:
            extra.append('lite')
        if self.model_preset != DEFAULT_MODEL:
            extra.append(self.model_preset)
        if extra:
            flag += ', ' + ', '.join(extra)
        print(f"  [OK] RapidOCR ({flag})")

    @staticmethod
    def preprocess_all(crop, preprocess_mode="auto"):
        """预处理（纯 CPU）"""
        variants = ImagePreprocessor.preprocess(crop, preprocess_mode)
        h0, w0 = crop.shape[:2] if len(crop.shape) == 3 else crop.shape
        need_resize = (w0 < 300 or h0 < 40)
        target_scale = min(3.0, max(2.0, 300 / w0)) if need_resize else 1.0
        if need_resize:
            resized = []
            for mode_name, img in variants:
                img = cv2.resize(img, None, fx=target_scale, fy=target_scale,
                                 interpolation=cv2.INTER_LANCZOS4)
                resized.append((mode_name, img))
            return resized, False, 1.0
        return variants, False, 1.0

    def recognize_preprocessed(self, variants, need_resize=False, target_scale=1.0) -> str:
        """RapidOCR 识别 + 加权投票"""
        candidates = []

        for mode_name, img in variants:
            if need_resize:
                img = cv2.resize(img, None, fx=target_scale, fy=target_scale,
                                 interpolation=cv2.INTER_LANCZOS4)

            result, elapse = self._engine(img)
            if not result:
                continue

            for box, text, score in result:
                if not text or not text.strip():
                    continue
                text = text.strip()
                score_f = float(score)
                if score_f < self.min_text_score:
                    continue
                length_bonus = min(len(text) / 20.0, 1.0)
                weight = score_f * 0.7 + length_bonus * 0.3
                candidates.append((text, weight, score_f))

                # 渐进式：高置信度跳过后续变体
                if len(text) >= 4 and score_f >= 0.9:
                    return _ocr_cleanup(text)

        if not candidates:
            return ""

        candidates.sort(key=lambda x: (-x[1], -len(x[0])))
        best_text = candidates[0][0]

        if len(candidates) >= 3:
            from collections import Counter
            top_texts = [t for t, w, s in candidates[:4]]
            text_counter = Counter(top_texts)
            most_common, count = text_counter.most_common(1)[0]
            if count >= 2:
                best_text = most_common

        return _ocr_cleanup(best_text)

    def recognize(self, crop: np.ndarray, preprocess_mode: str = "auto") -> str:
        """简便入口"""
        variants, need_resize, target_scale = self.preprocess_all(crop, preprocess_mode)
        return self.recognize_preprocessed(variants, need_resize, target_scale)



def softmax_np(x):
    """NumPy softmax, axis=-1"""
    x_max = x.max(axis=-1, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / x_exp.sum(axis=-1, keepdims=True)

# ═══════════════════════════════════════════════════════════════
# 字幕提取器（后台线程版本）
# ═══════════════════════════════════════════════════════════════

class ExtractionWorker:
    """后台提取线程"""

    def __init__(self, video_path: str, roi: tuple,
                 frame_interval: int, similarity_threshold: float,
                 preprocess_mode: str, min_duration_ms: int,
                 min_text_len: int, use_gpu: bool,
                 chinese_lite: bool, num_workers: int = 4,
                 model_preset: str = DEFAULT_MODEL,
                 progress_callback=None, done_callback=None):
        self.video_path = video_path
        self.roi = roi
        self.frame_interval = frame_interval
        self.similarity_threshold = similarity_threshold
        self.preprocess_mode = preprocess_mode
        self.min_duration_ms = min_duration_ms
        self.min_text_len = min_text_len
        self.use_gpu = use_gpu
        self.chinese_lite = chinese_lite
        self.model_preset = model_preset
        self.num_workers = max(1, num_workers)
        self.progress_cb = progress_callback
        self.done_cb = done_callback
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            subtitles = self._extract()
            if not self._cancel.is_set():
                self.done_cb(subtitles, None)
        except Exception as e:
            self.done_cb([], str(e))

    def _extract(self) -> list:
        """单线程顺序读帧 + 多线程并行 OCR — 零 seek, 最快路径"""
        from queue import Queue

        n = self.num_workers
        x, y, rw, rh = self.roi
        interval = self.frame_interval

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 30.0

        total_scan = total_frames // interval + 1
        t_start = time.time()

        # ── 共享队列：reader → N workers ──
        frame_queue = Queue(maxsize=n * 3)

        # ── Reader 线程：顺序读取帧（cap.read() 零 seek）──
        def reader():
            nonlocal cap
            fn = 0
            try:
                while True:
                    if self._cancel.is_set():
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if fn % interval == 0:
                        crop = frame[y:y+rh, x:x+rw]
                        frame_queue.put((fn, crop))
                    fn += 1
            finally:
                for _ in range(n):
                    frame_queue.put(None)

        threading.Thread(target=reader, daemon=True).start()

        # ── Worker 线程：OCR ──
        hits_lock = threading.Lock()
        raw_hits = []
        scanned = [0]

        def worker():
            engine = OCREngine(self.use_gpu, self.chinese_lite,
                               self.model_preset)
            engine.init()
            while True:
                item = frame_queue.get()
                if item is None:
                    frame_queue.task_done()
                    break
                fn, crop = item
                try:
                    text = engine.recognize(crop, self.preprocess_mode)
                    if text and len(text) >= self.min_text_len:
                        with hits_lock:
                            raw_hits.append((fn, text))
                finally:
                    with hits_lock:
                        scanned[0] += 1
                    frame_queue.task_done()

        workers = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
        for w in workers:
            w.start()

        # ── 进度轮询 ──
        last_update = 0
        while any(w.is_alive() for w in workers):
            time.sleep(0.12)
            now = time.time()
            if now - last_update > 0.25:
                elapsed = now - t_start
                s = scanned[0]
                pct = min(s, total_scan) / total_scan * 100
                eta = elapsed / max(s, 1) * (total_scan - s) if s > 0 else 999
                self.progress_cb({
                    'percent': pct,
                    'scanned': s,
                    'skipped': 0,
                    'total_scan': total_scan,
                    'hits': len(raw_hits),
                    'elapsed': elapsed,
                    'eta': eta,
                    'current_frame': 0,
                    'total_frames': total_frames,
                    'workers': n,
                })
                last_update = now

        for w in workers:
            w.join()
        cap.release()

        if self._cancel.is_set():
            return []

        if not raw_hits:
            return []

        raw_hits.sort(key=lambda x: x[0])

        segments = self._merge_hits(raw_hits, fps)
        segments = self._post_process(segments, fps)

        return segments

    def _merge_hits(self, hits: list, fps: float) -> list:
        if not hits:
            return []

        segments = []
        batch_start = hits[0][0]
        batch_end = hits[0][0]
        best_text = hits[0][1]
        gap_threshold = max(3, int(fps * 0.8))

        for i in range(1, len(hits)):
            fn, text = hits[i]
            prev_fn = hits[i - 1][0]
            gap = fn - prev_fn
            sim = text_similarity_fn(best_text, text)

            if sim >= self.similarity_threshold:
                batch_end = fn
                if len(text) > len(best_text):
                    best_text = text
            elif gap <= gap_threshold and sim >= 0.5:
                batch_end = fn
                if len(text) > len(best_text):
                    best_text = text
            else:
                segments.append((batch_start, batch_end, best_text))
                batch_start = fn
                batch_end = fn
                best_text = text

        segments.append((batch_start, batch_end, best_text))
        return segments

    def _post_process(self, segments: list, fps: float) -> list:
        if not segments:
            return []

        min_frames = int(self.min_duration_ms / 1000 * fps)

        # 1. 按最短持续时间过滤
        filtered = [(s, e, t) for s, e, t in segments
                    if e - s >= min_frames and t.strip()]

        # 2. 合并相邻相同/高度相似文本
        merged = []
        for sub in filtered:
            if not merged:
                merged.append(sub)
                continue
            ps, pe, pt = merged[-1]
            gap_ms = (sub[0] - pe) / fps * 1000
            sim = text_similarity_fn(pt, sub[2])
            if sim >= 0.9 and gap_ms < 1500:
                merged[-1] = (ps, sub[1],
                              sub[2] if len(sub[2]) > len(pt) else pt)
            else:
                merged.append(sub)

        # 3. 消除过短间隙
        final = []
        for i, sub in enumerate(merged):
            if i == 0:
                final.append(sub)
                continue
            ps, pe, pt = final[-1]
            gap_ms = (sub[0] - pe) / fps * 1000
            if gap_ms < 300 and text_similarity_fn(pt, sub[2]) >= 0.85:
                final[-1] = (ps, sub[1],
                             sub[2] if len(sub[2]) > len(pt) else pt)
            else:
                final.append(sub)

        return final


# ═══════════════════════════════════════════════════════════════
# ROI 绘制 Canvas
# ═══════════════════════════════════════════════════════════════

class ROICanvas(tk.Canvas):
    """支持鼠标拖拽框选 ROI 的视频预览 Canvas"""

    def __init__(self, parent, roi_change_callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.roi = None  # (x, y, w, h) in canvas coords
        self._frame_orig = None       # 原始帧 (numpy)
        self._photo = None            # tkinter PhotoImage
        self._photo_id = None         # canvas image item id
        self._rect_id = None          # canvas rect item id
        self._scale = 1.0             # 缩放比例
        self._offset_x = 0
        self._offset_y = 0
        self._drag_start = None
        self._drag_rect = None
        self._roi_change_cb = roi_change_callback

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Button-3>", self._clear_roi)
        self.bind("<Configure>", self._on_resize)

    def _notify_roi_change(self):
        if self._roi_change_cb:
            self._roi_change_cb(self.get_frame_roi())

    def set_frame(self, frame: np.ndarray):
        """设置要显示的帧 (BGR numpy array)"""
        self._frame_orig = frame.copy()
        self._render()

    def _render(self):
        """渲染帧到 canvas"""
        if self._frame_orig is None:
            return

        cw = self.winfo_width() or 640
        ch = self.winfo_height() or 360
        fh, fw = self._frame_orig.shape[:2]

        # 等比缩放
        self._scale = min(cw / fw, ch / fh)
        new_w = int(fw * self._scale)
        new_h = int(fh * self._scale)
        self._offset_x = (cw - new_w) // 2
        self._offset_y = (ch - new_h) // 2

        resized = cv2.resize(self._frame_orig, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)

        if self._photo_id:
            self.delete(self._photo_id)
        self._photo_id = self.create_image(
            self._offset_x, self._offset_y,
            anchor='nw', image=self._photo
        )

        # 重绘 ROI 矩形
        self._draw_roi()

    def _on_resize(self, event):
        if self._frame_orig is not None:
            self._render()

    def _canvas_to_frame(self, cx, cy):
        """Canvas 坐标 → 原始帧坐标"""
        fx = (cx - self._offset_x) / self._scale
        fy = (cy - self._offset_y) / self._scale
        return int(fx), int(fy)

    def _on_press(self, event):
        self._clear_roi()
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        sx, sy = self._drag_start
        if self._drag_rect:
            self.delete(self._drag_rect)
        self._drag_rect = self.create_rectangle(
            sx, sy, event.x, event.y,
            outline='#00ff00', width=2, dash=(5, 3)
        )

    def _on_release(self, event):
        if self._drag_start is None:
            return

        sx, sy = self._drag_start
        ex, ey = event.x, event.y

        # 确保 x1<x2, y1<y2
        cx1, cx2 = sorted([sx, ex])
        cy1, cy2 = sorted([sy, ey])
        cw, ch = cx2 - cx1, cy2 - cy1

        if cw < 5 or ch < 5:
            self._clear_roi()
            self._drag_start = None
            return

        self.roi = (cx1, cy1, cw, ch)
        self._drag_start = None

        if self._drag_rect:
            self.delete(self._drag_rect)
            self._drag_rect = None

        self._draw_roi()
        self._notify_roi_change()

    def _draw_roi(self):
        if self._rect_id:
            self.delete(self._rect_id)
            self._rect_id = None

        if self.roi is None:
            return

        x1, y1, w, h = self.roi
        self._rect_id = self.create_rectangle(
            x1, y1, x1 + w, y1 + h,
            outline='#00ff00', width=2,
            dash=(5, 3)
        )
        # 标签
        self.create_text(x1 + 4, y1 - 10 if y1 > 15 else y1 + h + 15,
                         text=f"{w}×{h}", anchor='w',
                         fill='#00ff00', font=('', 9))

    def _clear_roi(self, event=None):
        self.roi = None
        if self._rect_id:
            self.delete(self._rect_id)
            self._rect_id = None
        self._notify_roi_change()

    def get_frame_roi(self) -> tuple | None:
        """获取原始帧坐标系的 ROI (x, y, w, h)，或 None"""
        if self.roi is None or self._frame_orig is None:
            return None
        cx1, cy1, cw, ch = self.roi
        fx, fy = self._canvas_to_frame(cx1, cy1)
        fw = int(cw / self._scale)
        fh = int(ch / self._scale)
        # 裁剪到帧内
        fh_img, fw_img = self._frame_orig.shape[:2]
        fx = max(0, fx)
        fy = max(0, fy)
        fw = min(fw, fw_img - fx)
        fh = min(fh, fh_img - fy)
        if fw < 2 or fh < 2:
            return None
        return (fx, fy, fw, fh)


# ═══════════════════════════════════════════════════════════════
# 主应用窗口
# ═══════════════════════════════════════════════════════════════

class SubtitleOCRApp:
    """视频字幕 OCR 提取 GUI 应用"""

    WINDOW_TITLE = "视频字幕OCR提取工具"
    MIN_WIDTH, MIN_HEIGHT = 1100, 650
    DEFAULT_WIDTH, DEFAULT_HEIGHT = 1200, 750

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(self.WINDOW_TITLE)
        self.root.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.root.geometry(f"{self.DEFAULT_WIDTH}x{self.DEFAULT_HEIGHT}")

        # 变量
        self.video_path = tk.StringVar()
        self.frame_interval = tk.IntVar(value=10)
        self.similarity_threshold = tk.DoubleVar(value=0.85)
        self.min_duration = tk.IntVar(value=500)
        self.min_text_len = tk.IntVar(value=2)
        self.preprocess_mode = tk.StringVar(value="auto")
        self.use_gpu = tk.BooleanVar(value=True)
        self.num_workers = tk.IntVar(value=4)

        self.cap = None
        self.fps = 0.0
        self.total_frames = 0
        self.video_resolution = ""
        self.subtitles = []  # 提取结果
        self.worker = None
        self._progress_queue = queue.Queue()

        self._build_ui()
        self._update_gpu_status()
        self._poll_progress()

        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 后台预热 OCR 引擎（避免首次提取时等待）
        self._engine_loading = True
        self.statusbar.configure(text="正在预热 OCR 引擎...")
        GpuManager.instance().warmup_async(on_ready=self._on_engine_ready)

    # ── UI 构建 ──────────────────────────────────────────

    def _build_ui(self):
        # 主容器
        main = ttk.Frame(self.root)
        main.pack(fill='both', expand=True)

        # 顶部工具栏
        self._build_toolbar(main)

        # 主体：左侧预览 + 右侧控制
        body = ttk.Frame(main)
        body.pack(fill='both', expand=True, padx=8, pady=4)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._build_preview_panel(body)
        self._build_control_panel(body)

        # 底部状态栏
        self._build_statusbar(main)

    def _build_toolbar(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill='x', padx=8, pady=(6, 0))

        # GPU 状态
        self.gpu_label = ttk.Label(bar, text="", font=('', 10, 'bold'))
        self.gpu_label.pack(side='left', padx=(0, 20))

        # 标题
        ttk.Label(bar, text="视频字幕OCR提取工具",
                  font=('Microsoft YaHei UI', 13, 'bold')).pack(side='left')

        # 版本
        ttk.Label(bar, text="RapidOCR + DirectML",
                  foreground='#888').pack(side='left', padx=(10, 0))

    def _build_preview_panel(self, parent):
        """左侧：视频预览"""
        frame = ttk.LabelFrame(parent, text=" 视频预览 - 鼠标拖拽框选字幕区域 ",
                               padding=4)
        frame.grid(row=0, column=0, sticky='nsew', padx=(0, 4))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas = ROICanvas(frame,
                                roi_change_callback=self._on_roi_changed,
                                bg='#1a1a2e',
                                highlightthickness=1,
                                highlightbackground='#444')
        self.canvas.grid(row=0, column=0, sticky='nsew')

        # 占位提示
        self.canvas.create_text(
            320, 180, text="请选择视频文件",
            fill='#666', font=('Microsoft YaHei UI', 14),
            tags=('placeholder',)
        )

    def _build_control_panel(self, parent):
        """右侧：控制面板"""
        panel = ttk.Frame(parent)
        panel.grid(row=0, column=1, sticky='nsew')
        panel.columnconfigure(0, weight=1)

        row = 0

        # ── 视频选择 ──
        f = ttk.LabelFrame(panel, text=" 视频文件 ", padding=6)
        f.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        f.columnconfigure(1, weight=1)
        row += 1

        ttk.Button(f, text="选择视频...", command=self._select_video,
                   width=10).grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(f, textvariable=self.video_path, state='readonly',
                  font=('', 8)).grid(row=0, column=1, sticky='ew')

        self.video_info_label = ttk.Label(f, text="尚未选择视频",
                                          foreground='#888')
        self.video_info_label.grid(row=1, column=0, columnspan=2,
                                   sticky='w', pady=(4, 0))

        # ── 参数设置 ──
        f = ttk.LabelFrame(panel, text=" 识别参数 ", padding=6)
        f.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        f.columnconfigure(1, weight=1)
        row += 1

        params = [
            ("帧扫描间隔", self.frame_interval, 1, 60, "每N帧扫描一次，越小越精细但越慢"),
            ("相似度阈值", self.similarity_threshold, 0.50, 1.00,
             "越高去重越严格 (0~1)", "%.2f"),
            ("最短持续(ms)", self.min_duration, 100, 5000,
             "字幕最短持续时间，过滤碎片"),
            ("最短文本(字)", self.min_text_len, 1, 10,
             "最少字符数，过滤噪音"),
        ]

        for i, (label, var, vmin, vmax, tip, *fmt) in enumerate(params):
            ttk.Label(f, text=label + ":", font=('', 9)).grid(
                row=i, column=0, sticky='w', padx=(0, 6), pady=2)
            sb = ttk.Spinbox(f, textvariable=var, from_=vmin, to=vmax,
                             width=8, font=('', 9))
            if fmt:
                sb.configure(format=fmt[0], increment=0.05 if isinstance(var.get(), float) else 10)
            sb.grid(row=i, column=1, sticky='w', pady=2)
            self._create_tooltip(sb, tip)

        # 预处理模式
        ttk.Label(f, text="预处理:", font=('', 9)).grid(
            row=len(params), column=0, sticky='w', padx=(0, 6), pady=2)
        cb = ttk.Combobox(f, textvariable=self.preprocess_mode,
                          values=['auto', 'binary', 'adaptive', 'enhanced', 'original'],
                          state='readonly', width=10)
        cb.grid(row=len(params), column=1, sticky='w', pady=2)

        # GPU 开关
        ttk.Checkbutton(f, text="GPU 加速",
                        variable=self.use_gpu).grid(
            row=len(params) + 1, column=0, columnspan=2,
            sticky='w', pady=(4, 0))

        # 中文加速开关
        self.chinese_lite = tk.BooleanVar(value=False)
        cb_lite = ttk.Checkbutton(f, text="⚡ 中文加速 (关闭方向分类, 提速 ~30%)",
                                   variable=self.chinese_lite)
        cb_lite.grid(row=len(params) + 2, column=0, columnspan=2,
                      sticky='w', pady=(2, 0))

        # OCR 模型选择
        f = ttk.Frame(panel)
        f.grid(row=len(params) + 3, column=0, columnspan=2,
               sticky='ew', pady=(6, 0))
        ttk.Label(f, text="🧠 OCR 模型:").pack(side='left')
        self.model_preset = tk.StringVar(value=DEFAULT_MODEL)
        model_names = list(MODEL_PRESETS.keys())
        model_labels = [MODEL_PRESETS[k]['label'] for k in model_names]
        cb_model = ttk.Combobox(f, textvariable=self.model_preset,
                                values=model_labels, state='readonly', width=26)
        cb_model.current(0)
        cb_model.pack(side='left', padx=(6, 0))
        # 将 label 映射回 preset key
        self._model_label_map = dict(zip(model_labels, model_names))

        # 并行 worker 数量
        f = ttk.Frame(panel)
        f.grid(row=len(params) + 4, column=0, columnspan=2,
               sticky='ew', pady=(6, 0))
        ttk.Label(f, text="🚀 并行线程数:").pack(side='left')
        ttk.Spinbox(f, from_=1, to=8, textvariable=self.num_workers,
                    width=3).pack(side='left', padx=(6, 0))
        ttk.Label(f, text="(4=推荐, RTX 3060+)  |  8=榨干性能").pack(side='left', padx=(4, 0))

        # ── 操作按钮 ──
        f = ttk.Frame(panel)
        f.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        row += 1

        self.btn_start = ttk.Button(f, text="▶ 开始提取",
                                    command=self._start_extraction)
        self.btn_start.grid(row=0, column=0, sticky='ew', padx=(0, 3))

        self.btn_cancel = ttk.Button(f, text="⏹ 取消",
                                     command=self._cancel_extraction,
                                     state='disabled')
        self.btn_cancel.grid(row=0, column=1, sticky='ew', padx=(3, 0))

        # ── 进度 ──
        f = ttk.LabelFrame(panel, text=" 提取进度 ", padding=6)
        f.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        f.columnconfigure(0, weight=1)
        row += 1

        self.progress_bar = ttk.Progressbar(f, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky='ew')

        self.progress_text = ttk.Label(f, text="就绪", font=('', 8))
        self.progress_text.grid(row=1, column=0, sticky='w', pady=(2, 0))

        # ── 字幕预览 ──
        f = ttk.LabelFrame(panel, text=" 识别结果 ", padding=4)
        f.grid(row=row, column=0, sticky='nsew', pady=(0, 4))
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)
        panel.rowconfigure(row, weight=1)
        row += 1

        # 文本框 + 滚动条
        text_frame = ttk.Frame(f)
        text_frame.grid(row=0, column=0, sticky='nsew')
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        self.result_text = tk.Text(text_frame, wrap='word', state='disabled',
                                   font=('Consolas', 9), height=8,
                                   bg='#1e1e2e', fg='#cdd6f4',
                                   insertbackground='white',
                                   relief='flat', borderwidth=0)
        self.result_text.grid(row=0, column=0, sticky='nsew')

        scrollbar = ttk.Scrollbar(text_frame, orient='vertical',
                                  command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.result_text.configure(yscrollcommand=scrollbar.set)

        # 导出按钮
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        self.btn_save = ttk.Button(btn_frame, text="💾 保存 SRT",
                                   command=self._save_srt, state='disabled')
        self.btn_save.pack(side='left', padx=(0, 6))
        self.btn_copy = ttk.Button(btn_frame, text="📋 复制全部",
                                   command=self._copy_all, state='disabled')
        self.btn_copy.pack(side='left')

    def _build_statusbar(self, parent):
        self.statusbar = ttk.Label(parent, text="就绪",
                                   relief='sunken', anchor='w',
                                   padding=(8, 2))
        self.statusbar.pack(fill='x', padx=8, pady=(0, 6))

    def _create_tooltip(self, widget, text):
        """简单 tooltip"""
        tip = None

        def enter(e):
            nonlocal tip
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 2
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x}+{y}")
            ttk.Label(tip, text=text, background='#ffffcc',
                      relief='solid', borderwidth=1,
                      font=('', 8), padding=(4, 2)).pack()
            tip.attributes('-topmost', True)

        def leave(e):
            nonlocal tip
            if tip:
                tip.destroy()
                tip = None

        widget.bind('<Enter>', enter, add='+')
        widget.bind('<Leave>', leave, add='+')

    # ── GPU 状态 ─────────────────────────────────────────

    def _on_engine_ready(self):
        """引擎预热完成回调（后台线程 → 主线程）"""
        self._engine_loading = False
        self.root.after(0, lambda: self.statusbar.configure(
            text="就绪 — 请选择视频文件"))

    def _update_gpu_status(self):
        gm = GpuManager.instance()
        self.gpu_label.configure(
            text=gm.status_text(),
            foreground=gm.status_color()
        )
        if not gm.gpu_available:
            self.use_gpu.set(False)

    def _on_roi_changed(self, roi):
        """ROI 区域变化回调"""
        self._update_button_states()

    # ── 视频选择 ─────────────────────────────────────────

    def _select_video(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv *.webm *.m4v"),
                ("所有文件", "*.*"),
            ]
        )
        if not path:
            return
        self.video_path.set(path)
        self._load_video_info()

    def _load_video_info(self):
        """加载视频信息并显示预览帧"""
        path = self.video_path.get()
        if not path or not os.path.exists(path):
            return

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", f"无法打开视频:\n{path}")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = self.total_frames / self.fps if self.fps > 0 else 0

        self.video_resolution = f"{w}×{h}"
        info = f"📐 {w}×{h}  |  🎬 {self.fps:.1f} FPS  |  ⏱️ {timedelta(seconds=int(duration))}  |  📦 {self.total_frames} 帧"
        self.video_info_label.configure(text=info)

        # 加载预览帧（10% 位置）
        target = max(0, int(self.total_frames * 0.1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret:
            # 清除占位符
            self.canvas.delete('placeholder')
            self.canvas.set_frame(frame)

        self.statusbar.configure(text=f"已加载: {os.path.basename(path)} — 请在左侧画框选字幕区域")
        self._update_button_states()

    # ── 按钮状态 ─────────────────────────────────────────

    def _update_button_states(self):
        can_start = bool(self.video_path.get() and self.canvas.roi is not None)
        self.btn_start.configure(state='normal' if can_start else 'disabled')

    # ── 提取控制 ─────────────────────────────────────────

    def _start_extraction(self):
        roi = self.canvas.get_frame_roi()
        if roi is None:
            messagebox.showwarning("提示", "请先在视频预览中框选字幕区域")
            return

        path = self.video_path.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("提示", "请先选择视频文件")
            return

        # 清除旧结果
        self.subtitles = []
        self._clear_results()

        # UI 状态切换
        self.btn_start.configure(state='disabled')
        self.btn_cancel.configure(state='normal')
        self.btn_save.configure(state='disabled')
        self.btn_copy.configure(state='disabled')
        self.progress_bar['value'] = 0
        self.progress_text.configure(text="正在扫描视频帧...")
        self.statusbar.configure(text="正在提取字幕...")

        # 启动后台线程
        self.worker = ExtractionWorker(
            video_path=path,
            roi=roi,
            frame_interval=self.frame_interval.get(),
            similarity_threshold=self.similarity_threshold.get(),
            preprocess_mode=self.preprocess_mode.get(),
            min_duration_ms=self.min_duration.get(),
            min_text_len=self.min_text_len.get(),
            use_gpu=self.use_gpu.get(),
            chinese_lite=self.chinese_lite.get(),
            model_preset=self._model_label_map.get(
                self.model_preset.get(), DEFAULT_MODEL),
            progress_callback=lambda data: self._progress_queue.put(('progress', data)),
            done_callback=lambda subs, err: self._progress_queue.put(('done', (subs, err))),
            num_workers=self.num_workers.get(),
        )
        thread = threading.Thread(target=self.worker.run, daemon=True)
        thread.start()

    def _cancel_extraction(self):
        if self.worker:
            self.worker.cancel()
        self.statusbar.configure(text="已取消")
        self._reset_ui_state()
        self.progress_text.configure(text="已取消")

    def _poll_progress(self):
        """定时轮询进度队列，更新 UI"""
        try:
            while True:
                msg_type, data = self._progress_queue.get_nowait()
                if msg_type == 'progress':
                    self._update_progress_ui(data)
                elif msg_type == 'done':
                    self._on_extraction_done(data[0], data[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_progress)

    def _update_progress_ui(self, data):
        self.progress_bar['value'] = data['percent']
        eta_str = str(timedelta(seconds=int(data['eta']))) if data['eta'] < 86400 else "..."
        elapsed_str = str(timedelta(seconds=int(data['elapsed'])))
        skip_info = f"跳过: {data.get('skipped', 0)} | " if data.get('skipped', 0) > 0 else ""
        self.progress_text.configure(
            text=f"已扫: {data['scanned']}/{data['total_scan']}帧 | "
                 f"{skip_info}"
                 f"命中: {data['hits']}条 | "
                 f"已耗时: {elapsed_str} | "
                 f"剩余: {eta_str}"
        )
        self.statusbar.configure(
            text=f"提取中... {data['percent']:.0f}% | "
                 f"帧 {data['current_frame']}/{data['total_frames']}"
        )

    def _on_extraction_done(self, subtitles, error):
        self._reset_ui_state()

        if error:
            messagebox.showerror("提取失败", f"发生错误:\n{error}")
            self.statusbar.configure(text=f"错误: {error}")
            return

        self.subtitles = subtitles

        if not subtitles:
            self.statusbar.configure(text="未检测到字幕文本")
            self._show_no_result()
            return

        self._show_results(subtitles)
        self.statusbar.configure(
            text=f"✅ 提取完成 | 共 {len(subtitles)} 条字幕 | "
                 f"可直接保存或复制"
        )

    def _reset_ui_state(self):
        self.btn_start.configure(state='normal')
        self.btn_cancel.configure(state='disabled')
        self._update_button_states()

    # ── 结果显示 ─────────────────────────────────────────

    def _show_results(self, subtitles):
        self.result_text.configure(state='normal')
        self.result_text.delete('1.0', 'end')

        show_count = min(len(subtitles), 100)  # 最多显示100条
        for i, (s, e, text) in enumerate(subtitles[:show_count], 1):
            t1 = frame_to_time_str(s, self.fps)
            t2 = frame_to_time_str(e, self.fps)
            self.result_text.insert('end', f"{i:4d}  {t1} → {t2}\n", 'time')
            self.result_text.insert('end', f"      {text}\n\n", 'text')

        if len(subtitles) > show_count:
            self.result_text.insert('end',
                                    f"... 共 {len(subtitles)} 条，显示前 {show_count} 条\n",
                                    'info')

        # 样式
        self.result_text.tag_configure('time', foreground='#89b4fa',
                                       font=('Consolas', 9, 'bold'))
        self.result_text.tag_configure('text', foreground='#cdd6f4',
                                       font=('Microsoft YaHei UI', 10))
        self.result_text.tag_configure('info', foreground='#6c7086',
                                       font=('', 9, 'italic'))
        self.result_text.configure(state='disabled')
        self.result_text.see('1.0')

        self.btn_save.configure(state='normal')
        self.btn_copy.configure(state='normal')

    def _show_no_result(self):
        self.result_text.configure(state='normal')
        self.result_text.delete('1.0', 'end')
        self.result_text.insert('end',
                                "⚠️ 未检测到字幕文本\n\n"
                                "建议:\n"
                                "  1. 重新框选，确保覆盖字幕区域\n"
                                "  2. 减小帧扫描间隔 (如 3)\n"
                                "  3. 尝试切换预处理模式 (binary/adaptive)\n"
                                "  4. 检查视频字幕是否清晰可见\n",
                                'info')
        self.result_text.configure(state='disabled')

    def _clear_results(self):
        self.result_text.configure(state='normal')
        self.result_text.delete('1.0', 'end')
        self.result_text.configure(state='disabled')
        self.btn_save.configure(state='disabled')
        self.btn_copy.configure(state='disabled')

    # ── 导出操作 ─────────────────────────────────────────

    def _save_srt(self):
        if not self.subtitles:
            return

        default_name = Path(self.video_path.get()).stem + ".srt"
        path = filedialog.asksaveasfilename(
            title="保存 SRT 字幕",
            defaultextension=".srt",
            initialfile=default_name,
            filetypes=[("SRT字幕", "*.srt"), ("所有文件", "*.*")],
        )
        if not path:
            return

        with open(path, 'w', encoding='utf-8') as f:
            for i, (s, e, text) in enumerate(self.subtitles, 1):
                f.write(f"{i}\n")
                f.write(f"{frame_to_time_str(s, self.fps)} --> "
                        f"{frame_to_time_str(e, self.fps)}\n")
                f.write(f"{text}\n\n")

        self.statusbar.configure(
            text=f"💾 已保存: {path} ({len(self.subtitles)} 条)"
        )
        messagebox.showinfo("保存成功",
                            f"字幕已保存到:\n{path}\n共 {len(self.subtitles)} 条")

    def _copy_all(self):
        if not self.subtitles:
            return

        lines = []
        for i, (s, e, text) in enumerate(self.subtitles, 1):
            t1 = frame_to_time_str(s, self.fps)
            t2 = frame_to_time_str(e, self.fps)
            lines.append(f"{i}")
            lines.append(f"{t1} --> {t2}")
            lines.append(text)
            lines.append("")

        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(lines))
        self.statusbar.configure(text="📋 已复制全部字幕到剪贴板")

    # ── 生命周期 ─────────────────────────────────────────

    def _on_close(self):
        if self.worker:
            self.worker.cancel()
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    # 高DPI适配
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = SubtitleOCRApp()
    app.run()


if __name__ == "__main__":
    main()
