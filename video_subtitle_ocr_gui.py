#!/usr/bin/env python3
"""
Video Subtitle OCR Extractor GUI (视频字幕OCR提取工具 - 图形界面版)
===================================================================
功能: 框选字幕区域 → 播放器预览 → GPU加速OCR → 字幕叠加 + 可编辑 → 输出SRT

依赖: opencv-python, rapidocr-onnxruntime, onnxruntime-gpu, numpy, pillow, tkinter
"""

import sys, os, re, json, time, queue, threading, subprocess
from pathlib import Path
from datetime import timedelta
from difflib import SequenceMatcher

# ═══ 启动时自动注入 NVIDIA CUDA DLL ═══
def _setup_cuda_path():
    nvidia_root = os.path.join(os.environ.get('APPDATA', ''),
                               'Python', 'Python313', 'site-packages', 'nvidia')
    if os.path.isdir(nvidia_root):
        bins = [os.path.join(nvidia_root, d, 'bin')
                for d in os.listdir(nvidia_root)
                if os.path.isdir(os.path.join(nvidia_root, d, 'bin'))]
        for b in bins:
            if hasattr(os, 'add_dll_directory'):
                try: os.add_dll_directory(b)
                except OSError: pass
            if b not in os.environ.get('PATH', ''):
                os.environ['PATH'] = b + ';' + os.environ.get('PATH', '')
_setup_cuda_path()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ctypes import c_uint8, c_void_p, cast

# ═══ VLC DLL ═══
def _setup_vlc(path=r'C:\Program Files\VideoLAN\VLC'):
    if os.path.isdir(path):
        try: os.add_dll_directory(path)
        except Exception: pass
        if path not in os.environ.get('PATH', ''):
            os.environ['PATH'] = path + ';' + os.environ.get('PATH', '')
_setup_vlc()
import vlc
import ctypes as _ctypes
_user32 = _ctypes.windll.user32
_EnumWindows = _user32.EnumWindows
_EnumWindowsProc = _ctypes.WINFUNCTYPE(_ctypes.c_bool, _ctypes.c_int, _ctypes.c_int)
_GetWindowTextW = _user32.GetWindowTextW
_GetWindowTextLengthW = _user32.GetWindowTextLengthW
_IsWindowVisible = _user32.IsWindowVisible
_ShowWindow = _user32.ShowWindow
_SW_HIDE = 0

def _hide_vlc_windows():
    """隐藏 VLC 弹出的独立视频窗口（避免和 Canvas 双画面）"""
    hidden = []
    def _cb(hwnd, _):
        try:
            length = _GetWindowTextLengthW(hwnd)
            if length == 0: return True
            buff = _ctypes.create_unicode_buffer(length + 1)
            _GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value
            if 'vlc' in title.lower() and _IsWindowVisible(hwnd):
                _ShowWindow(hwnd, _SW_HIDE)
                hidden.append(hwnd)
        except Exception: pass
        return True
    _EnumWindows(_EnumWindowsProc(_cb), 0)
    return hidden


# ═══ VLC 自定义视频帧缓冲（音视频统一由 VLC 驱动）═══
class VlcFrameBuffer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pitch = width * 4
        self.size = height * self.pitch
        self.buf = (c_uint8 * self.size)()
        self.queue = queue.Queue(maxsize=2)

        @vlc.CallbackDecorators.VideoLockCb
        def lock(opaque, planes):
            planes[0] = cast(self.buf, c_void_p)
            return None

        @vlc.CallbackDecorators.VideoUnlockCb
        def unlock(opaque, picture, planes):
            pass

        @vlc.CallbackDecorators.VideoDisplayCb
        def display(opaque, picture):
            try:
                arr = np.frombuffer(self.buf, dtype=np.uint8).copy()
                arr = arr.reshape((self.height, self.width, 4))
                if self.queue.full():
                    self.queue.get_nowait()
                self.queue.put(arr, block=False)
            except Exception:
                pass

        self.lock = lock
        self.unlock = unlock
        self.display = display

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
# GPU 检测与引擎管理 (unchanged)
# ═══════════════════════════════════════════════════════════════

class GpuManager:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._detect_gpu()
        self._engines = {}
        self._engine_ready = False
        self._engine_lock = threading.Lock()

    def _detect_gpu(self):
        import onnxruntime as ort
        self.providers = ort.get_available_providers()
        self.gpu_provider = None
        self.gpu_label = "CPU"
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
        self._ocr_gpu_kwargs = {}
        if self.gpu_available:
            if self.gpu_provider == 'DmlExecutionProvider':
                self._ocr_gpu_kwargs = dict(
                    det_use_dml=True, cls_use_dml=True, rec_use_dml=True,
                    det_model_path='', cls_model_path='', rec_model_path='')
            else:
                self._ocr_gpu_kwargs = dict(
                    det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True,
                    det_model_path='', cls_model_path='', rec_model_path='')

    def create_engine(self, use_gpu=True, chinese_lite=False,
                      model_preset=DEFAULT_MODEL):
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
            kwargs['use_cls'] = False
        return RapidOCR(**kwargs)

    def get_engine(self, chinese_lite=False, model_preset=DEFAULT_MODEL):
        cache_key = f"{model_preset}_{'lite' if chinese_lite else 'full'}"
        if self._engines.get(cache_key) is not None:
            return self._engines[cache_key]
        with self._engine_lock:
            if self._engines.get(cache_key) is not None:
                return self._engines[cache_key]
            engine = self.create_engine(self.gpu_available, chinese_lite, model_preset)
            self._engines[cache_key] = engine
            self._engine_ready = True
            return engine

    def warmup_async(self, on_ready=None, model_preset=DEFAULT_MODEL):
        def _load():
            self.get_engine(chinese_lite=False, model_preset=model_preset)
            if on_ready:
                on_ready()
        t = threading.Thread(target=_load, daemon=True)
        t.start()

    @property
    def engine_ready(self):
        return self._engine_ready

    def status_text(self):
        if self.gpu_available:
            return f"⚡ GPU: {self.gpu_label} ✓"
        return "💻 CPU 模式"

    def status_color(self):
        return "#2ecc71" if self.gpu_available else "#e67e22"


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def clean_text(text):
    chars = (r"\s" r",，。、．.!！?？;；:：" r"\u201c\u201d\u2018\u2019"
             r"\u300c\u300d\u300e\u300f" r"\u3010\u3011" r"\uff08\uff09"
             r"()" r"\u2026\u2014\u2013" r"_/\\|@#$%^&*+=~`")
    return re.sub(f"[{chars}]", '', text)

def text_similarity_fn(a, b):
    if not a or not b: return 0.0
    ca, cb = clean_text(a), clean_text(b)
    if not ca or not cb: return 0.0
    if ca in cb or cb in ca: return 0.95
    return SequenceMatcher(None, ca, cb).ratio()

def frame_to_time_str(frame, fps):
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
    @staticmethod
    def preprocess(image, mode="auto"):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        variants = [("original", gray)]
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        variants.append(("denoised", denoised))
        if mode in ("auto", "sharpened"):
            blur = cv2.GaussianBlur(denoised, (0, 0), 3.0)
            sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
            variants.append(("sharpened", sharpened))
        if mode in ("auto", "binary"):
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(binary) < 128:
                binary = cv2.bitwise_not(binary)
            variants.append(("binary", binary))
        if mode in ("auto", "enhanced"):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            variants.append(("enhanced", clahe.apply(denoised)))
        if mode in ("auto", "adaptive"):
            for bs, c_val in [(9, 2), (15, 3), (21, 4)]:
                adp = cv2.adaptiveThreshold(denoised, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, bs, c_val)
                variants.append((f"adaptive_{bs}", adp))
        return variants


# ═══════════════════════════════════════════════════════════════
# OCR 引擎
# ═══════════════════════════════════════════════════════════════

def _ocr_cleanup(text):
    if not text: return text
    text = text.strip(".,;:!?。，、；：！？·… ")
    text = re.sub(r" {2,}", " ", text)
    return text

class OCREngine:
    def __init__(self, use_gpu=True, chinese_lite=False,
                 model_preset=DEFAULT_MODEL, min_text_score=0.5):
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

    @staticmethod
    def preprocess_all(crop, preprocess_mode="auto"):
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

    def recognize_preprocessed(self, variants, need_resize=False, target_scale=1.0):
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

    def recognize(self, crop, preprocess_mode="auto"):
        variants, need_resize, target_scale = self.preprocess_all(crop, preprocess_mode)
        return self.recognize_preprocessed(variants, need_resize, target_scale)


# ═══════════════════════════════════════════════════════════════
# 字幕提取器 (unchanged logic)
# ═══════════════════════════════════════════════════════════════

class ExtractionWorker:
    def __init__(self, video_path, roi, frame_interval, similarity_threshold,
                 preprocess_mode, min_duration_ms, min_text_len, use_gpu,
                 chinese_lite, num_workers=4, model_preset=DEFAULT_MODEL,
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

    def _extract(self):
        from queue import Queue
        n = self.num_workers
        x, y, rw, rh = self.roi
        interval = self.frame_interval

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 30.0
        total_scan = total_frames // interval + 1
        t_start = time.time()

        frame_queue = Queue(maxsize=n * 3)

        def reader():
            nonlocal cap
            fn = 0
            try:
                while True:
                    if self._cancel.is_set(): break
                    ret, frame = cap.read()
                    if not ret: break
                    if fn % interval == 0:
                        crop = frame[y:y+rh, x:x+rw]
                        frame_queue.put((fn, crop))
                    fn += 1
            finally:
                for _ in range(n):
                    frame_queue.put(None)

        threading.Thread(target=reader, daemon=True).start()

        hits_lock = threading.Lock()
        raw_hits = []
        scanned = [0]

        def worker_thread():
            engine = OCREngine(self.use_gpu, self.chinese_lite, self.model_preset)
            engine.init()
            while True:
                item = frame_queue.get()
                if item is None: frame_queue.task_done(); break
                fn, crop = item
                try:
                    text = engine.recognize(crop, self.preprocess_mode)
                    if text and len(text) >= self.min_text_len:
                        with hits_lock:
                            raw_hits.append((fn, text))
                finally:
                    with hits_lock: scanned[0] += 1
                    frame_queue.task_done()

        workers = [threading.Thread(target=worker_thread, daemon=True) for _ in range(n)]
        for w in workers: w.start()

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
                    'percent': pct, 'scanned': s, 'skipped': 0,
                    'total_scan': total_scan, 'hits': len(raw_hits),
                    'elapsed': elapsed, 'eta': eta,
                    'current_frame': 0, 'total_frames': total_frames, 'workers': n,
                })
                last_update = now

        for w in workers: w.join()
        cap.release()

        if self._cancel.is_set(): return []
        if not raw_hits: return []
        raw_hits.sort(key=lambda x: x[0])
        return self._post_process(self._merge_hits(raw_hits, fps), fps)

    def _merge_hits(self, hits, fps):
        if not hits: return []
        segments = []
        batch_start = batch_end = hits[0][0]
        best_text = hits[0][1]
        gap_threshold = max(3, int(fps * 0.8))
        for i in range(1, len(hits)):
            fn, text = hits[i]
            gap = fn - hits[i-1][0]
            sim = text_similarity_fn(best_text, text)
            if sim >= self.similarity_threshold:
                batch_end = fn
                if len(text) > len(best_text): best_text = text
            elif gap <= gap_threshold and sim >= 0.5:
                batch_end = fn
                if len(text) > len(best_text): best_text = text
            else:
                segments.append((batch_start, batch_end, best_text))
                batch_start = batch_end = fn
                best_text = text
        segments.append((batch_start, batch_end, best_text))
        return segments

    def _post_process(self, segments, fps):
        if not segments: return []
        min_frames = int(self.min_duration_ms / 1000 * fps)
        filtered = [(s, e, t) for s, e, t in segments
                    if e - s >= min_frames and t.strip()]
        merged = []
        for sub in filtered:
            if not merged: merged.append(sub); continue
            ps, pe, pt = merged[-1]
            gap_ms = (sub[0] - pe) / fps * 1000
            sim = text_similarity_fn(pt, sub[2])
            if sim >= 0.9 and gap_ms < 1500:
                merged[-1] = (ps, sub[1], sub[2] if len(sub[2]) > len(pt) else pt)
            else:
                merged.append(sub)
        final = []
        for i, sub in enumerate(merged):
            if i == 0: final.append(sub); continue
            ps, pe, pt = final[-1]
            gap_ms = (sub[0] - pe) / fps * 1000
            if gap_ms < 300 and text_similarity_fn(pt, sub[2]) >= 0.85:
                final[-1] = (ps, sub[1], sub[2] if len(sub[2]) > len(pt) else pt)
            else:
                final.append(sub)
        return final


# ═══════════════════════════════════════════════════════════════
# ROI Canvas — 支持字幕叠加
# ═══════════════════════════════════════════════════════════════

class ROICanvas(tk.Canvas):
    def __init__(self, parent, roi_change_callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.roi = None
        self._frame_orig = None
        self._photo = None
        self._photo_id = None
        self._rect_id = None
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._drag_start = None
        self._drag_rect = None
        self._roi_change_cb = roi_change_callback
        # 字幕叠加
        self._subtitles = []
        self._current_frame_fn = 0

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Button-3>", self._clear_roi)
        self.bind("<Configure>", self._on_resize)

    def set_subtitles(self, subtitles):
        """设置字幕列表 [(start_frame, end_frame, text), ...]"""
        self._subtitles = subtitles

    def set_current_frame(self, fn):
        self._current_frame_fn = fn

    def _get_current_subtitle(self):
        for s, e, text in self._subtitles:
            if s <= self._current_frame_fn <= e:
                return text
        return ""

    def set_frame(self, frame):
        self._frame_orig = frame.copy()
        self._render()

    def _render(self):
        if self._frame_orig is None: return
        cw = self.winfo_width() or 640
        ch = self.winfo_height() or 360
        fh, fw = self._frame_orig.shape[:2]
        self._scale = min(cw / fw, ch / fh)
        new_w = int(fw * self._scale)
        new_h = int(fh * self._scale)
        self._offset_x = (cw - new_w) // 2
        self._offset_y = (ch - new_h) // 2

        resized = cv2.resize(self._frame_orig, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # 叠加字幕
        sub_text = self._get_current_subtitle()
        if sub_text:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("msyh.ttc", max(16, int(new_h * 0.045)))
            except Exception:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), sub_text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = max(0, (new_w - tw) // 2)
            ty = max(0, int(new_h * 0.82))
            # 半透明黑底
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            odraw = ImageDraw.Draw(overlay)
            odraw.rectangle([tx - 10, ty - 4, tx + tw + 10, ty + th + 4],
                            fill=(0, 0, 0, 180))
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(img)
            draw.text((tx, ty), sub_text, fill=(255, 255, 255), font=font)

        self._photo = ImageTk.PhotoImage(img)
        if self._photo_id:
            self.delete(self._photo_id)
        self._photo_id = self.create_image(self._offset_x, self._offset_y,
                                           anchor='nw', image=self._photo)
        self._draw_roi()

    def _on_resize(self, event):
        if self._frame_orig is not None:
            self._render()

    def _canvas_to_frame(self, cx, cy):
        fx = (cx - self._offset_x) / self._scale
        fy = (cy - self._offset_y) / self._scale
        return int(fx), int(fy)

    def _on_press(self, event):
        self._clear_roi()
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start is None: return
        sx, sy = self._drag_start
        if self._drag_rect: self.delete(self._drag_rect)
        self._drag_rect = self.create_rectangle(sx, sy, event.x, event.y,
                                                 outline='#00ff00', width=2, dash=(5,3))

    def _on_release(self, event):
        if self._drag_start is None: return
        sx, sy = self._drag_start
        ex, ey = event.x, event.y
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

    def _notify_roi_change(self):
        if self._roi_change_cb:
            self._roi_change_cb(self.get_frame_roi())

    def _draw_roi(self):
        if self._rect_id:
            self.delete(self._rect_id); self._rect_id = None
        if self.roi is None: return
        x1, y1, w, h = self.roi
        self._rect_id = self.create_rectangle(x1, y1, x1+w, y1+h,
                                              outline='#00ff00', width=2, dash=(5,3))
        self.create_text(x1+4, y1-10 if y1>15 else y1+h+15,
                         text=f"{int(w/self._scale)}×{int(h/self._scale)}",
                         anchor='w', fill='#00ff00', font=('', 9))

    def _clear_roi(self, event=None):
        self.roi = None
        if self._rect_id:
            self.delete(self._rect_id); self._rect_id = None
        self._notify_roi_change()

    def get_frame_roi(self):
        if self.roi is None or self._frame_orig is None: return None
        cx1, cy1, cw, ch = self.roi
        fx, fy = self._canvas_to_frame(cx1, cy1)
        fw, fh = int(cw/self._scale), int(ch/self._scale)
        fh_img, fw_img = self._frame_orig.shape[:2]
        fx, fy = max(0, fx), max(0, fy)
        fw, fh = min(fw, fw_img-fx), min(fh, fh_img-fy)
        if fw < 2 or fh < 2: return None
        return (fx, fy, fw, fh)


# ═══════════════════════════════════════════════════════════════
# 主应用
# ═══════════════════════════════════════════════════════════════

class SubtitleOCRApp:
    WINDOW_TITLE = "视频字幕OCR提取工具"
    MIN_WIDTH, MIN_HEIGHT = 1100, 700
    DEFAULT_WIDTH, DEFAULT_HEIGHT = 1200, 780

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
        self.frame_position = tk.IntVar(value=0)
        self.play_speed = tk.DoubleVar(value=1.0)

        self.cap = None
        self.fps = 0.0
        self.total_frames = 0
        self.video_resolution = ""
        self.subtitles = []
        self.worker = None
        self._editing = False
        self._progress_queue = queue.Queue()

        # 播放器状态 (VLC 回调统一驱动音视频，Canvas 字幕叠加)
        self._playing = False
        self._player_gen = 0    # 代际计数器，防止旧回调继续
        self._vlc = None
        self._vlc_fb = None
        self._current_frame = 0
        self._frame_lock = threading.Lock()

        self._build_ui()
        self._update_gpu_status()
        self._poll_progress()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.statusbar.configure(text="正在预热 OCR 引擎...")
        GpuManager.instance().warmup_async(on_ready=self._on_engine_ready)

    # ── UI ─────────────────────────────────────────────

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill='both', expand=True)
        self._build_toolbar(main)
        body = ttk.Frame(main)
        body.pack(fill='both', expand=True, padx=8, pady=4)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        self._build_preview_panel(body)
        self._build_control_panel(body)
        self._build_statusbar(main)

    def _build_toolbar(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill='x', padx=8, pady=(6, 0))
        self.gpu_label = ttk.Label(bar, text="", font=('', 10, 'bold'))
        self.gpu_label.pack(side='left', padx=(0, 20))
        ttk.Label(bar, text="视频字幕OCR提取工具",
                  font=('Microsoft YaHei UI', 13, 'bold')).pack(side='left')
        ttk.Label(bar, text="RapidOCR + ONNX Runtime",
                  foreground='#888').pack(side='left', padx=(10, 0))

    def _build_preview_panel(self, parent):
        frame = ttk.LabelFrame(parent,
                               text=" 视频播放器 - 拖拽框选字幕区域 | 右键清除 | 空格键播放/暂停 ",
                               padding=4)
        frame.grid(row=0, column=0, sticky='nsew', padx=(0, 4))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas = ROICanvas(frame,
                                roi_change_callback=self._on_roi_changed,
                                bg='#1a1a2e', highlightthickness=1,
                                highlightbackground='#444')
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.create_text(320, 180, text="请选择视频文件",
                                fill='#666', font=('Microsoft YaHei UI', 14),
                                tags=('placeholder',))

        # ── 播放控制 ──
        ctrl = ttk.Frame(frame)
        ctrl.grid(row=1, column=0, sticky='ew', pady=(6, 0))
        ctrl.columnconfigure(1, weight=1)

        self.btn_play = ttk.Button(ctrl, text="▶ 播放", command=self._toggle_play,
                                   state='disabled', width=8)
        self.btn_play.grid(row=0, column=0, padx=(0, 4))

        self.frame_slider = ttk.Scale(ctrl, from_=0, to=100, orient='horizontal',
                                      variable=self.frame_position,
                                      command=self._on_frame_slider, state='disabled')
        self.frame_slider.grid(row=0, column=1, sticky='ew', padx=(4, 4))

        ttk.Label(ctrl, text="倍速:").grid(row=0, column=2, padx=(4, 2))
        self.speed_cb = ttk.Combobox(ctrl, textvariable=self.play_speed,
                          values=[0.5, 1.0, 1.5, 2.0], state='readonly', width=4)
        self.speed_cb.current(1)
        self.speed_cb.grid(row=0, column=3, padx=(0, 4))
        self.speed_cb.bind('<<ComboboxSelected>>', self._on_speed_change)

        self.frame_info = ttk.Label(ctrl, text="", font=('', 8),
                                    foreground='#888', width=30, anchor='e')
        self.frame_info.grid(row=0, column=4, padx=(4, 0))

        # 帧导航
        nav = ttk.Frame(frame)
        nav.grid(row=2, column=0, sticky='w', pady=(2, 0))
        ttk.Button(nav, text="◀◀ 前10帧", command=self._prev_10_frames, width=9).pack(side='left', padx=(0,2))
        ttk.Button(nav, text="◀ 前1帧", command=self._prev_frame, width=7).pack(side='left', padx=(0,2))
        ttk.Button(nav, text="▶ 后1帧", command=self._next_frame, width=7).pack(side='left', padx=(0,2))
        ttk.Button(nav, text="▶▶ 后10帧", command=self._next_10_frames, width=9).pack(side='left')

        # 快捷键
        self.root.bind('<Left>', lambda e: self._prev_frame())
        self.root.bind('<Right>', lambda e: self._next_frame())
        self.root.bind('<space>', lambda e: self._toggle_play())

    def _build_control_panel(self, parent):
        panel = ttk.Frame(parent)
        panel.grid(row=0, column=1, sticky='nsew')
        panel.columnconfigure(0, weight=1)

        # 视频文件
        f = ttk.LabelFrame(panel, text=" 视频文件 ", padding=6)
        f.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        f.columnconfigure(1, weight=1)
        ttk.Button(f, text="选择视频...", command=self._select_video,
                   width=10).grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(f, textvariable=self.video_path, state='readonly',
                  font=('', 8)).grid(row=0, column=1, sticky='ew')
        self.video_info_label = ttk.Label(f, text="尚未选择视频", foreground='#888')
        self.video_info_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=(4,0))
        self.btn_auto_detect = ttk.Button(f, text="🔍 自动框选", command=self._auto_detect_roi,
                                          state='disabled', width=10)
        self.btn_auto_detect.grid(row=2, column=0, sticky='w', pady=(4,0))
        self.btn_transcode = ttk.Button(f, text="🔄 转码 H.264", command=self._transcode_to_h264,
                                         state='disabled', width=12)
        self.btn_transcode.grid(row=2, column=1, sticky='e', pady=(4,0))

        # 参数
        f = ttk.LabelFrame(panel, text=" 识别参数 ", padding=6)
        f.grid(row=1, column=0, sticky='ew', pady=(0, 6))
        f.columnconfigure(1, weight=1)
        params = [
            ("帧扫描间隔", self.frame_interval, 1, 60),
            ("相似度阈值", self.similarity_threshold, 0.50, 1.00),
            ("最短持续(ms)", self.min_duration, 100, 5000),
            ("最短文本(字)", self.min_text_len, 1, 10),
        ]
        for i, (label, var, vmin, vmax) in enumerate(params):
            ttk.Label(f, text=label+":", font=('',9)).grid(
                row=i, column=0, sticky='w', padx=(0,6), pady=2)
            sb = ttk.Spinbox(f, textvariable=var, from_=vmin, to=vmax, width=8)
            if isinstance(var.get(), float):
                sb.configure(format="%.2f", increment=0.05)
            sb.grid(row=i, column=1, sticky='w', pady=2)

        r = len(params)
        ttk.Label(f, text="预处理:", font=('',9)).grid(row=r, column=0, sticky='w', padx=(0,6), pady=2)
        cb = ttk.Combobox(f, textvariable=self.preprocess_mode,
                          values=['auto','binary','adaptive','enhanced','original'],
                          state='readonly', width=10)
        cb.grid(row=r, column=1, sticky='w', pady=2)
        r += 1
        ttk.Checkbutton(f, text="GPU 加速", variable=self.use_gpu).grid(
            row=r, column=0, columnspan=2, sticky='w', pady=(4,0))
        r += 1
        self.chinese_lite = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="⚡ 中文加速 (关闭方向分类)",
                        variable=self.chinese_lite).grid(
            row=r, column=0, columnspan=2, sticky='w', pady=(2,0))
        r += 1
        mf = ttk.Frame(f)
        mf.grid(row=r, column=0, columnspan=2, sticky='ew', pady=(4,0))
        ttk.Label(mf, text="🧠 OCR 模型:").pack(side='left')
        self.model_preset = tk.StringVar(value=DEFAULT_MODEL)
        model_names = list(MODEL_PRESETS.keys())
        model_labels = [MODEL_PRESETS[k]['label'] for k in model_names]
        cb_m = ttk.Combobox(mf, textvariable=self.model_preset,
                            values=model_labels, state='readonly', width=26)
        cb_m.current(0); cb_m.pack(side='left', padx=(6,0))
        self._model_label_map = dict(zip(model_labels, model_names))
        r += 1
        wf = ttk.Frame(f)
        wf.grid(row=r, column=0, columnspan=2, sticky='ew', pady=(4,0))
        ttk.Label(wf, text="🚀 并行线程数:").pack(side='left')
        ttk.Spinbox(wf, from_=1, to=8, textvariable=self.num_workers, width=3).pack(side='left', padx=(6,0))
        ttk.Label(wf , text="(4=推荐)").pack(side='left', padx=(4,0))

        # 操作按钮
        f = ttk.Frame(panel)
        f.grid(row=2, column=0, sticky='ew', pady=(0,6))
        f.columnconfigure(0, weight=1); f.columnconfigure(1, weight=1)
        self.btn_start = ttk.Button(f, text="▶ 开始提取", command=self._start_extraction)
        self.btn_start.grid(row=0, column=0, sticky='ew', padx=(0,3))
        self.btn_cancel = ttk.Button(f, text="⏹ 取消", command=self._cancel_extraction, state='disabled')
        self.btn_cancel.grid(row=0, column=1, sticky='ew', padx=(3,0))

        # 进度
        f = ttk.LabelFrame(panel, text=" 提取进度 ", padding=6)
        f.grid(row=3, column=0, sticky='ew', pady=(0,6))
        f.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(f, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky='ew')
        self.progress_text = ttk.Label(f, text="就绪", font=('',8))
        self.progress_text.grid(row=1, column=0, sticky='w', pady=(2,0))

        # 结果（可编辑）
        f = ttk.LabelFrame(panel, text=" 识别结果 (可编辑) ", padding=4)
        f.grid(row=4, column=0, sticky='nsew')
        f.rowconfigure(1, weight=1); f.columnconfigure(0, weight=1)
        panel.rowconfigure(4, weight=1)

        text_frame = ttk.Frame(f)
        text_frame.grid(row=1, column=0, sticky='nsew', pady=(4,0))
        text_frame.rowconfigure(0, weight=1); text_frame.columnconfigure(0, weight=1)
        self.result_text = tk.Text(text_frame, wrap='word', state='disabled',
                                   font=('Consolas', 9), height=8,
                                   bg='#1e1e2e', fg='#cdd6f4',
                                   insertbackground='white', relief='flat', borderwidth=0)
        self.result_text.grid(row=0, column=0, sticky='nsew')
        sb = ttk.Scrollbar(text_frame, orient='vertical', command=self.result_text.yview)
        sb.grid(row=0, column=1, sticky='ns')
        self.result_text.configure(yscrollcommand=sb.set)

        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=0, column=0, sticky='ew')
        self.btn_save = ttk.Button(btn_frame, text="💾 保存 SRT",
                                   command=self._save_srt, state='disabled')
        self.btn_save.pack(side='left', padx=(0,6))
        self.btn_copy = ttk.Button(btn_frame, text="📋 复制全部",
                                   command=self._copy_all, state='disabled')
        self.btn_copy.pack(side='left', padx=(0,6))
        self.btn_edit = ttk.Button(btn_frame, text="✏️ 编辑", command=self._toggle_edit,
                                   state='disabled')
        self.btn_edit.pack(side='left')

    def _build_statusbar(self, parent):
        self.statusbar = ttk.Label(parent, text="就绪", relief='sunken',
                                   anchor='w', padding=(8,2))
        self.statusbar.pack(fill='x', padx=8, pady=(0,6))

    # ── GPU ─────────────────────────────────────────────

    def _on_engine_ready(self):
        self.root.after(0, lambda: self.statusbar.configure(text="就绪 — 请选择视频文件"))

    def _update_gpu_status(self):
        gm = GpuManager.instance()
        self.gpu_label.configure(text=gm.status_text(), foreground=gm.status_color())
        if not gm.gpu_available:
            self.use_gpu.set(False)

    def _on_roi_changed(self, roi):
        self._update_button_states()

    # ── 视频选择 ───────────────────────────────────────

    def _select_video(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv *.webm *.m4v"), ("所有文件","*.*")])
        if not path: return
        self.video_path.set(path)
        self._load_video_info()

    def _load_video_info(self):
        path = self.video_path.get()
        if not path or not os.path.exists(path): return
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", f"无法打开视频:\n{path}"); return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.video_resolution = f"{w}×{h}"
        self.video_info_label.configure(
            text=f"📐 {w}×{h}  |  🎬 {self.fps:.1f} FPS  |  ⏱️ {timedelta(seconds=int(duration))}  |  📦 {self.total_frames} 帧")
        self.frame_slider.configure(to=max(0, self.total_frames-1), state='normal')
        self.canvas.clear_all_rois() if hasattr(self.canvas, 'clear_all_rois') else self.canvas._clear_roi()
        self.canvas.set_subtitles([])
        self._seek_frame(max(0, int(self.total_frames * 0.1)))
        self.statusbar.configure(text=f"已加载: {os.path.basename(path)} — 空格键播放，拖拽框选字幕")
        self._clear_results()
        self._update_button_states()

    # ── 帧导航 & 播放器 ────────────────────────────────

    def _seek_frame(self, fn):
        if not self.cap or not self.cap.isOpened(): return
        fn = max(0, min(fn, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = self.cap.read()
        if not ret: return
        self.canvas.delete('placeholder')
        self.canvas.set_current_frame(fn)
        self.canvas.set_frame(frame)
        self.frame_position.set(fn)
        with self._frame_lock: self._current_frame = fn
        self.frame_info.configure(text=f"帧 {fn} / {self.total_frames}  |  {frame_to_time_str(fn, self.fps)}")

    def _on_frame_slider(self, val):
        if self._playing: return
        self._seek_frame(int(float(val)))

    def _prev_frame(self):
        was_playing = self._playing
        self._stop_player(); self._seek_frame(self.frame_position.get() - 1)
        if was_playing: self._start_player()

    def _next_frame(self):
        was_playing = self._playing
        self._stop_player(); self._seek_frame(self.frame_position.get() + 1)
        if was_playing: self._start_player()

    def _prev_10_frames(self):
        was_playing = self._playing
        self._stop_player(); self._seek_frame(self.frame_position.get() - 10)
        if was_playing: self._start_player()

    def _next_10_frames(self):
        was_playing = self._playing
        self._stop_player(); self._seek_frame(self.frame_position.get() + 10)
        if was_playing: self._start_player()

    def _toggle_play(self):
        if self._playing: self._stop_player()
        else: self._start_player()

    def _start_player(self):
        if not self.cap or not self.cap.isOpened(): return
        self._playing = True
        self._player_gen += 1  # 换代，旧轮询自动退出
        self.btn_play.configure(text="⏸ 暂停")
        # 获取视频原始尺寸，供 VLC 回调分配缓冲区
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._vlc_fb = VlcFrameBuffer(width, height)
        # VLC 音视频统一驱动
        try:
            inst = vlc.Instance()  # 默认 vout 会驱动回调；'Failed to set on top' 是无害警告
            self._vlc = inst.media_player_new()
            media = inst.media_new(self.video_path.get())
            self._vlc.set_media(media)
            self._vlc.video_set_callbacks(
                self._vlc_fb.lock, self._vlc_fb.unlock, self._vlc_fb.display, None)
            self._vlc.video_set_format("RV32", width, height, width * 4)
            self._vlc.play()
            # 隐藏 VLC 默认弹出的独立视频窗口（避免双画面）
            self.root.after(150, _hide_vlc_windows)
            self.root.after(800, _hide_vlc_windows)
            while self._vlc.get_state() == vlc.State.Opening:
                time.sleep(0.02)
            target_ms = int(self.frame_position.get() / max(self.fps, 1) * 1000)
            self._vlc.set_time(target_ms)
            try: self._vlc.set_rate(self.play_speed.get())
            except Exception: pass
        except Exception as e:
            self.statusbar.configure(text=f"⚠️ 播放器初始化失败: {e}")
            return
        gen = self._player_gen
        self.root.after(20, lambda: self._poll_vlc_frame(gen))

    def _poll_vlc_frame(self, gen):
        if not self._playing or self._player_gen != gen or not self._vlc_fb:
            return
        if self._vlc and self._vlc.get_state() == vlc.State.Ended:
            self._stop_player()
            return
        try:
            if not self._vlc_fb.queue.empty():
                rgba = self._vlc_fb.queue.get_nowait()
                bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                vlc_ms = self._vlc.get_time() if self._vlc else -1
                if vlc_ms >= 0:
                    fn = int(vlc_ms / 1000.0 * self.fps)
                    fn = max(0, min(fn, self.total_frames - 1))
                else:
                    fn = self.frame_position.get()
                # 直接渲染到 Canvas（不经过不存在的 _show_frame）
                self.canvas.delete('placeholder')
                self.canvas.set_current_frame(fn)
                self.canvas.set_frame(bgr)
                self.frame_position.set(fn)
                with self._frame_lock:
                    self._current_frame = fn
                self.frame_info.configure(
                    text=f"帧 {fn} / {self.total_frames}  |  {frame_to_time_str(fn, self.fps)}")
        except Exception as e:
            # 打印异常方便调试，不再静默吞掉
            import traceback
            traceback.print_exc()
        self.root.after(20, lambda: self._poll_vlc_frame(gen))

    def _stop_player(self):
        self._playing = False
        self._player_gen += 1  # 旧轮询自毁
        self.btn_play.configure(text="▶ 播放")
        if self._vlc:
            try: self._vlc.stop()
            except Exception: pass
            try: self._vlc.release()
            except Exception: pass
            self._vlc = None
        self._vlc_fb = None

    def _on_speed_change(self, event=None):
        if self._playing and self._vlc:
            try: self._vlc.set_rate(self.play_speed.get())
            except Exception: pass

    # ── 按钮状态 ───────────────────────────────────────

    def _update_button_states(self):
        video_loaded = bool(self.video_path.get() and self.cap and self.cap.isOpened())
        can_start = video_loaded and self.canvas.roi is not None
        self.btn_start.configure(state='normal' if can_start else 'disabled')
        self.btn_auto_detect.configure(state='normal' if video_loaded else 'disabled')
        self.btn_play.configure(state='normal' if video_loaded else 'disabled')
        if hasattr(self, 'btn_transcode'):
            self.btn_transcode.configure(state='normal' if video_loaded else 'disabled')

    # ── 自动框选 (unchanged) ───────────────────────────

    def _transcode_to_h264(self):
        """转码为 H.264+AAC，存到缓存目录，完成后自动加载。"""
        src = self.video_path.get()
        if not src or not os.path.exists(src):
            messagebox.showwarning("提示", "请先加载视频文件"); return
        cache_dir = Path(os.environ.get('TEMP', '/tmp')) / 'video-subtitle-ocr-cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        src_path = Path(src)
        dst_path = cache_dir / (src_path.stem + '_h264.mp4')
        if dst_path.exists() and dst_path.stat().st_size > 1024:
            if messagebox.askyesno("缓存已存在", "转码后的文件已存在:\n" + str(dst_path) + "\n\n直接使用?"):
                self.video_path.set(str(dst_path))
                self._load_video_info()
                self.statusbar.configure(text="已加载缓存: " + dst_path.name)
            return
        ffmpeg_exe = self._find_ffmpeg()
        if not ffmpeg_exe:
            messagebox.showerror("未找到 ffmpeg",
                "需要 ffmpeg 才能转码。\n\n安装方法:\n  winget install ffmpeg\n  或  choco install ffmpeg")
            return
        try:
            cap = cv2.VideoCapture(src)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration_sec = total / fps if fps > 0 else 0
        except Exception:
            duration_sec = 0
        size_mb = os.path.getsize(src) / 1024 / 1024
        if not messagebox.askokcancel("开始转码",
            "将转码为 H.264 + AAC 并保存到缓存:\n\n"
            "输入: " + src_path.name + " (" + f"{size_mb:.1f}" + " MB)\n"
            "输出: " + str(dst_path) + "\n\n"
            "转码可能需要数分钟，完成后会自动加载。"):
            return
        self._tcode_win = tk.Toplevel(self.root)
        self._tcode_win.title("转码中...")
        self._tcode_win.geometry("480x200")
        self._tcode_win.transient(self.root)
        self._tcode_win.grab_set()
        self._tcode_win.protocol("WM_DELETE_WINDOW", lambda: None)
        ttk.Label(self._tcode_win, text="转码中，请稍候...",
                  font=('Microsoft YaHei UI', 11, 'bold')).pack(pady=(15, 5))
        self._tcode_label = ttk.Label(self._tcode_win, text="准备中...",
                                      font=('', 9), foreground='#888')
        self._tcode_label.pack(pady=(0, 5))
        self._tcode_bar = ttk.Progressbar(self._tcode_win, mode='determinate', maximum=100)
        self._tcode_bar.pack(padx=20, fill='x', pady=5)
        self._tcode_status = ttk.Label(self._tcode_win, text="",
                                       font=('', 8), foreground='#666')
        self._tcode_status.pack(pady=(5, 10))
        self._tcode_cancel = ttk.Button(self._tcode_win, text="取消", command=self._cancel_transcode)
        self._tcode_cancel.pack()
        self._tcode_proc = None
        self._tcode_cancel_flag = threading.Event()
        last_us_holder = [0]
        def do_transcode():
            cmd = [ffmpeg_exe, '-y', '-i', src,
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
                   '-c:a', 'aac', '-b:a', '128k',
                   '-progress', 'pipe:1', '-nostats',
                   str(dst_path)]
            try:
                self._tcode_proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except FileNotFoundError as e:
                self.root.after(0, lambda: self._on_transcode_done(None, "无法启动 ffmpeg: " + str(e)))
                return
            for line in self._tcode_proc.stdout:
                line = line.decode('utf-8', errors='replace').strip()
                if self._tcode_cancel_flag.is_set():
                    self._tcode_proc.kill()
                    self.root.after(0, lambda: self._on_transcode_done(None, "已取消"))
                    return
                if line.startswith('out_time_ms='):
                    try:
                        us = int(line.split('=', 1)[1])
                        last_us_holder[0] = us
                        if duration_sec > 0:
                            pct = min(us / 1_000_000 / duration_sec * 100, 100)
                            elapsed = us / 1_000_000
                            self.root.after(0, lambda p=pct, e=elapsed: self._update_tcode_progress(
                                p, e, duration_sec))
                    except (ValueError, IndexError): pass
                elif line == 'progress=end':
                    self.root.after(0, lambda: self._update_tcode_progress(100, duration_sec, duration_sec))
            self._tcode_proc.wait()
            rc = self._tcode_proc.returncode
            if rc == 0 and dst_path.exists() and dst_path.stat().st_size > 1024:
                self.root.after(0, lambda: self._on_transcode_done(str(dst_path), None))
            else:
                err_bytes = self._tcode_proc.stderr.read()
                err = err_bytes.decode('utf-8', errors='replace')
                err_tail = err[-500:] if err else ("ffmpeg 退出码 " + str(rc))
                self.root.after(0, lambda: self._on_transcode_done(None, err_tail))
        threading.Thread(target=do_transcode, daemon=True).start()

    def _find_ffmpeg(self):
        import shutil
        p = shutil.which('ffmpeg')
        if p: return p
        candidates = [
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\ffmpeg\bin\ffmpeg.exe',
            os.path.expandvars(r'%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe'),
        ]
        for c in candidates:
            if os.path.exists(c): return c
        return None

    def _update_tcode_progress(self, pct, elapsed, total):
        try:
            if not self._tcode_bar.winfo_exists(): return
        except Exception: return
        self._tcode_bar['value'] = pct
        self._tcode_label.configure(text=f"{pct:.1f}%   {elapsed:.1f}s / {total:.1f}s")
        self._tcode_status.configure(text=f"输出: {pct:.1f}% 完成")

    def _cancel_transcode(self):
        self._tcode_cancel_flag.set()
        if hasattr(self, '_tcode_proc') and self._tcode_proc:
            try: self._tcode_proc.kill()
            except Exception: pass

    def _on_transcode_done(self, dst_path, error):
        try:
            if hasattr(self, '_tcode_win'):
                self._tcode_win.destroy()
        except Exception: pass
        if error:
            messagebox.showerror("转码失败", "转码失败:\n" + str(error)[:1000])
            return
        self.video_path.set(dst_path)
        self._load_video_info()
        self.statusbar.configure(text="已加载转码后视频: " + os.path.basename(dst_path))
        messagebox.showinfo("转码完成", "转码完成！已自动加载:\n" + os.path.basename(dst_path) +
                            "\n\n缓存位置:\n" + dst_path)

    def _auto_detect_roi(self):
        path = self.video_path.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("提示", "请先选择视频文件"); return
        if not self.cap or not self.cap.isOpened(): return
        ok = messagebox.askokcancel("自动框选",
            "将自动扫描视频中的文字区域并框选字幕位置。\n\n这可能需要 10-30 秒，是否继续？")
        if not ok: return
        self.statusbar.configure(text="🔍 正在自动检测字幕区域...")
        self.btn_auto_detect.configure(state='disabled')
        self.btn_start.configure(state='disabled')
        self.root.update_idletasks()

        def do_auto_detect():
            try:
                from rapidocr_onnxruntime import RapidOCR
                fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                positions = np.linspace(0.15, 0.85, 6)
                frame_indices = [int(total_frames * p) for p in positions]
                detector = RapidOCR(box_thresh=0.2, text_score=0.1,
                                    use_text_det=True, use_angle_cls=False)
                all_boxes = []
                for fn in frame_indices:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                    ret, frame = self.cap.read()
                    if not ret: continue
                    dt_boxes, _ = detector.text_detector(frame)
                    all_boxes.append(dt_boxes)
                if not all_boxes:
                    self._auto_detect_queue.put(('error', '未检测到任何文字区域，请手动框选')); return
                min_w = fw * 0.15; max_h = fh * 0.15; bottom_start = fh * 0.55
                candidate_boxes = []
                for dt_boxes in all_boxes:
                    for box in dt_boxes:
                        xs = box[:,0]; ys = box[:,1]
                        bx, by = xs.min(), ys.min()
                        bw, bh = xs.max()-bx, ys.max()-by
                        if by >= bottom_start and bw >= min_w and bh <= max_h and bw/max(bh,1) >= 2.0:
                            candidate_boxes.append((bx, by, bw, bh))
                if not candidate_boxes:
                    self._auto_detect_queue.put(('error', '未在底部检测到字幕文字，请手动框选')); return
                margin = 8
                xs = [b[0] for b in candidate_boxes]; ys = [b[1] for b in candidate_boxes]
                xs2 = [b[0]+b[2] for b in candidate_boxes]; ys2 = [b[1]+b[3] for b in candidate_boxes]
                x = max(0, int(np.min(xs))-margin); y = max(0, int(np.min(ys))-margin)
                w = min(fw-x, int(np.max(xs2))-x+margin); h = min(fh-y, int(np.max(ys2))-y+margin)
                self._auto_detect_queue.put(('ok', (x,y,w,h), len(candidate_boxes), len(all_boxes)))
            except Exception as e:
                self._auto_detect_queue.put(('error', str(e)))

        self._auto_detect_queue = queue.Queue()
        threading.Thread(target=do_auto_detect, daemon=True).start()
        self._poll_auto_detect()

    def _poll_auto_detect(self):
        try:
            result = self._auto_detect_queue.get_nowait()
        except queue.Empty:
            self.root.after(200, self._poll_auto_detect); return
        status, *data = result
        if status == 'ok':
            roi, n_c, n_f = data
            x, y, w, h = roi
            self.canvas.roi = (x, y, w, h)
            self.canvas._draw_roi()
            self.statusbar.configure(
                text=f"✅ 自动框选完成: x={x} y={y} w={w} h={h}  |  {n_c} 个候选框, 采样 {n_f} 帧")
        else:
            msg = data[0] if data else '未知错误'
            messagebox.showwarning("自动框选", msg)
            self.statusbar.configure(text=f"⚠️ 自动框选失败: {msg}")
        self.btn_auto_detect.configure(state='normal')
        self._update_button_states()

    # ── 提取控制 ───────────────────────────────────────

    def _start_extraction(self):
        roi = self.canvas.get_frame_roi()
        if roi is None:
            messagebox.showwarning("提示", "请先在视频预览中框选字幕区域"); return
        path = self.video_path.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("提示", "请先选择视频文件"); return
        self.subtitles = []
        self.canvas.set_subtitles([])
        self._clear_results()
        self.btn_start.configure(state='disabled')
        self.btn_auto_detect.configure(state='disabled')
        self.btn_cancel.configure(state='normal')
        self.btn_save.configure(state='disabled')
        self.btn_copy.configure(state='disabled')
        self.btn_edit.configure(state='disabled')
        self.progress_bar['value'] = 0
        self.progress_text.configure(text="正在扫描视频帧...")
        self.statusbar.configure(text="正在提取字幕...")
        self.worker = ExtractionWorker(
            video_path=path, roi=roi,
            frame_interval=self.frame_interval.get(),
            similarity_threshold=self.similarity_threshold.get(),
            preprocess_mode=self.preprocess_mode.get(),
            min_duration_ms=self.min_duration.get(),
            min_text_len=self.min_text_len.get(),
            use_gpu=self.use_gpu.get(),
            chinese_lite=self.chinese_lite.get(),
            model_preset=self._model_label_map.get(self.model_preset.get(), DEFAULT_MODEL),
            progress_callback=lambda d: self._progress_queue.put(('progress', d)),
            done_callback=lambda s, e: self._progress_queue.put(('done', (s, e))),
            num_workers=self.num_workers.get())
        threading.Thread(target=self.worker.run, daemon=True).start()

    def _cancel_extraction(self):
        if self.worker: self.worker.cancel()
        self.statusbar.configure(text="已取消")
        self._reset_ui_state()
        self.progress_text.configure(text="已取消")

    def _poll_progress(self):
        try:
            while True:
                msg_type, data = self._progress_queue.get_nowait()
                if msg_type == 'progress': self._update_progress_ui(data)
                elif msg_type == 'done': self._on_extraction_done(data[0], data[1])
        except queue.Empty: pass
        self.root.after(100, self._poll_progress)

    def _update_progress_ui(self, data):
        self.progress_bar['value'] = data['percent']
        eta_str = str(timedelta(seconds=int(data['eta']))) if data['eta'] < 86400 else "..."
        self.progress_text.configure(
            text=f"已扫: {data['scanned']}/{data['total_scan']}帧 | "
                 f"命中: {data['hits']}条 | 已耗时: {timedelta(seconds=int(data['elapsed']))} | 剩余: {eta_str}")

    def _on_extraction_done(self, subtitles, error):
        self._reset_ui_state()
        if error:
            messagebox.showerror("提取失败", f"发生错误:\n{error}")
            self.statusbar.configure(text=f"错误: {error}"); return
        self.subtitles = subtitles
        self.canvas.set_subtitles(subtitles)
        if not subtitles:
            self.statusbar.configure(text="未检测到字幕文本")
            self._show_no_result(); return
        self._show_results(subtitles)
        self.statusbar.configure(text=f"✅ 提取完成 | 共 {len(subtitles)} 条字幕 | 播放可预览字幕叠加")

    def _reset_ui_state(self):
        self.btn_start.configure(state='normal')
        self.btn_cancel.configure(state='disabled')
        self._update_button_states()

    # ── 结果显示 & 编辑 ────────────────────────────────

    def _show_results(self, subtitles):
        self.result_text.configure(state='normal')
        self.result_text.delete('1.0', 'end')
        for i, (s, e, text) in enumerate(subtitles, 1):
            self.result_text.insert('end', f"{i:4d}  {frame_to_time_str(s, self.fps)} → {frame_to_time_str(e, self.fps)}\n", 'time')
            self.result_text.insert('end', f"      {text}\n\n", 'text')
        self.result_text.tag_configure('time', foreground='#89b4fa', font=('Consolas', 9, 'bold'))
        self.result_text.tag_configure('text', foreground='#cdd6f4', font=('Microsoft YaHei UI', 10))
        self.result_text.configure(state='disabled')
        self.result_text.see('1.0')
        self.btn_save.configure(state='normal')
        self.btn_copy.configure(state='normal')
        self.btn_edit.configure(state='normal', text="✏️ 编辑")
        self._editing = False

    def _toggle_edit(self):
        if self._editing:
            edited = self._parse_edited_text()
            if edited is not None:
                self.subtitles = edited
                self.canvas.set_subtitles(edited)
                self._show_results(edited)
                self.statusbar.configure(text=f"✅ 已保存编辑 | 共 {len(edited)} 条字幕")
            else:
                self._show_results(self.subtitles)
                messagebox.showwarning("解析错误", "无法解析编辑内容，已恢复原字幕")
            self._editing = False
        else:
            self.result_text.configure(state='normal')
            self.result_text.focus_set()
            self._editing = True
            self.btn_edit.configure(text="✅ 完成编辑")
            self.btn_save.configure(state='disabled')
            self.statusbar.configure(text="✏️ 编辑模式 — 直接修改文字后点击「完成编辑」")

    def _parse_edited_text(self):
        content = self.result_text.get('1.0', 'end').strip()
        if not content: return []
        lines = content.split('\n')
        subtitles = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line: i += 1; continue
            m = re.match(r'^\s*\d*\s*(\d{2}:\d{2}:\d{2},\d{3})\s*[→\-–>]+\s*(\d{2}:\d{2}:\d{2},\d{3})\s*$', line)
            if m:
                t1_str, t2_str = m.group(1), m.group(2)
                s = self._time_to_frame(t1_str)
                e = self._time_to_frame(t2_str)
                i += 1
                text = ""
                while i < len(lines):
                    nl = lines[i]
                    if re.match(r'^\s*\d*\s*\d{2}:\d{2}:\d{2},\d{3}\s*[→\-–>]', nl): break
                    stripped = nl.strip()
                    if stripped: text = stripped; i += 1; break
                    i += 1
                if text and s >= 0:
                    subtitles.append((s, e, text))
            else: i += 1
        return subtitles

    def _time_to_frame(self, ts):
        try:
            h, m, rest = ts.split(':')
            s, ms = rest.split(',')
            secs = int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0
            return int(secs * self.fps)
        except ValueError: return -1

    def _show_no_result(self):
        self.result_text.configure(state='normal'); self.result_text.delete('1.0', 'end')
        self.result_text.insert('end',
            "⚠️ 未检测到字幕文本\n\n建议:\n  1. 重新框选\n  2. 减小帧扫描间隔\n  3. 切换预处理模式\n", 'info')
        self.result_text.tag_configure('info', foreground='#6c7086', font=('',9,'italic'))
        self.result_text.configure(state='disabled')

    def _clear_results(self):
        self.result_text.configure(state='normal'); self.result_text.delete('1.0', 'end')
        self.result_text.configure(state='disabled')
        self.btn_save.configure(state='disabled'); self.btn_copy.configure(state='disabled')
        self.btn_edit.configure(state='disabled')
        self._editing = False

    # ── 导出 ───────────────────────────────────────────

    def _save_srt(self):
        if self._editing:
            edited = self._parse_edited_text()
            if edited is not None:
                self.subtitles = edited; self.canvas.set_subtitles(edited)
            else:
                messagebox.showwarning("提示", "编辑内容有误，请先完成编辑"); return
        if not self.subtitles: return
        default_name = Path(self.video_path.get()).stem + ".srt"
        path = filedialog.asksaveasfilename(title="保存 SRT 字幕", defaultextension=".srt",
                                            initialfile=default_name,
                                            filetypes=[("SRT字幕","*.srt"), ("所有文件","*.*")])
        if not path: return
        with open(path, 'w', encoding='utf-8') as f:
            for i, (s, e, text) in enumerate(self.subtitles, 1):
                f.write(f"{i}\n{frame_to_time_str(s, self.fps)} --> {frame_to_time_str(e, self.fps)}\n{text}\n\n")
        self.statusbar.configure(text=f"💾 已保存: {path} ({len(self.subtitles)} 条)")
        messagebox.showinfo("保存成功", f"字幕已保存到:\n{path}\n共 {len(self.subtitles)} 条")

    def _copy_all(self):
        if not self.subtitles: return
        lines = []
        for i, (s, e, text) in enumerate(self.subtitles, 1):
            lines.append(f"{i}\n{frame_to_time_str(s, self.fps)} --> {frame_to_time_str(e, self.fps)}\n{text}\n")
        self.root.clipboard_clear(); self.root.clipboard_append("".join(lines))
        self.statusbar.configure(text="📋 已复制全部字幕到剪贴板")

    # ── 生命周期 ───────────────────────────────────────

    def _on_close(self):
        self._stop_player()
        if self.worker: self.worker.cancel()
        if self.cap: self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════

def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception: pass
    SubtitleOCRApp().run()

if __name__ == "__main__":
    main()
