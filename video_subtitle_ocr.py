#!/usr/bin/env python3
"""
Video Subtitle OCR Extractor (视频字幕OCR提取工具)
==================================================
功能: 框选视频字幕区域 → 逐帧OCR识别 → 输出SRT字幕文件

使用:
  python video_subtitle_ocr.py video.mp4
  python video_subtitle_ocr.py video.mp4 -o output.srt -i 5
  python video_subtitle_ocr.py video.mp4 --threshold 0.9

依赖: opencv-python, rapidocr-onnxruntime, onnxruntime-gpu, numpy, pillow
"""

import sys
import os

# ═══ 启动时自动注入 NVIDIA CUDA DLL（必须在 import onnxruntime 之前）═══
def _setup_cuda_path():
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

import cv2
import numpy as np
import os
import sys
import re
import argparse
import time
import json
from pathlib import Path
from difflib import SequenceMatcher
from datetime import timedelta
from collections import defaultdict


# ── 工具函数 ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """清洗文本：去空格、标点，用于相似度比较"""
    # 构建字符类：空格 + 中英文标点 + 特殊符号
    chars = (
        r"\s"                          # 空白字符
        r",，。、．.!！?？;；:："           # 基本标点
        r"\u201c\u201d\u2018\u2019"       # 弯引号
        r"\u300c\u300d\u300e\u300f"       # 「」『』
        r"\u3010\u3011"                   # 【】
        r"\uff08\uff09"                   # （）
        r"()"                             # 半角括号
        r"\u2026\u2014\u2013"             # … — –
        r"_/\\|@#$%^&*+=~`"              # 其他符号
    )
    return re.sub(f"[{chars}]", '', text)


def text_similarity(a: str, b: str) -> float:
    """计算两段文本的相似度 (0~1)"""
    if not a or not b:
        return 0.0
    ca, cb = clean_text(a), clean_text(b)
    if not ca or not cb:
        return 0.0
    # 双向包含检测（短文本完全被包含=高相似）
    if ca in cb or cb in ca:
        return 0.95
    return SequenceMatcher(None, ca, cb).ratio()


def frame_to_time(frame: int, fps: float) -> str:
    """帧号 → SRT时间格式 HH:MM:SS,mmm"""
    seconds = frame / fps
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)
    ms = td.microseconds // 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── 图像预处理 ────────────────────────────────────────────

class ImagePreprocessor:
    """字幕区域图像预处理 — 多策略流水线，最大化 OCR 准确率"""

    @staticmethod
    def all_variants(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """生成所有预处理变体"""
        variants = []

        # ── 1. 原始灰度 ──
        variants.append(("original", gray))

        # ── 2. 双边滤波去噪 (保边缘) ──
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        variants.append(("denoised", denoised))

        # ── 3. 锐化 (unsharp mask) — 对付模糊字幕 ──
        blur = cv2.GaussianBlur(denoised, (0, 0), 3.0)
        sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
        variants.append(("sharpened", sharpened))

        # ── 4. OTSU 二值化 (去噪后) ──
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) < 128:
            binary = cv2.bitwise_not(binary)  # 黑底白字
        variants.append(("binary", binary))

        # ── 5. CLAHE 对比度增强 ──
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        variants.append(("enhanced", enhanced))

        # ── 6-8. 多尺度自适应阈值 (适应不同字号/粗细) ──
        for bs, c_val in [(9, 2), (15, 3), (21, 4)]:
            adp = cv2.adaptiveThreshold(denoised, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, bs, c_val)
            variants.append((f"adaptive_{bs}", adp))

        # ── 9. 形态学闭运算 (连接断裂笔画) ──
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        variants.append(("morph_closed", morph))

        return variants

    @staticmethod
    def preprocess(image: np.ndarray, mode: str = "auto") -> list[tuple[str, np.ndarray]]:
        """按模式预处理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if mode == "auto":
            return ImagePreprocessor.all_variants(gray)
        elif mode == "binary":
            return ImagePreprocessor.all_variants(gray)[:2]
        elif mode == "adaptive":
            return [ImagePreprocessor.all_variants(gray)[2]]
        elif mode == "enhanced":
            return [ImagePreprocessor.all_variants(gray)[3]]
        return [("original", gray)]


# ── ROI 选择器 ────────────────────────────────────────────

class ROISelector:
    """交互式框选字幕区域"""

    @staticmethod
    def select(video_path: str) -> tuple[int, int, int, int]:
        """打开视频首帧让用户框选，返回 (x, y, w, h)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # 跳到 10% 位置取帧（跳过黑屏开场）
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total_frames * 0.1)))
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("无法读取视频帧")

        h, w = frame.shape[:2]
        print(f"\n{'='*55}")
        print(f"  📹 {Path(video_path).name}")
        print(f"  📐 {w}×{h}  |  {fps:.2f} FPS  |  {timedelta(seconds=int(duration))}")
        print(f"{'='*55}")
        print("\n  🖱️  鼠标拖拽框选字幕区域")
        print("     SPACE/ENTER = 确认    C = 取消    R = 重选\n")

        # 底边提示
        display = frame.copy()
        tips = [
            "Draw a rectangle over the subtitle area",
            "SPACE/ENTER=Confirm | C=Cancel | R=Reset",
        ]
        for i, tip in enumerate(tips):
            cv2.putText(display, tip, (10, h - 15 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        roi = cv2.selectROI("框选字幕区域 - Subtitle Region Selector", display,
                            showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()

        if roi == (0, 0, 0, 0):
            print("  ❌ 已取消")
            sys.exit(0)

        x, y, rw, rh = roi
        print(f"  ✅ 选中区域: x={x} y={y} w={rw} h={rh}")

        # 预览裁剪效果
        crop = frame[y:y+rh, x:x+rw]
        scale = min(900 / rw, 250 / rh, 3.0)
        preview = cv2.resize(crop, None, fx=scale, fy=scale)
        cv2.imshow("预览 (按任意键继续)", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return roi

    @staticmethod
    def load(roi_file: str) -> tuple[int, int, int, int] | None:
        """加载已保存的ROI配置"""
        if not os.path.exists(roi_file):
            return None
        with open(roi_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        roi = (data['x'], data['y'], data['w'], data['h'])
        print(f"  📐 已加载ROI: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
        return roi

    @staticmethod
    def save(roi: tuple, roi_file: str):
        """保存ROI配置"""
        x, y, w, h = roi
        with open(roi_file, 'w', encoding='utf-8') as f:
            json.dump({"x": x, "y": y, "w": w, "h": h}, f, indent=2)
        print(f"  💾 ROI已保存: {roi_file}")


# ── OCR 引擎 ──────────────────────────────────────────────

def _ocr_cleanup(text: str) -> str:
    """OCR 后处理：修正常见识别错误"""
    if not text:
        return text
    # 常见中文形近字纠正 (从识别结果推断)
    fixes = {
        # 英文常见
        "l ": "I ", " l": " I",  # 小写L→大写I (英文语境)
    }
    # 移除首尾残留标点
    text = text.strip(".,;:!?。，、；：！？·… ")
    # 合并多余空格
    text = re.sub(r" {2,}", " ", text)
    return text


class GpuManager:
    """GPU 加速管理器（全局单例，缓存引擎实例）"""
    _instance = None
    _engines = {}

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._gpu_available = self._detect_gpu()
        self._ocr_kwargs = self._build_kwargs()

    def _detect_gpu(self):
        try:
            import onnxruntime as rt
            providers = rt.get_available_providers()
            gpu_eps = {'CUDAExecutionProvider', 'TensorrtExecutionProvider', 'DmlExecutionProvider'}
            return any(ep in gpu_eps for ep in providers)
        except Exception:
            return False

    def _build_kwargs(self):
        if not self._gpu_available:
            return {}
        import onnxruntime as rt
        providers = rt.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            return {'use_cuda': True}
        elif 'DmlExecutionProvider' in providers:
            return {'use_dml': True}
        return {}

    def get_engine(self, use_gpu=True, chinese_lite=False):
        key = (use_gpu, chinese_lite)
        if key not in self._engines:
            from rapidocr_onnxruntime import RapidOCR
            kwargs = {'use_angle_cls': True}
            if use_gpu and self._gpu_available:
                kwargs.update(self._ocr_kwargs)
            if chinese_lite:
                kwargs['text_score'] = 0.5
            if kwargs:
                self._engines[key] = RapidOCR(**kwargs)
            else:
                self._engines[key] = RapidOCR()
        return self._engines[key]

    def warmup_async(self, on_ready=None):
        import threading
        def _warmup():
            try:
                self.get_engine(use_gpu=True)
                self.get_engine(use_gpu=True, chinese_lite=True)
            except Exception:
                pass
            if on_ready:
                on_ready()
        t = threading.Thread(target=_warmup, daemon=True)
        t.start()


class OCREngine:
    """RapidOCR 封装 — 复用 GpuManager 缓存的全局引擎实例"""

    def __init__(self, use_gpu: bool = True, chinese_lite: bool = False,
                 min_text_score: float = 0.5):
        self.use_gpu = use_gpu
        self.chinese_lite = chinese_lite
        self.min_text_score = min_text_score
        self._engine = None

    def init(self):
        if self._engine is not None:
            return
        mgr = GpuManager.instance()
        self._engine = mgr.get_engine(self.use_gpu, self.chinese_lite)
        print(f"  [OK] RapidOCR 就绪 ({'GPU' if self.use_gpu else 'CPU'}{', chinese_lite' if self.chinese_lite else ''})")

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

# ── 字幕提取核心 ──────────────────────────────────────────

class SubtitleExtractor:
    """视频字幕提取主类"""

    def __init__(self, video_path: str, roi: tuple = None,
                 frame_interval: int = 10,
                 similarity_threshold: float = 0.85,
                 box_thresh: float = 0.3,
                 text_score: float = 0.5,
                 preprocess_mode: str = "auto",
                 min_duration_ms: int = 500,
                 min_text_len: int = 2,
                 use_gpu: bool = True,
                 chinese_lite: bool = False,
                 num_workers: int = 4):
        self.video_path = video_path
        self.roi = roi
        self.frame_interval = max(1, frame_interval)
        self.similarity_threshold = similarity_threshold
        self.box_thresh = box_thresh
        self.text_score = text_score
        self.preprocess_mode = preprocess_mode
        self.min_duration_ms = min_duration_ms
        self.min_text_len = min_text_len

        self.ocr = OCREngine(use_gpu, chinese_lite)
        self.cap = None
        self.fps = 0.0
        self.total_frames = 0

    def extract(self) -> list[tuple[int, int, str]]:
        """单线程顺序读帧 + 多线程并行 OCR — 零 seek, 最快路径"""
        from queue import Queue

        if self.roi is None:
            raise ValueError("请先选择ROI区域")

        n = self.num_workers
        x, y, rw, rh = self.roi
        interval = self.frame_interval

        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0:
            self.fps = 30.0

        total_scan = self.total_frames // interval + 1
        t_start = time.time()

        print(f"\n  🎬 扫描 {self.total_frames} 帧 (间隔={interval}, 并行={n}线程)")

        # ── 共享队列 ──
        frame_queue = Queue(maxsize=n * 3)

        # ── Reader: 顺序读帧 ──
        def reader():
            fn = 0
            try:
                while True:
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

        # ── Workers: 并行 OCR ──
        hits_lock = threading.Lock()
        raw_hits = []
        scanned = [0]

        def worker():
            engine = OCREngine(self.ocr.use_gpu, self.ocr.chinese_lite,
                              min_text_score=self.text_score)
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
        last_pct = -1
        while any(w.is_alive() for w in workers):
            time.sleep(0.12)
            s = scanned[0]
            pct = int(s / total_scan * 100)
            if pct != last_pct:
                elapsed = time.time() - t_start
                eta = elapsed / max(s, 1) * (total_scan - s) if s > 0 else 999
                print(f"\r    进度: {pct:2d}% | 命中: {len(raw_hits):4d} | "
                      f"剩余: {timedelta(seconds=int(eta))}",
                      end="", flush=True)
                last_pct = pct

        for w in workers:
            w.join()
        cap.release()

        elapsed = time.time() - t_start
        print(f"\r    完成! 耗时 {timedelta(seconds=int(elapsed))} | "
              f"命中 {len(raw_hits)} 条\n")

        if not raw_hits:
            return []

        raw_hits.sort(key=lambda x: x[0])
        segments = self._merge_hits(raw_hits)
        segments = self._post_process(segments)
        return segments

    def _merge_hits(self, hits: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
        """
        将连续相似的OCR命中合并为字幕段
        使用三态状态机:
          - 相同/相似文本连续 → 延长结束帧
          - 文本变化但间隙不大 → 继续延展（同一句字幕逐字出现）
          - 文本完全变化且间隙大 → 新字幕
        """
        if not hits:
            return []

        segments = []
        batch_start = hits[0][0]
        batch_end = hits[0][0]
        batch_texts = [hits[0][1]]  # 收集一段内的所有文本变体
        best_text = hits[0][1]

        GAP_THRESHOLD_FRAMES = max(3, int(self.fps * 0.8))  # 约0.8秒的帧数

        for i in range(1, len(hits)):
            fn, text = hits[i]
            prev_fn = hits[i - 1][0]
            gap = fn - prev_fn
            sim = text_similarity(best_text, text)

            if sim >= self.similarity_threshold:
                # 相同/相似 → 延长
                batch_end = fn
                batch_texts.append(text)
                if len(text) > len(best_text):
                    best_text = text
            elif gap <= GAP_THRESHOLD_FRAMES and sim >= 0.5:
                # 可能是逐步出现的同一句 → 继续延展
                batch_end = fn
                batch_texts.append(text)
                if len(text) > len(best_text):
                    best_text = text
            else:
                # 真正的新字幕
                segments.append((batch_start, batch_end, best_text))
                batch_start = fn
                batch_end = fn
                batch_texts = [text]
                best_text = text

        # 最后一段
        segments.append((batch_start, batch_end, best_text))

        print(f"    阶段1-合并: {len(hits)} 条命中 → {len(segments)} 段")
        return segments

    def _post_process(self, segments: list) -> list:
        """后处理：过滤噪音、合并相邻相同文本、调整边界"""
        if not segments:
            return []

        # 1. 按最小持续时间过滤
        min_frames = int(self.min_duration_ms / 1000 * self.fps)
        filtered = []
        for s, e, t in segments:
            if e - s >= min_frames and t.strip():
                filtered.append((s, e, t))

        # 2. 合并相邻的相同/高度相似文本
        merged = []
        for sub in filtered:
            if not merged:
                merged.append(sub)
                continue
            prev_s, prev_e, prev_t = merged[-1]
            gap_frames = sub[0] - prev_e
            gap_ms = gap_frames / self.fps * 1000
            sim = text_similarity(prev_t, sub[2])

            if sim >= 0.9 and gap_ms < 1500:
                # 相同文本，延长结束时间
                merged[-1] = (prev_s, sub[1],
                              sub[2] if len(sub[2]) > len(prev_t) else prev_t)
            else:
                merged.append(sub)

        # 3. 合并过短的间隙（如果前后文本相同）
        final = []
        for i, sub in enumerate(merged):
            if i == 0:
                final.append(sub)
                continue
            prev_s, prev_e, prev_t = final[-1]
            gap = sub[0] - prev_e
            gap_ms = gap / self.fps * 1000

            if gap_ms < 300 and text_similarity(prev_t, sub[2]) >= 0.85:
                final[-1] = (prev_s, sub[1],
                             sub[2] if len(sub[2]) > len(prev_t) else prev_t)
            else:
                final.append(sub)

        print(f"    阶段2-后处理: {len(segments)} → {len(filtered)} → {len(final)} 段")
        return final

    def get_video_info(self) -> dict:
        """获取视频信息（需先执行extract或手动开cap）"""
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.total_frames / self.fps if self.fps > 0 else 0,
            "roi": self.roi,
            "frame_interval": self.frame_interval,
        }


# ── SRT 输出 ──────────────────────────────────────────────

def save_srt(subtitles: list[tuple], fps: float, output_path: str):
    """保存为SRT字幕文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (start_fn, end_fn, text) in enumerate(subtitles, 1):
            f.write(f"{i}\n")
            f.write(f"{frame_to_time(start_fn, fps)} --> {frame_to_time(end_fn, fps)}\n")
            f.write(f"{text}\n\n")
    print(f"  ✅ 字幕已保存: {output_path}")


def preview_subtitles(subtitles: list, fps: float, count: int = 8):
    """打印字幕预览"""
    print(f"\n  📋 字幕预览 (共 {len(subtitles)} 条):")
    print(f"  {'─'*50}")
    for i, (s, e, text) in enumerate(subtitles[:count], 1):
        t1, t2 = frame_to_time(s, fps), frame_to_time(e, fps)
        print(f"  {i:3d}. {t1} → {t2}")
        print(f"       {text}")
    if len(subtitles) > count:
        print(f"  ... 还有 {len(subtitles) - count} 条")


# ── CLI 入口 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="视频字幕OCR提取工具 - 框选区域 → OCR识别 → 输出SRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s video.mp4                           # 框选后提取，输出 video.srt
  %(prog)s video.mp4 -o out.srt -i 5           # 每5帧扫一次
  %(prog)s video.mp4 -i 5 -t 0.9              # 精细扫描+高阈值
  %(prog)s video.mp4 -t 0.9 -i 15              # 高相似阈值+稀疏扫描
  %(prog)s video.mp4 --save-roi                # 保存框选位置
  %(prog)s video.mp4 --no-select               # 使用已保存的框选位置

RapidOCR 引擎 — 基于 ONNX Runtime，内置中英文识别，无需配置语言
        """
    )
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("-o", "--output", help="输出SRT路径 (默认与视频同名)")
    parser.add_argument("-i", "--interval", type=int, default=10,
                        help="帧扫描间隔，默认10 (每10帧扫1帧)")
    parser.add_argument("-t", "--threshold", type=float, default=0.85,
                        help="文本相似度阈值 0~1，默认0.85，越高去重越严格")
    parser.add_argument("--min-duration", type=int, default=500,
                        help="字幕最短持续时间(ms)，默认500")
    parser.add_argument("--min-text-len", type=int, default=2,
                        help="最短文本长度(字符)，默认2")
    parser.add_argument("--preprocess", default="auto",
                        choices=["auto", "binary", "adaptive", "enhanced", "original"],
                        help="图像预处理模式，默认auto")
    parser.add_argument("--no-select", action="store_true",
                        help="跳过区域选择，使用已保存的ROI")
    parser.add_argument("--save-roi", action="store_true",
                        help="保存ROI区域到 .roi.json 文件")
    parser.add_argument("--gpu", action="store_true", default=None,
                        dest="use_gpu", help="启用GPU加速 (默认自动检测)")
    parser.add_argument("--no-gpu", action="store_false", default=None,
                        dest="use_gpu", help="强制使用CPU")
    parser.add_argument("--chinese-lite", action="store_true", default=False,
                        help="中文加速模式：关闭方向分类器，提速约30%% (仅适用于水平字幕)")
    parser.add_argument("--box-thresh", type=float, default=0.3,
                        help="文本框阈值 0~1, 越低检出越敏感 (默认:0.3)")
    parser.add_argument("--text-score", type=float, default=0.5,
                        help="文字置信度阈值 0~1, 越低越宽松 (默认:0.5)")
    parser.add_argument("--workers", type=int, default=4, metavar="N",
                        help="并行线程数 (默认:4, RTX 3060+ 推荐 4-8, 榨干性能选 8)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ 文件不存在: {args.video}")
        sys.exit(1)

    video_name = Path(args.video).stem
    output_path = args.output or str(Path(args.video).with_suffix(".srt"))
    roi_file = str(Path(args.video).with_suffix(".roi.json"))

    # ── 获取/选择 ROI ──
    roi = None
    if args.no_select:
        roi = ROISelector.load(roi_file)
        if roi is None:
            print("  ⚠️ 未找到ROI配置文件，进入手动选择...")
            roi = ROISelector.select(args.video)
    else:
        roi = ROISelector.select(args.video)

    if args.save_roi:
        ROISelector.save(roi, roi_file)

    # ── 提取 ──
    use_gpu = args.use_gpu if args.use_gpu is not None else True
    extractor = SubtitleExtractor(
        video_path=args.video,
        roi=roi,
        frame_interval=args.interval,
        similarity_threshold=args.threshold,
        box_thresh=args.box_thresh,
        text_score=args.text_score,
        preprocess_mode=args.preprocess,
        chinese_lite=args.chinese_lite,
        min_duration_ms=args.min_duration,
        min_text_len=args.min_text_len,
        use_gpu=use_gpu,
        num_workers=args.workers,
    )

    subtitles = extractor.extract()

    if not subtitles:
        print("\n  ⚠️ 未检测到字幕文本。建议:")
        print("    1. 检查框选区域是否覆盖字幕")
        print("    2. 尝试减小 --interval (如 -i 3)")
        print("    3. 尝试切换 --preprocess binary 或 adaptive")
        sys.exit(0)

    # ── 保存 ──
    info = extractor.get_video_info()
    save_srt(subtitles, info["fps"], output_path)
    preview_subtitles(subtitles, info["fps"])

    print(f"\n  🎉 完成! 共提取 {len(subtitles)} 条字幕")


if __name__ == "__main__":
    main()
