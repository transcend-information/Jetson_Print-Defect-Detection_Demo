import os
import cv2
import sys
import platform
import threading
import time
import traceback
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import ctypes
import math

import json

import numpy as np

import gc
import multiprocessing as mp


# --- 環境變數設定 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "1"
os.environ["FLAGS_check_nan_inf"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["GST_DEBUG"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"


# ============================================================
# === 展場模式設定                                          ===
# ============================================================
DEMO_MODE = True          # True: 隱藏 PCB；False: 兩個模式都顯示
AUTO_DETECT_INTERVAL = 10  # (保留:單張靜物模式才會用到,目前未啟用)
CYCLE_DELAY_MS = 200       # 連續偵測模式:每輪之間的喘息時間 (ms)
OCR_LANG = 'en'            # PaddleOCR 語言代碼：'en' / 'ch' / 'japan' 等

# YOLO / OCR 框顏色
YOLO_BOX_COLOR = "#f85149"   # 紅 (與左側 ROI 框同色 COLORS['danger'])
OCR_BOX_COLOR = "#bc8cff"    # 紫


PROJECT_CONFIGS = {
    "stick": {
        "label": "Sticker",
        "output_dir": r'/home/user/win_share/factory',
        "model_path": r'/home/user/win_share/stick/best.pt',
        "result_dir": r'/home/user/win_share/factory/yolo_result'
    },
    "pcb": {
        "label": "PCB Scratch",
        "output_dir": r'/home/user/win_share/pcb_scratch',
        "model_path": r'/home/user/win_share/pcb_scratch/best.pt',
        "result_dir": r'/home/user/win_share/pcb_scratch/yolo_result'
    }
}

ENABLED_MODES = ["stick"] if DEMO_MODE else list(PROJECT_CONFIGS.keys())


# === 顯示尺寸 (左 40% / 右 60%) ===
MAIN_DISPLAY_W = 700
MAIN_DISPLAY_H = 520
ROI_DISPLAY_W = 1100
ROI_DISPLAY_H = 600

# === 字體設定 ===
FONT_LABEL = ("Arial", 10, "bold")
FONT_ENTRY = ("Arial", 10)
FONT_TEXT = ("Arial", 10)
FONT_RESULT = ("Arial", 11, "bold")
FONT_BTN = ("Arial", 12, "bold")


# === 配色 ===
COLORS = {
    'bg': '#0d1117',
    'panel': '#161b22',
    'panel_alt': '#21262d',
    'border': '#30363d',
    'border_hover': '#484f58',
    'text': '#f0f6fc',
    'text_dim': '#8b949e',
    'accent': '#58a6ff',
    'accent_hover': '#79b8ff',
    'success': '#3fb950',
    'warning': '#d29922',
    'warning_hover': '#e3a008',
    'danger': '#f85149',
    'purple': '#bc8cff',
}


# ==========================================================================
# === YOLO Worker 子行程                                                  ===
# ==========================================================================

def _yolo_worker_main(model_path, in_q, out_q, ready_event, stop_event):
    """YOLO 子行程進入點。所有 GPU 操作都在這裡。"""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

    try:
        from ultralytics import YOLO as _YOLO
        import torch as _torch
        import numpy as _np

        print(f"[YoloWorker:{os.getpid()}] 載入模型: {model_path}")
        model = _YOLO(model_path)

        print(f"[YoloWorker:{os.getpid()}] 預熱中...")
        warmup_img = _np.zeros((640, 640, 3), dtype=_np.uint8)
        with _torch.no_grad():
            model(warmup_img, verbose=False, imgsz=640)
            if _torch.cuda.is_available():
                _torch.cuda.synchronize()

        ready_event.set()
        print(f"[YoloWorker:{os.getpid()}] ✅ 就緒")

    except Exception as e:
        print(f"[YoloWorker:{os.getpid()}] ❌ 初始化失敗: {e}")
        traceback.print_exc()
        try:
            out_q.put(("INIT_ERR", str(e)))
        except Exception:
            pass
        return

    while not stop_event.is_set():
        try:
            msg = in_q.get(timeout=1.0)
        except Exception:
            continue
        if msg is None:
            break

        try:
            img = msg
            with _torch.no_grad():
                results = model(img, verbose=False, imgsz=640)
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()

            boxes = []
            for box in results[0].boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                boxes.append((bx1, by1, bx2, by2, cls))

            out_q.put(("OK", boxes))

        except Exception as e:
            print(f"[YoloWorker:{os.getpid()}] 推論錯誤: {e}")
            traceback.print_exc()
            try:
                out_q.put(("ERR", str(e)))
            except Exception:
                pass

    print(f"[YoloWorker:{os.getpid()}] 收到停止訊號，退出")


# ==========================================================================
# === OCR Worker 子行程（PaddleOCR）                                      ===
# ==========================================================================

def _ocr_worker_main(lang, in_q, out_q, ready_event, stop_event):
    """OCR 子行程進入點。PaddleOCR 也用 GPU/CPU，必須隔離。"""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    try:
        from paddleocr import PaddleOCR
        import numpy as _np

        print(f"[OcrWorker:{os.getpid()}] 載入 PaddleOCR (lang={lang})...")
        ocr = None
        init_errs = []
        for kwargs in [
            {"use_angle_cls": True, "lang": lang, "show_log": False},
            {"use_angle_cls": True, "lang": lang},
            {"lang": lang},
        ]:
            try:
                ocr = PaddleOCR(**kwargs)
                break
            except TypeError as ie:
                init_errs.append(f"{kwargs}: {ie}")
                continue
        if ocr is None:
            raise RuntimeError(f"無法初始化 PaddleOCR，嘗試過: {init_errs}")

        print(f"[OcrWorker:{os.getpid()}] 預熱中...")
        warmup_img = _np.zeros((100, 300, 3), dtype=_np.uint8)
        try:
            ocr.ocr(warmup_img, cls=True)
        except TypeError:
            try:
                ocr.ocr(warmup_img)
            except Exception:
                pass

        ready_event.set()
        print(f"[OcrWorker:{os.getpid()}] ✅ 就緒")

    except ImportError:
        print(f"[OcrWorker:{os.getpid()}] ❌ 找不到 paddleocr，請執行: pip install paddlepaddle paddleocr")
        try:
            out_q.put(("INIT_ERR", "paddleocr 未安裝"))
        except Exception:
            pass
        return
    except Exception as e:
        print(f"[OcrWorker:{os.getpid()}] ❌ 初始化失敗: {e}")
        traceback.print_exc()
        try:
            out_q.put(("INIT_ERR", str(e)))
        except Exception:
            pass
        return

    while not stop_event.is_set():
        try:
            msg = in_q.get(timeout=1.0)
        except Exception:
            continue
        if msg is None:
            break

        try:
            img = msg
            try:
                result = ocr.ocr(img, cls=True)
            except TypeError:
                result = ocr.ocr(img)

            simplified = []
            if result and len(result) > 0 and result[0] is not None:
                for line in result[0]:
                    try:
                        bbox = line[0]
                        text_conf = line[1]
                        text = str(text_conf[0])
                        conf = float(text_conf[1])
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        simplified.append((
                            float(min(xs)), float(min(ys)),
                            float(max(xs)), float(max(ys)),
                            text, conf
                        ))
                    except (IndexError, TypeError, ValueError) as parse_e:
                        print(f"[OcrWorker:{os.getpid()}] 解析單行失敗: {parse_e}")
                        continue

            out_q.put(("OK", simplified))

        except Exception as e:
            print(f"[OcrWorker:{os.getpid()}] OCR 錯誤: {e}")
            traceback.print_exc()
            try:
                out_q.put(("ERR", str(e)))
            except Exception:
                pass

    print(f"[OcrWorker:{os.getpid()}] 收到停止訊號，退出")


# ==========================================================================
# === Worker 通用 Wrapper                                                ===
# ==========================================================================

class _BaseWorker:
    """Worker 共用基底:管理子行程生命週期 + queue + ready event。"""

    def __init__(self, name, target_fn, target_args):
        self.name = name
        self._target_fn = target_fn
        self._target_args_extra = target_args
        self.process = None
        self.in_q = None
        self.out_q = None
        self.ready_event = None
        self.stop_event = None

    def start(self):
        ctx = mp.get_context("spawn")
        self.in_q = ctx.Queue()
        self.out_q = ctx.Queue()
        self.ready_event = ctx.Event()
        self.stop_event = ctx.Event()
        all_args = list(self._target_args_extra) + [self.in_q, self.out_q, self.ready_event, self.stop_event]
        self.process = ctx.Process(
            target=self._target_fn,
            args=tuple(all_args),
            daemon=True,
            name=f"Worker-{self.name}",
        )
        self.process.start()
        print(f"[Main] 已啟動 {self.name} worker (PID: {self.process.pid})")

    def is_ready(self):
        return (self.ready_event is not None
                and self.ready_event.is_set()
                and self.process is not None
                and self.process.is_alive())

    def is_alive(self):
        return self.process is not None and self.process.is_alive()

    def predict(self, img, timeout=15):
        if not self.is_ready():
            raise RuntimeError(f"Worker [{self.name}] 尚未就緒")

        try:
            while True:
                self.out_q.get_nowait()
        except Exception:
            pass

        self.in_q.put(img)
        try:
            kind, payload = self.out_q.get(timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Worker [{self.name}] 無回應 (timeout {timeout}s): {e}")

        if kind == "OK":
            return payload
        else:
            raise RuntimeError(f"Worker [{self.name}] 錯誤: {payload}")

    def stop(self, timeout=3):
        if self.process is None:
            return
        try:
            if self.stop_event is not None:
                self.stop_event.set()
            if self.in_q is not None:
                try:
                    self.in_q.put_nowait(None)
                except Exception:
                    pass
            self.process.join(timeout=timeout)
        except Exception:
            pass
        try:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        except Exception:
            pass
        try:
            if self.process.is_alive():
                self.process.kill()
        except Exception:
            pass
        self.process = None
        self.in_q = None
        self.out_q = None
        self.ready_event = None
        self.stop_event = None


class YoloWorker(_BaseWorker):
    def __init__(self, model_path, mode_name):
        super().__init__(f"YOLO-{mode_name}", _yolo_worker_main, (model_path,))


class OcrWorker(_BaseWorker):
    def __init__(self, lang):
        super().__init__("OCR", _ocr_worker_main, (lang,))


# ==========================================================================
# === 主應用程式                                                          ===
# ==========================================================================

class App:
    def __init__(self, root, window_title):
        self.root = root
        self.root.title(window_title)

        self.status_var = tk.StringVar()
        self.smooth_rect = None

        try:
            icon_img = tk.PhotoImage(file="ts.png")
            self.root.iconphoto(True, icon_img)
        except Exception as e:
            print(f"[System] 無法載入圖示: {e}")

        self.root.resizable(True, True)

        self.project_var = tk.StringVar(value="stick")
        self.current_cfg = PROJECT_CONFIGS["stick"]

        # --- 狀態變數 ---
        self.frozen_roi_img = None        
        self.gallery_img = None           
        self.tracked_boxes = []           
        self.current_frame = None
        self.roi_x1 = self.roi_y1 = self.roi_x2 = self.roi_y2 = None

        self.drag_start_x = self.drag_start_y = None

        self.detected_stickers = []
        self.detected_boxes = []          
        self.detected_ocr_boxes = []      

        self.roi_scale = 1.0
        self.roi_offset_x = 0
        self.roi_offset_y = 0
        self.status_timer = None

        self.best_rect_info = None

        self.base_roi_scale = 1.0
        self.base_offset_x = 0
        self.base_offset_y = 0

        self.current_roi_w = "--"
        self.current_roi_h = "--"
        self.current_yolo_time = "--"
        self.current_ocr_time = "--"

        # --- 自動偵測排程相關 ---
        self.auto_after_id = None
        self.detection_token = 0
        self.is_detecting = False

        # --- Worker 子行程 ---
        self.workers = {}
        self.ocr_worker = None
        self.current_loaded_mode = "stick"

        # --- Camera ---
        self.cap, self.target_cam_path = self._scan_and_open_camera()

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
            self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[System] 相機啟動成功: {self.target_cam_path} ({self.actual_w}x{self.actual_h})")
        else:
            print("[System] 🚨 嚴重錯誤：初始無法找到任何可輸出畫面的相機設備！")
            self.actual_w, self.actual_h = 1920, 1080

        self._calculate_display_size()

        self.setup_ui()

        try:
            self.root.attributes('-fullscreen', True)
            print("[System] 已進入全螢幕模式 (按 ESC 切換 windowed)")
        except Exception as e:
            print(f"[System] 全螢幕設定失敗: {e}")

        self._start_all_workers()

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.update_video()

    def _scan_and_open_camera(self):
        import glob

        paths = glob.glob("/dev/v4l/by-path/*video-index0")
        for path in paths:
            tmp_cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if tmp_cap.isOpened():
                ret, frame = tmp_cap.read()
                if ret and frame is not None:
                    return tmp_cap, path
            tmp_cap.release()

        video_nodes = sorted(glob.glob("/dev/video*"))
        for path in video_nodes:
            tmp_cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if tmp_cap.isOpened():
                ret, frame = tmp_cap.read()
                if ret and frame is not None:
                    return tmp_cap, path
            tmp_cap.release()

        for idx in range(6):
            tmp_cap = cv2.VideoCapture(idx)
            if tmp_cap.isOpened():
                ret, frame = tmp_cap.read()
                if ret and frame is not None:
                    return tmp_cap, idx
            tmp_cap.release()

        return None, None

    # ==========================================================
    # ===    Worker 管理                                     ===
    # ==========================================================

    def _start_all_workers(self):
        for mode_key in ENABLED_MODES:
            self._start_yolo_worker(mode_key)
        self._start_ocr_worker()

    def _start_yolo_worker(self, mode_key):
        local_model_path = os.path.join("models", f"{mode_key}_best.pt")

        if not os.path.exists(local_model_path):
            print(f"[Main] ❌ 找不到模型: {local_model_path}")
            self.set_status(f"❌ Detection model not found: {mode_key}", 5)
            return

        old = self.workers.get(mode_key)
        if old is not None:
            try:
                old.stop(timeout=2)
            except Exception as e:
                print(f"[Main] 停止舊 YOLO worker 警告: {e}")

        worker = YoloWorker(local_model_path, mode_key)
        worker.start()
        self.workers[mode_key] = worker
        self._monitor_worker_ready_async("Detection model", worker)

    def _start_ocr_worker(self):
        if self.ocr_worker is not None:
            try:
                self.ocr_worker.stop(timeout=2)
            except Exception as e:
                print(f"[Main] 停止舊 OCR worker 警告: {e}")

        worker = OcrWorker(OCR_LANG)
        worker.start()
        self.ocr_worker = worker
        self._monitor_worker_ready_async("OCR", worker)

    def _monitor_worker_ready_async(self, label, worker):
        def _bg():
            ready = False
            try:
                ready = worker.ready_event.wait(timeout=120)
            except Exception as e:
                print(f"[Main] 等 {label} ready 出錯: {e}")
            if ready:
                msg = f"✅ {label} ready"
                print(f"[Main] {msg}")
                self.root.after(0, lambda: self.set_status(msg, 2))
            else:
                msg = f"❌ {label} startup failed or timeout"
                print(f"[Main] {msg}")
                self.root.after(0, lambda: self.set_status(msg, 5))

        threading.Thread(target=_bg, daemon=True).start()

    def _get_active_yolo_worker(self):
        return self.workers.get(self.current_loaded_mode)

    # ==========================================================
    # ===    UI                                              ===
    # ==========================================================
    def setup_ui(self):
        self.root.configure(bg=COLORS['bg'])

        main_container = tk.Frame(self.root, bg=COLORS['bg'])
        main_container.pack(fill="both", expand=True, padx=12, pady=(10, 6))

        main_container.columnconfigure(0, weight=4)
        main_container.columnconfigure(1, weight=6)
        main_container.rowconfigure(0, weight=1)

        # ============================================
        # === 左側:CONFIG & PREVIEW (40%)          ===
        # ============================================
        left_panel = tk.Frame(main_container, bg=COLORS['panel'],
                              highlightthickness=1,
                              highlightbackground=COLORS['border'])
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        left_header = tk.Frame(left_panel, bg=COLORS['panel'])
        left_header.pack(fill="x", padx=14, pady=(12, 4))

        tk.Label(left_header, text="◆  CONFIG & PREVIEW",
                 font=("Arial", 12, "bold"),
                 fg=COLORS['accent'], bg=COLORS['panel']).pack(side="left")

        tk.Label(left_header, text="LIVE",
                 font=("Consolas", 10, "bold"),
                 fg=COLORS['success'], bg=COLORS['panel']).pack(side="right")

        tk.Frame(left_panel, bg=COLORS['border'], height=1).pack(fill="x", padx=14, pady=(2, 10))

        # --- 模式選擇 ---
        mode_frame = tk.Frame(left_panel, bg=COLORS['panel'])
        mode_frame.pack(fill="x", padx=14, pady=(0, 10))

        for key in ENABLED_MODES:
            cfg = PROJECT_CONFIGS[key]
            rb = tk.Radiobutton(
                mode_frame, text=cfg["label"], variable=self.project_var,
                value=key, font=("Arial", 9, "bold"),
                bg=COLORS['panel_alt'], fg=COLORS['text'],
                selectcolor=COLORS['accent'],
                activebackground=COLORS['accent_hover'],
                activeforeground='white',
                indicatoron=0, padx=10, pady=7,
                bd=0, highlightthickness=1,
                highlightbackground=COLORS['border'],
                highlightcolor=COLORS['accent'],
                command=self.on_project_change
            )
            rb.pack(side="left", expand=True, fill="x", padx=2)

        # --- 主預覽畫面 ---
        canvas_center_frame = tk.Frame(left_panel, bg=COLORS['panel'])
        canvas_center_frame.pack(fill="both", expand=True, padx=14, pady=(2, 14))

        canvas_wrap = tk.Frame(canvas_center_frame, bg=COLORS['border'])
        canvas_wrap.pack(expand=True, anchor="center")

        self.canvas_main = tk.Canvas(canvas_wrap, width=self.display_w,
                                     height=self.display_h, bg="#000000",
                                     highlightthickness=0)
        self.canvas_main.pack(padx=1, pady=1)
        self.img_container_main = self.canvas_main.create_image(0, 0, anchor="nw")
        self.roi_rect_main = self.canvas_main.create_rectangle(
            0, 0, 0, 0, outline=COLORS['danger'], width=2)

        

        # ============================================
        # === 右側:ZOOM INSPECTION (60%)           ===
        # ============================================
        right_panel = tk.Frame(main_container, bg=COLORS['panel'],
                               highlightthickness=1,
                               highlightbackground=COLORS['border'])
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        right_header = tk.Frame(right_panel, bg=COLORS['panel'])
        right_header.pack(fill="x", padx=14, pady=(12, 4))

        tk.Label(right_header, text="◆  ZOOM INSPECTION",
                 font=("Arial", 12, "bold"),
                 fg=COLORS['accent'], bg=COLORS['panel']).pack(side="left")

        self.lbl_countdown = tk.Label(
            right_header, text="⚡ AWAITING ROI",
            width=20, anchor="e",
            font=("Consolas", 13, "bold"),
            fg=COLORS['warning'], bg=COLORS['panel']
        )
        self.lbl_countdown.pack(side="right")

        tk.Frame(right_panel, bg=COLORS['border'], height=1).pack(fill="x", padx=14, pady=(2, 10))

        # --- ROI Canvas ---
        roi_wrap = tk.Frame(right_panel, bg=COLORS['border'])
        roi_wrap.pack(padx=14, pady=(2, 6), expand=False)

        self.canvas_roi = tk.Canvas(roi_wrap, width=ROI_DISPLAY_W,
                                    height=ROI_DISPLAY_H, bg="#050505",
                                    highlightthickness=0)
        self.canvas_roi.pack(padx=1, pady=1)
        self.img_container_roi = self.canvas_roi.create_image(
            ROI_DISPLAY_W // 2, ROI_DISPLAY_H // 2, anchor="center")

        # --- Stats Bar ---
        stats_bar = tk.Frame(right_panel, bg=COLORS['panel'])
        stats_bar.pack(fill="x", padx=14, pady=(0, 8))

        self.lbl_tech_stats = tk.Label(
            stats_bar,
            text="ROI -- × --     DETECT -- ms     OCR -- ms",
            width=48, anchor="w",  # width 微調避免字體變大後超出邊界
            font=("Consolas", 18, "bold"),  # 👈 從 14 放大到 18
            fg=COLORS['text_dim'], bg=COLORS['panel']
        )
        self.lbl_tech_stats.pack(side="left")

        # --- OCR 結果區塊 ---
        ocr_section = tk.Frame(right_panel, bg=COLORS['panel'])
        ocr_section.pack(fill="both", expand=True, padx=14, pady=(8, 14))

        ocr_header = tk.Frame(ocr_section, bg=COLORS['panel'])
        ocr_header.pack(fill="x", pady=(0, 6))

        tk.Label(ocr_header, text="◆  OCR  READOUT",
                 font=("Arial", 12, "bold"),
                 fg=COLORS['purple'], bg=COLORS['panel']).pack(side="left")

        # OCR 字數顯示
        self.lbl_ocr_count = tk.Label(
            ocr_header, text="0 strings",
            width=12, anchor="e",
            font=("Consolas", 24, "bold"),  # 👈 配合放大
            fg=COLORS['text_dim'], bg=COLORS['panel']
        )
        self.lbl_ocr_count.pack(side="right")

        # 瑕疵數量顯示 (預設先隱藏，等有瑕疵再顯示)
        self.lbl_defect_count = tk.Label(
            ocr_header, text="",
            width=0, anchor="e",
            font=("Consolas", 24, "bold"),
            fg=COLORS['danger'], bg=COLORS['panel']
        )
        self.lbl_defect_count.pack(side="right", padx=(0, 15))

        self.ocr_results_frame = tk.Frame(
            ocr_section, bg=COLORS['panel_alt'],
            height=220,
            highlightthickness=1,
            highlightbackground=COLORS['border'],
        )
        self.ocr_results_frame.pack(fill="both", expand=True)
        self.ocr_results_frame.pack_propagate(False)
        self._show_ocr_placeholder("(awaiting first OCR result...)")

        # ==========================================
        # === 狀態列                                ===
        # ==========================================
        self.status_bar = tk.Label(
            self.root, textvariable=self.status_var,
            bg=COLORS['panel_alt'], fg=COLORS['text_dim'],
            anchor="w", font=("Arial", 9),
            padx=12, pady=5
        )
        self.status_bar.pack(side="bottom", fill="x")
        self.set_status("System starting... loading workers (10-30s)")

        self.lbl_vis_res = tk.Label(self.root)

        # --- Bindings ---
        self.canvas_main.bind("<Button-1>", self.on_main_mouse_down)
        self.canvas_main.bind("<B1-Motion>", self.on_main_mouse_drag)
        self.canvas_main.bind("<ButtonRelease-1>", self.on_main_mouse_up)

        self.canvas_roi.bind("<MouseWheel>", self.on_roi_scroll)
        self.canvas_roi.bind("<Button-4>", self.on_roi_scroll)
        self.canvas_roi.bind("<Button-5>", self.on_roi_scroll)

        self.root.bind("<s>", self.save_training_data)
        self.root.bind("<S>", self.save_training_data)

        self.root.bind("<Escape>", self._toggle_fullscreen)

    def _show_ocr_placeholder(self, msg):
        for w in self.ocr_results_frame.winfo_children():
            w.destroy()
        
        for r in range(10):
            self.ocr_results_frame.rowconfigure(r, weight=0)
        for c in range(10):
            self.ocr_results_frame.columnconfigure(c, weight=0)
            
        self.ocr_results_frame.rowconfigure(0, weight=1)
        self.ocr_results_frame.columnconfigure(0, weight=1)
        
        tk.Label(
            self.ocr_results_frame, text=msg,
            bg=COLORS['panel_alt'], fg=COLORS['text_dim'],
            font=("Consolas", 14, "bold")
        ).grid(row=0, column=0)

    def _toggle_fullscreen(self, event=None):
        try:
            current = bool(self.root.attributes('-fullscreen'))
            self.root.attributes('-fullscreen', not current)
            print(f"[System] 全螢幕切換: {not current}")
        except Exception as e:
            print(f"[System] 切換全螢幕失敗: {e}")

    

    def on_project_change(self):
        selected_key = self.project_var.get()
        self.current_cfg = PROJECT_CONFIGS[selected_key]
        self.current_loaded_mode = selected_key

        worker = self.workers.get(selected_key)
        if worker is None or not worker.is_alive():
            self.set_status(f"⚠️ Worker [{selected_key}] missing, restarting...", 0)
            self._start_yolo_worker(selected_key)
        else:
            self.set_status(f"✅ Switched: {self.current_cfg['label']}", 2)

    # ==========================================================
    # ===    狀態 / Stats                                      ===
    # ==========================================================
    def update_tech_stats(self, w=None, h=None, yolo=None, ocr=None):
        if w is not None: self.current_roi_w = w
        if h is not None: self.current_roi_h = h
        if yolo is not None:
            self.current_yolo_time = (
                f"{yolo:.0f}" if isinstance(yolo, float) else yolo
            )
        if ocr is not None:
            self.current_ocr_time = (
                f"{ocr:.0f}" if isinstance(ocr, float) else ocr
            )

        msg = (f"ROI {self.current_roi_w} × {self.current_roi_h}   "
               f"DETECT {self.current_yolo_time} ms   "
               f"OCR {self.current_ocr_time} ms")

        color = COLORS['accent'] if self.frozen_roi_img is not None else COLORS['text_dim']
        self.lbl_tech_stats.config(text=msg, fg=color)

    def set_status(self, msg, delay=0):
        if msg.startswith("✅"):
            color = COLORS['success']
        elif msg.startswith("⚠️"):
            color = COLORS['warning']
        elif msg.startswith("❌"):
            color = COLORS['danger']
        elif msg.startswith("⏳") or msg.startswith("🔄"):
            color = COLORS['accent']
        elif msg.startswith("💾"):
            color = COLORS['accent']
        else:
            color = COLORS['text_dim']

        self.status_bar.config(fg=color)

        if self.status_timer: self.root.after_cancel(self.status_timer)
        self.status_var.set(msg)
        if delay > 0:
            def _restore():
                self.status_var.set("AUTO mode running")
                self.status_bar.config(fg=COLORS['text_dim'])
            self.status_timer = self.root.after(delay * 1000, _restore)

    def _calculate_display_size(self):
        ratio = min(MAIN_DISPLAY_W / self.actual_w, MAIN_DISPLAY_H / self.actual_h)
        self.display_w = int(self.actual_w * ratio)
        self.display_h = int(self.actual_h * ratio)
        self.scale_x = self.actual_w / self.display_w
        self.scale_y = self.actual_h / self.display_h

    # ==========================================================
    # ===    自動偵測排程                                      ===
    # ==========================================================
    def _schedule_auto_detect(self, delay_ms):
        if self.auto_after_id is not None:
            try:
                self.root.after_cancel(self.auto_after_id)
            except Exception:
                pass
        self.auto_after_id = self.root.after(delay_ms, self._do_auto_detect)

    def _cancel_auto_detect(self):
        if self.auto_after_id is not None:
            try:
                self.root.after_cancel(self.auto_after_id)
            except Exception:
                pass
            self.auto_after_id = None

    # ==========================================================
    # ===    多貼紙偵測核心 與 顯示畫廊                        ===
    # ==========================================================
    
    def _calculate_iou(self, boxA, boxB):
        """計算兩個矩形的交併比 (Intersection over Union)，用於追蹤同一張貼紙"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea)

    def _is_same_object(self, boxA, boxB):
        """混合追蹤法: 只要有重疊，或是中心點距離相近，就視為同一物件"""
        if self._calculate_iou(boxA, boxB) > 0.15: 
            return True
            
        cxA, cyA = boxA[0] + boxA[2]/2, boxA[1] + boxA[3]/2
        cxB, cyB = boxB[0] + boxB[2]/2, boxB[1] + boxB[3]/2
        dist = ((cxA - cxB)**2 + (cyA - cyB)**2)**0.5
        if dist < 150:  
            return True
            
        return False

    def _deskew_image(self, img):
        """轉正處理核心：用來把裁切出來的圖物理轉平"""
        if img is None or img.size == 0:
            return img
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return img

            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 1000: 
                return img

            rect = cv2.minAreaRect(c)
            angle = rect[-1]

            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90

            if abs(angle) < 1.0:
                return img

            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception as e:
            print(f"[Main] 影像轉正失敗: {e}")
            return img

    def _build_gallery_image(self):
        """建立右側專用的預覽畫廊：直接顯示抓到並已經轉平的貼紙"""
        if not getattr(self, 'detected_stickers', []):
            return self.frozen_roi_img.copy() if self.frozen_roi_img is not None else None
            
        rows = []
        for s in self.detected_stickers:
            crop = s.get('crop')
            if crop is None or crop.size == 0:
                continue
                
            drawn = crop.copy()
            for bx1, by1, bx2, by2, cls in s.get('boxes', []):
                cv2.rectangle(drawn, (int(bx1), int(by1)), (int(bx2), int(by2)), (73, 81, 248), 2) 
                cv2.putText(drawn, "Defect", (int(bx1)+3, max(int(by1)-5, 12)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (73, 81, 248), 1, cv2.LINE_AA)
            
            rows.append(drawn)
        
        if not rows:
            return self.frozen_roi_img.copy() if self.frozen_roi_img is not None else None
            
        if len(rows) == 1:
            return rows[0]
        
        max_w = max(r.shape[1] for r in rows)
        padded_rows = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded_rows.append(r)
            padded_rows.append(np.zeros((20, max_w, 3), dtype=np.uint8)) 
        padded_rows.pop() 
        return np.vstack(padded_rows)

    def _do_auto_detect(self):
        """執行一輪自動偵測"""
        if self.current_frame is None or self.roi_x1 is None:
            self._schedule_auto_detect(1000)
            return

        yolo_worker = self._get_active_yolo_worker()
        ocr_worker = self.ocr_worker

        if yolo_worker is None or not yolo_worker.is_ready():
            self.set_status("⏳ Waiting for detection model...", 0)
            self._schedule_auto_detect(1000)
            return

        try:
            rough_roi = self.current_frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2].copy()
        except Exception as e:
            print(f"[Main] 切 ROI 失敗: {e}")
            self._schedule_auto_detect(1000)
            return

        if rough_roi.size == 0:
            self._schedule_auto_detect(1000)
            return

        sticker_candidates = []
        if self.current_loaded_mode == "stick":
            try:
                raw_candidates = self._detect_stickers(rough_roi)
                
                new_tracked_boxes = []
                for cand in raw_candidates:
                    box = cand['rect']
                    is_tracked = False
                    
                    for t_box in self.tracked_boxes:
                        if self._is_same_object(box, t_box):
                            is_tracked = True
                            new_tracked_boxes.append(box) 
                            break

                    if not is_tracked:
                        sticker_candidates.append(cand)
                        new_tracked_boxes.append(box)

                self.tracked_boxes = new_tracked_boxes

            except Exception as e:
                print(f"[Main] 多貼紙偵測失敗,退回整張: {e}")
                traceback.print_exc()

            if not sticker_candidates:
                self.lbl_countdown.config(
                    text="⚡ WAITING STICKER",
                    fg=COLORS['warning']
                )
                self._schedule_auto_detect(CYCLE_DELAY_MS)
                return
        else:
            sticker_candidates = [{
                'crop':   rough_roi.copy(),
                'offset': (0, 0),
                'rect':   (0, 0, rough_roi.shape[1], rough_roi.shape[0]),
                'area':   rough_roi.shape[0] * rough_roi.shape[1],
            }]

        self.is_detecting = True
        self.detection_token += 1
        token = self.detection_token
        
        self.lbl_countdown.config(
            text="⚡ SCANNING ROI", fg=COLORS['accent']
        )

        threading.Thread(
            target=self._do_inference_in_bg,
            args=(token, rough_roi, sticker_candidates, yolo_worker, ocr_worker),
            daemon=True
        ).start()

    def _detect_stickers(self, rough_roi,
                         min_area=3000,
                         max_area_ratio=0.90,
                         min_aspect=0.15,
                         max_aspect=8.0,
                         padding=5,
                         edge_margin=15): 
        """
        在 rough_roi 上找出所有貼紙候選區(支援多顆 SSD)。
        """
        if rough_roi is None or rough_roi.size == 0:
            return []

        gray = cv2.cvtColor(rough_roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel_small = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(thresh, kernel_small, iterations=2)
        kernel_big = np.ones((11, 31), np.uint8)
        closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_big)

        flood_mask = closed.copy()
        f_h, f_w = flood_mask.shape[:2]
        f_temp_mask = np.zeros((f_h + 2, f_w + 2), np.uint8)
        for sx, sy in [(0, 0), (f_w - 1, 0), (0, f_h - 1), (f_w - 1, f_h - 1)]:
            if flood_mask[sy, sx] == 255:
                cv2.floodFill(flood_mask, f_temp_mask, (sx, sy), 0)

        contours, _ = cv2.findContours(
            flood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        total_area = f_h * f_w
        candidates = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            if area > total_area * max_area_ratio:
                continue

            px, py, pw, ph = cv2.boundingRect(c)
            if pw == 0 or ph == 0:
                continue

            if (px < edge_margin or
                py < edge_margin or
                (px + pw) > (f_w - edge_margin) or
                (py + ph) > (f_h - edge_margin)):
                continue

            extent = area / (pw * ph)
            if extent < 0.75:  
                continue

            aspect = pw / float(ph)
            if aspect < min_aspect or aspect > max_aspect:
                continue

            px2 = max(0, px - padding)
            py2 = max(0, py - padding)
            pw2 = min(f_w - px2, pw + padding * 2)
            ph2 = min(f_h - py2, ph + padding * 2)

            raw_crop = rough_roi[py2:py2 + ph2, px2:px2 + pw2].copy()
            if raw_crop.size == 0:
                continue
            
            deskewed_crop = self._deskew_image(raw_crop)

            candidates.append({
                'crop':   deskewed_crop,
                'offset': (px2, py2),
                'rect':   (px2, py2, pw2, ph2),
                'area':   area,
            })

        candidates.sort(key=lambda d: (d['rect'][1] // 50, d['rect'][0]))
        return candidates

    def _do_inference_in_bg(self, token, rough_roi, sticker_candidates,
                            yolo_worker, ocr_worker):
        sticker_results = []
        ocr_results = []
        yolo_err = None
        ocr_err = None
        yolo_ms_total = 0.0
        ocr_ms = 0.0

        for sticker in sticker_candidates:
            try:
                t0 = time.time()
                boxes = yolo_worker.predict(sticker['crop'], timeout=15)
                yolo_ms_total += (time.time() - t0) * 1000

                sticker_results.append({
                    'crop':   sticker['crop'],
                    'offset': sticker['offset'],
                    'rect':   sticker['rect'],
                    'boxes':  boxes,  
                })
            except Exception as e:
                yolo_err = e
                print(f"[Main] YOLO 推論失敗 (offset={sticker['offset']}): {e}")
                sticker_results.append({
                    'crop':   sticker['crop'],
                    'offset': sticker['offset'],
                    'rect':   sticker['rect'],
                    'boxes':  [],
                })

        if ocr_worker is not None and ocr_worker.is_ready():
            try:
                t0 = time.time()
                ocr_results = ocr_worker.predict(rough_roi, timeout=15)
                ocr_ms = (time.time() - t0) * 1000
            except Exception as e:
                ocr_err = e
                print(f"[Main] OCR 推論失敗: {e}")
        else:
            ocr_ms = -1

        self.root.after(0, lambda: self._on_inference_done(
            token, rough_roi, sticker_results, ocr_results,
            yolo_err, ocr_err, yolo_ms_total, ocr_ms
        ))

    def _on_inference_done(self, token, rough_roi, sticker_results, ocr_results,
                           yolo_err, ocr_err, yolo_ms, ocr_ms):
        self.is_detecting = False

        if token != self.detection_token:
            print(f"[Main] 偵測結果已過期 (token {token} != {self.detection_token}),丟棄")
            self._schedule_auto_detect(CYCLE_DELAY_MS)
            return

        if yolo_err is not None:
            self.set_status(f"⚠️ Detection failed, restarting in background...", 5)
            self._start_yolo_worker(self.current_loaded_mode)

        if ocr_err is not None:
            self.set_status(f"⚠️ OCR failed, restarting in background...", 5)
            self._start_ocr_worker()

        self.frozen_roi_img = rough_roi
        self.detected_stickers = sticker_results
        
        self.gallery_img = self._build_gallery_image()

        if self.gallery_img is not None:
            h, w = self.gallery_img.shape[:2]
            ratio = min(ROI_DISPLAY_W / w, ROI_DISPLAY_H / h)
            self.roi_scale = ratio
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            self.roi_offset_x = (ROI_DISPLAY_W - new_w) // 2
            self.roi_offset_y = (ROI_DISPLAY_H - new_h) // 2
            self.base_roi_scale = ratio
            self.base_offset_x = self.roi_offset_x
            self.base_offset_y = self.roi_offset_y

        self.detected_boxes = []
        total_defects = 0
        for s in sticker_results:
            ox, oy = s['offset']
            for bx1, by1, bx2, by2, cls in s['boxes']:
                self.detected_boxes.append({
                    'cls': cls,
                    'img_rect': [bx1 + ox, by1 + oy, bx2 + ox, by2 + oy]
                })
                total_defects += 1

        self.detected_ocr_boxes = []
        for x1, y1, x2, y2, text, conf in ocr_results:
            self.detected_ocr_boxes.append({
                'text': text, 'conf': conf,
                'img_rect': [x1, y1, x2, y2]
            })

        self._redraw_inspection()
        self._update_ocr_panel(ocr_results)

        self.current_roi_h, self.current_roi_w = rough_roi.shape[:2]
        self.update_tech_stats(
            self.current_roi_w, self.current_roi_h,
            float(yolo_ms),
            float(ocr_ms) if ocr_ms >= 0 else "N/A"
        )

        # 🌟 核心修改：儲存當前瑕疵數量，並依據數量切換 UI 顯示
        self.current_defects = total_defects
        if total_defects == 0:
            self.lbl_defect_count.config(text="", width=0) # 隱藏 Defects
        else:
            self.lbl_defect_count.config(text=f"Defects: {total_defects}", width=12, fg=COLORS['danger']) # 顯示紅字

        self._schedule_auto_detect(CYCLE_DELAY_MS)

    def _redraw_inspection(self):
        if getattr(self, 'gallery_img', None) is None:
            return

        h, w = self.gallery_img.shape[:2]
        new_w = int(w * self.roi_scale)
        new_h = int(h * self.roi_scale)
        if new_w <= 0 or new_h <= 0:
            return

        try:
            resized = cv2.resize(self.gallery_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.tk_roi_photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            self.canvas_roi.itemconfig(self.img_container_roi, image=self.tk_roi_photo, anchor="nw")
            self.canvas_roi.coords(self.img_container_roi, self.roi_offset_x, self.roi_offset_y)
        except Exception as e:
            print(f"[Main] 畫廊重繪失敗: {e}")
            return

        self.canvas_roi.delete("yolo_box")
        self.canvas_roi.delete("ocr_box")
        self.canvas_roi.delete("glow_border") # 刪除舊的光暈

        # 🌟 狀態漸層色外框 (右側 ZOOM 畫廊區域)
        is_defect = getattr(self, 'current_defects', 0) > 0
        glow_colors = ['#f85149', '#9b2c28', '#451210'] if is_defect else ['#3fb950', '#236b2d', '#0e2e13']
        
        for i, c_hex in enumerate(glow_colors):
            self.canvas_roi.create_rectangle(
                self.roi_offset_x - i*2, self.roi_offset_y - i*2, 
                self.roi_offset_x + new_w + i*2, self.roi_offset_y + new_h + i*2, 
                outline=c_hex, width=2, tags="glow_border"
            )

    def _update_ocr_panel(self, ocr_results):
        """動態網格魔法：算好數量後，平均撐開所有空間絕對置中！"""
        try:
            for w in self.ocr_results_frame.winfo_children():
                w.destroy()

            if not ocr_results:
                self._show_ocr_placeholder("(no text detected)")
                self.lbl_ocr_count.config(text="0 strings")
                return

            sorted_results = sorted(ocr_results, key=lambda r: r[1])
            n = len(sorted_results)

            if n <= 3:
                cols = n
            elif n <= 8:
                cols = 4
            else:
                cols = 5
                
            rows = math.ceil(n / cols)

            for r in range(10):
                self.ocr_results_frame.rowconfigure(r, weight=0)
            for c in range(10):
                self.ocr_results_frame.columnconfigure(c, weight=0)

            for r in range(rows):
                self.ocr_results_frame.rowconfigure(r, weight=1)
            for c in range(cols):
                self.ocr_results_frame.columnconfigure(c, weight=1, uniform="col")

            if rows == 1:
                font_size = 14
                pad_y = 6
            elif rows == 2:
                font_size = 13
                pad_y = 4
            elif rows == 3:
                font_size = 11
                pad_y = 2
            else:
                font_size = 10
                pad_y = 1

            for i, (x1, y1, x2, y2, text_str, conf) in enumerate(sorted_results):
                r, c = divmod(i, cols)
                
                lbl = tk.Label(
                    self.ocr_results_frame,
                    text=text_str,
                    bg=COLORS['border'],
                    fg=COLORS['text'],
                    font=("Consolas", font_size, "bold"),
                    padx=8, pady=pad_y,
                    wraplength=160, 
                    justify="center"
                )
                lbl.grid(row=r, column=c, padx=4, pady=4)

            self.lbl_ocr_count.config(text=f"{n} strings")
        except Exception as e:
            print(f"[Main] update OCR panel failed: {e}")

    # ==========================================================
    # ===    存訓練資料(按 S):每張 sticker 各存一筆           ===
    # ==========================================================
    def save_training_data(self, event=None):
        if not getattr(self, 'detected_stickers', None):
            self.set_status("⚠️ No detection result to save yet", 2)
            return

        mode = self.project_var.get()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{mode}_{timestamp}"

        img_dir = f"dataset_update/{mode}/images/train"
        lbl_dir = f"dataset_update/{mode}/labels/train"
        verify_dir = f"dataset_update/{mode}/verify"
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        os.makedirs(verify_dir, exist_ok=True)

        saved_count = 0
        for i, sticker in enumerate(self.detected_stickers):
            crop = sticker.get('crop')
            boxes = sticker.get('boxes', [])

            if crop is None or crop.size == 0:
                continue

            sub_name = f"{base_name}_s{i:02d}"
            img_path = os.path.join(img_dir, f"{sub_name}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{sub_name}.txt")
            verify_path = os.path.join(verify_dir, f"{sub_name}_verify.jpg")

            cv2.imwrite(img_path, crop)

            ch, cw = crop.shape[:2]
            verify = crop.copy()
            with open(lbl_path, "w") as f:
                for bx1, by1, bx2, by2, cls in boxes:
                    cv2.rectangle(verify,
                                  (int(bx1), int(by1)),
                                  (int(bx2), int(by2)),
                                  (0, 0, 255), 2)
                    xc = max(0.0, min(1.0, ((bx1 + bx2) / 2.0) / cw))
                    yc = max(0.0, min(1.0, ((by1 + by2) / 2.0) / ch))
                    bw = max(0.0, min(1.0, (bx2 - bx1) / cw))
                    bh = max(0.0, min(1.0, (by2 - by1) / ch))
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            cv2.imwrite(verify_path, verify)
            saved_count += 1

        if self.current_frame is not None and self.roi_x1 is not None:
            loc_mode = "locator_pcb"
            loc_img_dir = f"dataset_update/{loc_mode}/images/train"
            loc_lbl_dir = f"dataset_update/{loc_mode}/labels/train"
            loc_verify_dir = f"dataset_update/{loc_mode}/verify"
            os.makedirs(loc_img_dir, exist_ok=True)
            os.makedirs(loc_lbl_dir, exist_ok=True)
            os.makedirs(loc_verify_dir, exist_ok=True)

            loc_img_path = os.path.join(loc_img_dir, f"{base_name}_full.jpg")
            loc_lbl_path = os.path.join(loc_lbl_dir, f"{base_name}_full.txt")
            loc_verify_path = os.path.join(loc_verify_dir, f"{base_name}_full_verify.jpg")

            cv2.imwrite(loc_img_path, self.current_frame)

            fh, fw = self.current_frame.shape[:2]
            rx1 = max(0, min(fw, self.roi_x1))
            ry1 = max(0, min(fh, self.roi_y1))
            rx2 = max(0, min(fw, self.roi_x2))
            ry2 = max(0, min(fh, self.roi_y2))

            with open(loc_lbl_path, "w") as f_loc:
                xc = ((rx1 + rx2) / 2.0) / fw
                yc = ((ry1 + ry2) / 2.0) / fh
                bw = (rx2 - rx1) / fw
                bh = (ry2 - ry1) / fh
                f_loc.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            loc_verify = self.current_frame.copy()
            cv2.rectangle(loc_verify, (rx1, ry1), (rx2, ry2), (255, 165, 0), 4)
            cv2.imwrite(loc_verify_path, loc_verify)

        self.set_status(f"💾 Saved {saved_count} sticker sample(s): {base_name}", 3)
        print(f"[Data] 樣本: {base_name}, 共 {saved_count} 張")

    # ==========================================================
    # ===    滾輪縮放                                          ===
    # ==========================================================
    def on_roi_scroll(self, event):
        if getattr(self, 'gallery_img', None) is None:
            return

        if event.num == 4 or getattr(event, 'delta', 0) > 0:
            factor = 1.15
        elif event.num == 5 or getattr(event, 'delta', 0) < 0:
            factor = 0.85
        else:
            return

        new_scale = self.roi_scale * factor

        if factor < 1 and new_scale < (self.base_roi_scale * 0.98):
            self.roi_scale = self.base_roi_scale
            self.roi_offset_x = self.base_offset_x
            self.roi_offset_y = self.base_offset_y
            self._redraw_inspection()
            return

        if new_scale > 3.0: new_scale = 3.0

        actual_factor = new_scale / self.roi_scale

        mx = self.canvas_roi.canvasx(event.x)
        my = self.canvas_roi.canvasy(event.y)

        self.roi_offset_x = mx - (mx - self.roi_offset_x) * actual_factor
        self.roi_offset_y = my - (my - self.roi_offset_y) * actual_factor

        self.roi_scale = new_scale
        self._redraw_inspection()

    # ==========================================================
    # ===    主畫面 ROI 框選                                   ===
    # ==========================================================
    def on_main_mouse_down(self, e):
        self.best_rect_info = None
        self.canvas_main.delete("sticker_box_main") 
        self.drag_start_x, self.drag_start_y = e.x, e.y
        self.canvas_main.coords(self.roi_rect_main, e.x, e.y, e.x, e.y)

    def on_main_mouse_drag(self, e):
        if not self.drag_start_x: return
        self.canvas_main.coords(self.roi_rect_main, self.drag_start_x, self.drag_start_y, e.x, e.y)

    def on_main_mouse_up(self, e):
        if not self.drag_start_x: return
        x1, y1, x2, y2 = (sorted([self.drag_start_x, e.x])[0],
                          sorted([self.drag_start_y, e.y])[0],
                          sorted([self.drag_start_x, e.x])[1],
                          sorted([self.drag_start_y, e.y])[1])
        rx1, ry1 = int(x1 * self.scale_x), int(y1 * self.scale_y)
        rx2, ry2 = int(x2 * self.scale_x), int(y2 * self.scale_y)
        rx1 = max(0, rx1); ry1 = max(0, ry1)
        rx2 = min(self.actual_w, rx2); ry2 = min(self.actual_h, ry2)
        if (rx2 - rx1) > 10 and (ry2 - ry1) > 10:
            self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = rx1, ry1, rx2, ry2
            self._cancel_auto_detect()
            self._schedule_auto_detect(200)
            self.lbl_countdown.config(text="⚡ STARTING...", fg=COLORS['accent'])
            self.set_status("✅ ROI set, continuous detection starting...", 2)
        else:
            self.canvas_main.coords(self.roi_rect_main, 0, 0, 0, 0)
            self.canvas_main.delete("sticker_box_main") 
            self.roi_x1 = None
            self._cancel_auto_detect()
            self.lbl_countdown.config(text="⚡ AWAITING ROI", fg=COLORS['warning'])
        self.drag_start_x = None

    # ==========================================================
    # ===    主相機更新                                        ===
    # ==========================================================
    def update_video(self):
        if not hasattr(self, 'fail_count'):
            self.fail_count = 0
            self.start_work_time = None
            self.is_first_run = True

        ret = False
        frame = None

        try:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
        except Exception as e:
            ret = False
            print(f"[System] 影像讀取拋出異常: {e}")

        if not ret or frame is None:
            if self.fail_count == 0:
                self.moment_of_failure = time.time()

            self.fail_count += 1

            if self.fail_count < 40:
                self.root.after(10, self.update_video)
                return
            else:
                now_str = datetime.now().strftime("%H:%M:%S")
                if self.start_work_time:
                    total_run_duration = self.moment_of_failure - self.start_work_time
                    if self.is_first_run:
                        print(f"\n[{now_str}] 🚨 【首次崩潰】！")
                        print(f"[{now_str}] 📊 [數據] 從程式開啟並顯示畫面，到第一次掛掉，共撐了: {total_run_duration:.2f} 秒")
                        self.is_first_run = False
                    else:
                        print(f"\n[{now_str}] 🚨 【再次崩潰】！")
                        print(f"[{now_str}] 📊 [數據] 自上次恢復連線後，到本次掛掉，共正常運作: {total_run_duration:.2f} 秒")

                self.fail_count = 0
                self.start_work_time = None
                self.reconnect_camera()
                return

        if self.start_work_time is None:
            self.start_work_time = time.time()
            state_msg = "初始影像顯示" if self.is_first_run else "影像恢復連線"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [System] {state_msg}，開始計時運作時長...")

        self.fail_count = 0
        self.current_frame = frame

        display_frame = cv2.resize(frame, (self.display_w, self.display_h),
                                   interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        self.tk_photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas_main.itemconfig(self.img_container_main, image=self.tk_photo)
        
        # === 🌟 核心優化：讓左側捕捉框跟隨 30FPS 即時更新 ===
        self.canvas_main.delete("sticker_box_main")
        if self.roi_x1 is not None and self.current_loaded_mode == "stick":
            try:
                live_roi = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
                if live_roi.size > 0:
                    # 使用非常輕量的設定即時掃描 (只供顯示用，不送推論)
                    live_cands = self._detect_stickers(live_roi)
                    for cand in live_cands:
                        px, py, pw, ph = cand['rect']
                        orig_x1 = self.roi_x1 + px
                        orig_y1 = self.roi_y1 + py
                        orig_x2 = orig_x1 + pw
                        orig_y2 = orig_y1 + ph

                        disp_x1 = orig_x1 / self.scale_x
                        disp_y1 = orig_y1 / self.scale_y
                        disp_x2 = orig_x2 / self.scale_x
                        disp_y2 = orig_y2 / self.scale_y

                        # 🌟 改為淡藍色框 (COLORS['accent'])
                        self.canvas_main.create_rectangle(
                            disp_x1, disp_y1, disp_x2, disp_y2,
                            outline=COLORS['accent'], width=2, tags="sticker_box_main"
                        )
            except Exception:
                pass
        # ==========================================================

        self.canvas_main.tag_raise(self.roi_rect_main)
        self.canvas_main.tag_raise("sticker_box_main") 
        self.root.after(33, self.update_video)

    def reconnect_camera(self):
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{now}] [System] 正在掃描可用節點以恢復連線...")

        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

        start_search = time.time()
        new_cap, new_path = self._scan_and_open_camera()
        search_duration = (time.time() - start_search) * 1000

        if new_cap is not None:
            self.cap = new_cap
            self.target_cam_path = new_path
            finish_now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{finish_now}] [System] 🔗 硬體恢復成功於 {self.target_cam_path} (搜尋耗時: {search_duration:.1f} ms)")

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
            self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.root.after(100, self.update_video)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 仍找不到裝置，1秒後再次嘗試...")
            self.root.after(1000, self.update_video)

    def quit_app(self):
        print("\n[System] 啟動徹底清理程序...")

        self._cancel_auto_detect()

        for key, worker in list(self.workers.items()):
            try:
                print(f"[System] 停止 YOLO worker: {key}")
                worker.stop(timeout=2)
            except Exception as e:
                print(f"[System] 停止 worker [{key}] 警告: {e}")
        self.workers.clear()

        if self.ocr_worker is not None:
            try:
                print(f"[System] 停止 OCR worker")
                self.ocr_worker.stop(timeout=2)
            except Exception as e:
                print(f"[System] 停止 OCR worker 警告: {e}")
            self.ocr_worker = None

        if getattr(self, 'cap', None) is not None and self.cap.isOpened():
            self.cap.release()
            print("[System] 相機硬體已釋放。")

        self.root.destroy()
        print("[System] 已徹底關閉程式。")
        os._exit(0)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    root = tk.Tk()
    app = App(root, "Defect Inspector  v4.2")
    root.mainloop()