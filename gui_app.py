"""
口罩检测 — 图形化集成界面 (tkinter)
所有功能整合在一个窗口：训练 / 单张预测 / 批量预测 / 实时检测 / 报告导出
"""

import os
import sys
import io
import threading
import time
import csv
import queue
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (MODEL_PATH, TRAIN_DIR, IMG_SIZE, CLASS_NAMES,
                    SAVE_DIR, LOG_DIR, load_trained_model)

# matplotlib 嵌入 tkinter
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class RedirectText(io.StringIO):
    """将 stdout 重定向到 tkinter Text 控件"""
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
    def write(self, s):
        self.widget.insert(tk.END, s)
        self.widget.see(tk.END)
        self.widget.update_idletasks()
    def flush(self):
        pass


class MaskDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("口罩检测系统 — Mask Detection")
        self.root.geometry("960x700")
        self.root.minsize(800, 600)

        self.model = None
        self.cap = None
        self.cam_running = False
        self.train_thread = None
        self._detector = None
        self._pending_detector = None
        self._display_timer = None
        self._frame_queue = queue.Queue(maxsize=2)
        self._infer_interval = 3
        self._frame_count = 0
        self._worker_stop = threading.Event()
        self._fps_hist = []

        self._build_tabs()

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=2, pady=2)

        self._build_train_tab()
        self._build_single_tab()
        self._build_batch_tab()
        self._build_realtime_tab()
        self._build_report_tab()

    # ═══════════════ 训练标签页 ═══════════════
    def _build_train_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="训练")

        # 参数区
        pf = ttk.LabelFrame(tab, text="训练参数", padding=10)
        pf.pack(fill='x', padx=10, pady=5)

        row1 = ttk.Frame(pf)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="批次大小:").pack(side='left')
        self.batch_var = tk.IntVar(value=16)
        ttk.Spinbox(row1, from_=4, to=128, textvariable=self.batch_var, width=8).pack(side='left', padx=5)

        ttk.Label(row1, text="最大轮数:").pack(side='left', padx=(20, 0))
        self.epoch_var = tk.IntVar(value=15)
        ttk.Spinbox(row1, from_=1, to=100, textvariable=self.epoch_var, width=8).pack(side='left', padx=5)

        ttk.Label(row1, text="学习率:").pack(side='left', padx=(20, 0))
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(row1, textvariable=self.lr_var, width=8).pack(side='left', padx=5)

        self.train_btn = ttk.Button(pf, text="开始训练", command=self._start_training)
        self.train_btn.pack(pady=(10, 0))

        self.train_progress = ttk.Progressbar(pf, mode='indeterminate')

        # 日志区
        lf = ttk.LabelFrame(tab, text="训练日志 (实时)", padding=5)
        lf.pack(fill='both', expand=True, padx=10, pady=5)

        self.log_text = tk.Text(lf, font=('Consolas', 9), wrap='none',
                                 bg='#1e1e1e', fg='#d4d4d4',
                                 insertbackground='white')
        self.log_text.pack(fill='both', expand=True)

        scroll = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scroll.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=scroll.set)

    def _start_training(self):
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showwarning("训练中", "训练已在运行中")
            return
        self.train_btn.config(state='disabled')
        self.train_progress.pack(fill='x', pady=(5, 0))
        self.train_progress.start()
        self.log_text.delete(1.0, tk.END)
        self.train_thread = threading.Thread(target=self._do_train, daemon=True)
        self.train_thread.start()

    def _do_train(self):
        from train import train
        old_stdout = sys.stdout
        sys.stdout = RedirectText(self.log_text)
        try:
            train(
                batch_size=self.batch_var.get(),
                epochs=self.epoch_var.get(),
                learning_rate=float(self.lr_var.get()),
            )
        except Exception as e:
            print(f"\n[错误] 训练失败: {e}")
        finally:
            sys.stdout = old_stdout
            self.root.after(0, self._on_train_done)

    def _on_train_done(self):
        self.train_progress.stop()
        self.train_progress.pack_forget()
        self.train_btn.config(state='normal')
        self.log_text.insert(tk.END, "\n>>> 训练结束 <<<\n")
        self.log_text.see(tk.END)
        self.model = None  # 让预测标签页重新加载

    # ═══════════════ 单张预测标签页 ═══════════════
    def _build_single_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="单张预测")

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill='x', padx=10, pady=10)
        ttk.Button(btn_frame, text="选择图片", command=self._single_predict).pack(side='left')
        ttk.Button(btn_frame, text="粘贴图片路径", command=self._paste_predict).pack(side='left', padx=10)

        self.single_path_var = tk.StringVar()
        ttk.Entry(btn_frame, textvariable=self.single_path_var, width=50).pack(side='left', padx=5)

        # 图片 + 结果并排
        display = ttk.Frame(tab)
        display.pack(fill='both', expand=True, padx=10, pady=5)

        self.img_label = ttk.Label(display, relief='sunken', anchor='center')
        self.img_label.pack(side='left', fill='both', expand=True, padx=(0, 10))

        result_frame = ttk.LabelFrame(display, text="预测结果", padding=10)
        result_frame.pack(side='right', fill='both', padx=(10, 0))

        self.single_result_text = tk.Text(result_frame, font=('微软雅黑', 12),
                                           width=30, height=15, wrap='word')
        self.single_result_text.pack(fill='both', expand=True)

    def _get_model(self):
        if self.model is None:
            if not os.path.exists(MODEL_PATH):
                messagebox.showerror("错误", f"模型不存在: {MODEL_PATH}\n请先训练")
                return None
            from tensorflow.keras.models import load_model
            self.model = load_model(MODEL_PATH, compile=False)
        return self.model

    def _single_predict(self):
        path = filedialog.askopenfilename(
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.single_path_var.set(path)
            self._do_single_predict(path)

    def _paste_predict(self):
        path = self.single_path_var.get().strip()
        if path:
            self._do_single_predict(path)

    def _do_single_predict(self, path):
        model = self._get_model()
        if model is None:
            return
        from predict import predict as predict_fn
        try:
            name, conf, probs = predict_fn(model, path)
        except Exception as e:
            messagebox.showerror("错误", str(e))
            return

        # 显示图片
        img = Image.open(path).resize((300, 300))
        photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=photo)
        self.img_label.image = photo

        # 显示结果
        self.single_result_text.delete(1.0, tk.END)
        self.single_result_text.insert(tk.END, f"文件: {os.path.basename(path)}\n\n")
        self.single_result_text.insert(tk.END, f"预测: {name}\n")
        self.single_result_text.insert(tk.END, f"置信度: {conf:.2%}\n\n")
        self.single_result_text.insert(tk.END, "各类概率:\n")
        for k, v in probs.items():
            bar = '█' * int(v * 30)
            self.single_result_text.insert(tk.END, f"  {k}: {v:.4f}\n")
            self.single_result_text.insert(tk.END, f"  {bar}\n")

    # ═══════════════ 批量预测标签页 ═══════════════
    def _build_batch_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="批量预测")

        ctrl = ttk.Frame(tab)
        ctrl.pack(fill='x', padx=10, pady=10)
        ttk.Label(ctrl, text="图片目录:").pack(side='left')
        self.batch_dir_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.batch_dir_var, width=50).pack(side='left', padx=5)
        ttk.Button(ctrl, text="浏览", command=self._pick_batch_dir).pack(side='left')
        ttk.Button(ctrl, text="开始推理", command=self._batch_predict).pack(side='left', padx=10)

        self.batch_csv_var = tk.StringVar(value='batch_result.csv')
        ttk.Label(ctrl, text="输出:").pack(side='left', padx=(20, 0))
        ttk.Entry(ctrl, textvariable=self.batch_csv_var, width=20).pack(side='left', padx=5)

        # 结果表格
        treef = ttk.Frame(tab)
        treef.pack(fill='both', expand=True, padx=10, pady=5)
        cols = ('file', 'pred', 'conf', 'mask_p', 'nomask_p')
        self.batch_tree = ttk.Treeview(treef, columns=cols, show='headings', height=20)
        self.batch_tree.heading('file', text='文件名')
        self.batch_tree.heading('pred', text='预测')
        self.batch_tree.heading('conf', text='置信度')
        self.batch_tree.heading('mask_p', text='Mask概率')
        self.batch_tree.heading('nomask_p', text='NoMask概率')
        self.batch_tree.column('file', width=200)
        self.batch_tree.column('pred', width=80)
        self.batch_tree.column('conf', width=80)
        self.batch_tree.column('mask_p', width=100)
        self.batch_tree.column('nomask_p', width=100)
        self.batch_tree.pack(side='left', fill='both', expand=True)

        scroll = ttk.Scrollbar(treef, command=self.batch_tree.yview)
        scroll.pack(side='right', fill='y')
        self.batch_tree.configure(yscrollcommand=scroll.set)

        self.batch_stats = ttk.Label(tab, text="")
        self.batch_stats.pack(pady=5)

    def _pick_batch_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.batch_dir_var.set(d)

    def _batch_predict(self):
        d = self.batch_dir_var.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showerror("错误", "请选择有效的图片目录")
            return
        model = self._get_model()
        if model is None:
            return

        from batch_predict import collect_images, preprocess_batch
        self.batch_tree.delete(*self.batch_tree.get_children())

        paths = collect_images(d)
        if not paths:
            messagebox.showinfo("提示", "目录中没有图片")
            return

        batch, valid = preprocess_batch(paths)
        if len(batch) == 0:
            return
        probs = model.predict(batch, verbose=0)

        mask_count = 0
        for i, row in enumerate(probs):
            cls = int(row.argmax())
            if cls == 0:
                mask_count += 1
            self.batch_tree.insert('', 'end', values=(
                os.path.basename(valid[i]),
                CLASS_NAMES[cls],
                f"{row[cls]:.4f}",
                f"{row[0]:.4f}",
                f"{row[1]:.4f}",
            ))

        total = len(valid)
        csv_path = self.batch_csv_var.get().strip()
        if csv_path:
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['文件名', '预测', '置信度', 'Mask概率', 'NoMask概率'])
                for row in self.batch_tree.get_children():
                    w.writerow(self.batch_tree.item(row)['values'])

        self.batch_stats.config(
            text=f"总计: {total}  |  Mask: {mask_count} ({mask_count/total*100:.1f}%)  |  "
                 f"NoMask: {total-mask_count} ({(total-mask_count)/total*100:.1f}%)  →  {csv_path}")

    # ═══════════════ 实时检测标签页 ═══════════════
    def _build_realtime_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="实时检测")

        ctrl = ttk.Frame(tab)
        ctrl.pack(fill='x', padx=10, pady=5)

        ttk.Label(ctrl, text="检测器:").pack(side='left')
        self.detector_var = tk.StringVar(value='ssd')
        self.detector_combo = ttk.Combobox(ctrl, textvariable=self.detector_var,
                                            values=['ssd', 'ssd-e2e', 'haar'],
                                            width=12, state='readonly')
        self.detector_combo.pack(side='left', padx=5)
        self.detector_combo.bind('<<ComboboxSelected>>', self._on_detector_change)

        ttk.Label(ctrl, text="推理间隔:").pack(side='left', padx=(15, 0))
        self.skip_var = tk.IntVar(value=2)
        self.skip_spin = ttk.Spinbox(ctrl, from_=1, to=10, textvariable=self.skip_var,
                                      width=4, command=self._on_skip_change)
        self.skip_spin.pack(side='left', padx=5)
        ttk.Label(ctrl, text="帧").pack(side='left')

        self.cam_btn = ttk.Button(ctrl, text="启动摄像头", command=self._toggle_camera)
        self.cam_btn.pack(side='left', padx=10)

        self.fps_label = ttk.Label(ctrl, text="FPS: --")
        self.fps_label.pack(side='left', padx=20)

        self.face_count_label = ttk.Label(ctrl, text="人脸: --")
        self.face_count_label.pack(side='left', padx=10)

        self.cam_status = ttk.Label(ctrl, text="", foreground='blue')
        self.cam_status.pack(side='left', padx=10)

        self.cam_label = ttk.Label(tab, relief='sunken', anchor='center')
        self.cam_label.pack(fill='both', expand=True, padx=10, pady=5)
        self.cam_label.configure(text="点击「启动摄像头」开始")

        self._frame_queue = queue.Queue(maxsize=1)  # 永远只保留最新帧，消除延迟
        self._infer_interval = 2  # 每 2 帧推理一次（流畅不卡）
        self._frame_count = 0
        self._display_timer = None
        self._worker_stop = threading.Event()  # 线程安全退出信号

    def _toggle_camera(self):
        if self.cam_running:
            self._stop_camera()
        else:
            self.cam_btn.config(state='disabled')
            self.cam_status.config(text="正在打开摄像头...", foreground='blue')
            threading.Thread(target=self._start_camera, daemon=True).start()

    def _stop_camera(self):
        self.cam_running = False
        self._worker_stop.set()  # 通知工作线程退出
        if self._display_timer:
            self.root.after_cancel(self._display_timer)
            self._display_timer = None
        # 等工作线程结束（最多 1 秒），然后安全释放摄像头
        time.sleep(0.3)
        if self.cap:
            self.cap.release()
            self.cap = None
        self._detector = None
        self._pending_detector = None
        # 清空队列
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self.cam_btn.config(text="启动摄像头", state='normal')
        self.cam_status.config(text="")
        self.fps_label.config(text="FPS: --")
        self.face_count_label.config(text="人脸: --")
        self.cam_label.configure(image='', text="点击「启动摄像头」开始")

    def _start_camera(self):
        import cv2
        backends = [(cv2.CAP_DSHOW, 'DirectShow'), (cv2.CAP_ANY, 'Default')]
        cap = None
        last_error = ""
        for bid, bname in backends:
            try:
                c = cv2.VideoCapture(0, bid)
            except Exception as e:
                last_error = str(e)
                continue
            if not c.isOpened():
                c.release()
                continue
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(10):
                ret, test = c.read()
                if ret and test.mean() > 10:
                    cap = c
                    break
            if cap is not None:
                break
            c.release()

        if cap is None:
            msg = "摄像头不可用 — 请检查是否被其他应用占用"
            if last_error:
                msg += f" ({last_error})"
            self.root.after(0, lambda m=msg: (
                self.cam_status.config(text=m, foreground='red'),
                self.cam_btn.config(state='normal')
            ))
            return

        self.cap = cap
        self.cam_running = True
        self._worker_stop.clear()
        self._detector = None
        self._pending_detector = None
        self._frame_count = 0
        self._fps_hist = []
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self.root.after(0, lambda: (
            self.cam_btn.config(text="停止摄像头", state='normal'),
            self.cam_status.config(text="加载模型中...", foreground='blue')
        ))
        threading.Thread(target=self._worker_loop, daemon=True).start()
        self._start_display()

    def _worker_loop(self):
        """后台线程：读帧 → 推理 → 画框 → 入队。用 Event 安全退出"""
        import cv2
        det_type = self.detector_var.get()
        detector = None
        frame_idx = 0
        load_fail_count = 0
        cached_dets = []  # 缓存上一次识别结果，框永远不消失

        try:
            while self.cam_running and not self._worker_stop.is_set() and self.cap is not None:
                pending = getattr(self, '_pending_detector', None)
                if pending is not None:
                    det_type = pending
                    detector = None
                    cached_dets = []
                    self._pending_detector = None

                ret, frame = self.cap.read()
                if not ret:
                    if self._worker_stop.wait(0.01):
                        break
                    continue

                frame_idx += 1
                do_infer = (frame_idx % self._infer_interval == 0) and detector is not None

                if detector is None:
                    try:
                        import predict_realtime as prt
                        detector = prt.MaskDetector(detector_type=det_type)
                        load_fail_count = 0
                        self.root.after(0, lambda: self.cam_status.config(
                            text="检测中", foreground='green'))
                    except Exception:
                        load_fail_count += 1
                        if load_fail_count == 1:
                            self.root.after(0, lambda: self.cam_status.config(
                                text="模型加载失败，重试中...", foreground='red'))
                        if load_fail_count > 5:
                            self.root.after(0, lambda: self.cam_status.config(
                                text="模型加载失败，请检查 models/ 目录", foreground='red'))
                            break
                        self._worker_stop.wait(1)
                        continue

                # 推理帧 → 更新缓存；非推理帧 → 复用缓存
                if do_infer:
                    try:
                        _, cached_dets = detector.predict(frame, with_gradcam=False)
                    except Exception:
                        pass

                # 缩放 + 画框（每帧都画，用缓存的结果，框永不消失）
                oh, ow = frame.shape[:2]
                dw, dh = 640, 480
                display = cv2.resize(frame, (dw, dh))
                sx, sy = dw / ow, dh / oh

                for d in cached_dets:
                    x1, y1, x2, y2 = d['bbox']
                    nx1 = int(x1 * sx)
                    ny1 = int(y1 * sy)
                    nx2 = int(x2 * sx)
                    ny2 = int(y2 * sy)
                    color = (0, 255, 0) if d['class'] == 0 else (0, 0, 255)
                    label = d.get('label', CLASS_NAMES[d['class']])
                    text = f"{label} {d['confidence']:.0%}"

                    cv2.rectangle(display, (nx1, ny1), (nx2, ny2), color, 3)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
                    cv2.rectangle(display, (nx1, ny1 - th - 8), (nx1 + tw + 6, ny1), color, -1)
                    cv2.putText(display, text, (nx1 + 3, ny1 - 5),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

                display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

                # 丢弃旧帧 → 放入最新帧
                while True:
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        break
                try:
                    self._frame_queue.put_nowait((display, len(cached_dets)))
                except queue.Full:
                    pass
        finally:
            # 工作线程退出时确保摄像头释放
            cap = self.cap
            self.root.after(0, lambda c=cap: self._release_camera(c))

    def _release_camera(self, cap):
        if cap and cap.isOpened():
            cap.release()

    def _start_display(self):
        """主线程定时器：从队列取帧刷新 GUI，~30fps"""
        try:
            data = self._frame_queue.get_nowait()
            display, face_count = data
        except queue.Empty:
            if self.cam_running and not self._worker_stop.is_set():
                self._display_timer = self.root.after(33, self._start_display)
            return

        try:
            img = Image.fromarray(display)
            photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=photo, text='')
            self.cam_label.image = photo
        except Exception:
            pass

        self.face_count_label.config(text=f"人脸: {face_count}")

        now = time.time()
        self._fps_hist.append(now)
        while len(self._fps_hist) > 1 and self._fps_hist[-1] - self._fps_hist[0] > 3:
            self._fps_hist.pop(0)
        if len(self._fps_hist) >= 2:
            span = self._fps_hist[-1] - self._fps_hist[0]
            fps = (len(self._fps_hist) - 1) / span if span > 0 else 0
            self.fps_label.config(text=f"FPS: {fps:.0f}")

        if self.cam_running and not self._worker_stop.is_set():
            self._display_timer = self.root.after(33, self._start_display)

    def _on_detector_change(self, event):
        """切换检测器时不重启摄像头，只通知工作线程重建 detector"""
        if self.cam_running:
            new_type = self.detector_var.get()
            self._pending_detector = new_type
            self.cam_status.config(text=f"切换中: {new_type}...", foreground='orange')

    def _on_skip_change(self):
        """实时调整推理间隔，无需重启"""
        v = self.skip_var.get()
        try:
            self._infer_interval = int(v)
        except ValueError:
            self._infer_interval = 2

    # ═══════════════ 报告/导出标签页 ═══════════════
    def _build_report_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="报告/导出")

        # 报告区
        rf = ttk.LabelFrame(tab, text="生成报告图表", padding=15)
        rf.pack(fill='x', padx=10, pady=10)

        ttk.Label(rf, text="生成：数据集样本网格 + 模型结构图 + 混淆矩阵").pack()
        ttk.Button(rf, text="一键生成报告", command=self._run_report).pack(pady=10)

        self.report_status = ttk.Label(rf, text="", foreground='green')
        self.report_status.pack()

        # 导出区
        ef = ttk.LabelFrame(tab, text="TFLite 模型导出", padding=15)
        ef.pack(fill='x', padx=10, pady=10)

        ttk.Label(ef, text="导出：FP16 (~11MB) + 动态量化 (~6MB) + INT8 (~5MB)").pack()
        self.tflite_frame = ef
        ttk.Button(ef, text="导出 TFLite", command=self._run_export).pack(pady=10)
        self.export_status = ttk.Label(ef, text="", foreground='green')
        self.export_status.pack()

        # 曲线预览区
        cf = ttk.LabelFrame(tab, text="训练曲线预览", padding=5)
        cf.pack(fill='both', expand=True, padx=10, pady=5)

        curve_path = os.path.join(SAVE_DIR, 'training_curve.png')
        if os.path.exists(curve_path):
            img = Image.open(curve_path).resize((700, 250))
            photo = ImageTk.PhotoImage(img)
            lbl = ttk.Label(cf, image=photo)
            lbl.image = photo
            lbl.pack()
        else:
            ttk.Label(cf, text="（训练后自动显示）").pack()

    def _run_report(self):
        self.report_status.config(text="正在生成...")
        self.root.update()
        try:
            from report import generate_report
            generate_report()
            self.report_status.config(text="报告生成完毕 → models/report_*.png")
        except Exception as e:
            self.report_status.config(text=f"失败: {e}", foreground='red')

    def _run_export(self):
        self.export_status.config(text="正在导出...")
        self.root.update()
        try:
            import subprocess
            subprocess.run([sys.executable, 'export_tflite.py'], check=True, timeout=120)
            self.export_status.config(text="导出完毕 → models/*.tflite")
        except Exception as e:
            self.export_status.config(text=f"失败: {e}", foreground='red')

    def on_close(self):
        self.cam_running = False
        self._worker_stop.set()
        if self._display_timer:
            self.root.after_cancel(self._display_timer)
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = MaskDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
