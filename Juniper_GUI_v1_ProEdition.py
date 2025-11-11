#!/usr/bin/env python3
"""
MCC128 Pro Look GUI (PyQt5)
- Dark theme across widgets + plot
- Thick white frame, subtle grid, yellow trace
- Start/Stop styled buttons
- Pause/Resume display (acquisition continues)
- Auto-range toggle
- Save CSV button
- Live RMS/peak overlay
- FFT tab (updates ~1 Hz) with Hann window

Install (Raspberry Pi OS):
  sudo apt update && sudo apt install -y python3-pyqt5 python3-matplotlib python3-daqhats mcc-daqhats

If using virtualenv (keep daqhats via apt):
  python3 -m venv ~/venv && source ~/venv/bin/activate
  pip install PyQt5 matplotlib numpy
"""
import sys
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt

from daqhats import (
    mcc128, HatIDs, hat_list, HatError,
    AnalogInputMode, AnalogInputRange, OptionFlags
)

# ---------------- Acquisition worker ----------------
class ADCWorker(QtCore.QObject):
    sampleReady = QtCore.pyqtSignal(float, float)  # (t_rel_s, volts)
    error = QtCore.pyqtSignal(str)
    stopped = QtCore.pyqtSignal()

    def __init__(self, address: int, channel: int, mode: AnalogInputMode,
                 rng: AnalogInputRange, sample_hz: float, parent=None):
        super().__init__(parent)
        self._address = address
        self._channel = channel
        self._mode = mode
        self._range = rng
        self._sample_hz = max(1.0, float(sample_hz))
        self._running = False
        self._lock = threading.Lock()
        self._hat = None

    @QtCore.pyqtSlot()
    def start(self):
        if self._running:
            return
        self._running = True
        try:
            self._hat = mcc128(self._address)
            self._hat.a_in_mode_write(self._mode)
            self._hat.a_in_range_write(self._range)
        except HatError as e:
            self.error.emit(f"DAQ init error: {e}")
            self._running = False
            self.stopped.emit()
            return

        dt = 1.0 / self._sample_hz
        t0 = time.perf_counter()
        next_t = t0

        try:
            while self._running:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(next_t - now)
                next_t += dt
                try:
                    v = self._hat.a_in_read(self._channel, OptionFlags.DEFAULT)
                except HatError as e:
                    self.error.emit(f"Read error: {e}")
                    v = float('nan')
                self.sampleReady.emit(time.perf_counter() - t0, float(v))
        finally:
            self.stopped.emit()

    @QtCore.pyqtSlot()
    def stop(self):
        self._running = False

    # Live setters
    def set_channel(self, ch: int):
        with self._lock:
            self._channel = int(ch)

    def set_rate(self, hz: float):
        with self._lock:
            self._sample_hz = max(1.0, float(hz))

    def reconfigure(self, mode: AnalogInputMode, rng: AnalogInputRange):
        try:
            if self._hat is not None:
                self._hat.a_in_mode_write(mode)
                self._hat.a_in_range_write(rng)
            self._mode = mode
            self._range = rng
        except Exception:
            pass

# ---------------- Matplotlib helpers ----------------
def style_axes_pro(ax, window_s):
    # Dark theme and styling
    ax.set_facecolor("#2f2f2f")  # dark gray
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.grid(True, color='white', alpha=0.15, linestyle='--', linewidth=0.7)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Voltage (V)", color="white")
    ax.set_xlim(-window_s, 0.0)

# ---------------- Main window ----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCC128 Live Analyzer (Pro Edition)")
        self.resize(1100, 700)

        # Runtime holders
        self.address = None
        self.worker = None
        self.thread = None

        # Buffers
        self.window_s = 10.0
        self.sample_hz = 500.0
        self._alloc_buffers()

        self.display_paused = False

        # Build UI and theme
        self._apply_dark_palette()
        self._build_ui()
        self._connect()
        self.find_device()

        # UI refresh timers
        self.timer_plot = QtCore.QTimer(self)
        self.timer_plot.setInterval(50)  # ~20 FPS
        self.timer_plot.timeout.connect(self._update_plot)

        self.timer_fft = QtCore.QTimer(self)
        self.timer_fft.setInterval(1000) # 1 Hz FFT updates
        self.timer_fft.timeout.connect(self._update_fft)

    # ---------- Theme ----------
    def _apply_dark_palette(self):
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#2f2f2f"))
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#1e1e1e"))
        pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#2a2a2a"))
        pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#2f2f2f"))
        pal.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("white"))
        pal.setColor(QtGui.QPalette.Text, QtGui.QColor("white"))
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#3a3a3a"))
        pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("white"))
        pal.setColor(QtGui.QPalette.BrightText, QtGui.QColor("#FFD300"))
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#003366"))
        pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("white"))
        self.setPalette(pal)

    # ---------- UI ----------
    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # Top controls / toolbar row
        controls = QtWidgets.QHBoxLayout(); vbox.addLayout(controls)

        self.btnDetect = QtWidgets.QPushButton("Detect MCC128")
        self.btnStart  = QtWidgets.QPushButton("Start")
        self.btnStop   = QtWidgets.QPushButton("Stop"); self.btnStop.setEnabled(False)
        self.btnPause  = QtWidgets.QPushButton("Pause Display"); self.btnPause.setCheckable(True)
        self.btnSave   = QtWidgets.QPushButton("Save CSV…")

        # Styled buttons
        self.btnStart.setStyleSheet("QPushButton{background-color:#006400;color:white;border-radius:4px;padding:6px;} QPushButton:disabled{background-color:#2a2a2a;color:#777;}")
        self.btnStop.setStyleSheet("QPushButton{background-color:#8B0000;color:white;border-radius:4px;padding:6px;} QPushButton:disabled{background-color:#2a2a2a;color:#777;}")

        controls.addWidget(self.btnDetect)
        controls.addWidget(self.btnStart)
        controls.addWidget(self.btnStop)
        controls.addWidget(self.btnPause)
        controls.addWidget(self.btnSave)
        controls.addSpacing(12)

        # Settings cluster
        controls.addWidget(QtWidgets.QLabel("Mode:"))
        self.cmbMode = QtWidgets.QComboBox(); self.cmbMode.addItems(["SE","DIFF"]) ; controls.addWidget(self.cmbMode)
        controls.addWidget(QtWidgets.QLabel("Range:"))
        self.cmbRange = QtWidgets.QComboBox(); self.cmbRange.addItems(["±10V","±5V","±2V","±1V"]) ; controls.addWidget(self.cmbRange)
        controls.addWidget(QtWidgets.QLabel("Channel:"))
        self.spnChan = QtWidgets.QSpinBox(); self.spnChan.setRange(0,7); self.spnChan.setValue(0); controls.addWidget(self.spnChan)
        controls.addWidget(QtWidgets.QLabel("Rate (Hz):"))
        self.spnRate = QtWidgets.QDoubleSpinBox(); self.spnRate.setRange(1, 5000); self.spnRate.setDecimals(0); self.spnRate.setValue(self.sample_hz); controls.addWidget(self.spnRate)
        controls.addWidget(QtWidgets.QLabel("Window (s):"))
        self.spnWin = QtWidgets.QDoubleSpinBox(); self.spnWin.setRange(1, 60); self.spnWin.setDecimals(0); self.spnWin.setValue(self.window_s); controls.addWidget(self.spnWin)
        self.chkAutorng = QtWidgets.QCheckBox("Auto-range")
        controls.addWidget(self.chkAutorng)

        controls.addStretch(1)
        self.lblStatus = QtWidgets.QLabel("Status: idle"); controls.addWidget(self.lblStatus)

        # Tabs for Live and FFT
        self.tabs = QtWidgets.QTabWidget(); vbox.addWidget(self.tabs, 1)

        # --- Live tab ---
        live = QtWidgets.QWidget(); self.tabs.addTab(live, "Live Plot")
        lv = QtWidgets.QVBoxLayout(live)
        self.fig_live = Figure(figsize=(6,4), facecolor="#2f2f2f")
        self.ax_live = self.fig_live.add_subplot(111)
        self.canvas_live = FigureCanvas(self.fig_live)
        lv.addWidget(self.canvas_live, 1)

        style_axes_pro(self.ax_live, self.window_s)
        (self.line_live,) = self.ax_live.plot([], [], lw=1.6, color="#FFD300")  # yellow
        # Lab watermark
        self.ax_live.text(0.98, 0.02, "Juniper", color="#CCCCCC", fontsize=8,
                          transform=self.ax_live.transAxes, ha="right", va="bottom")
        # RMS/peaks overlay
        self.text_overlay = self.ax_live.text(0.02, 0.96, "", color="white",
                                              transform=self.ax_live.transAxes, ha="left", va="top", fontsize=9)

        # --- FFT tab ---
        fftw = QtWidgets.QWidget(); self.tabs.addTab(fftw, "FFT")
        fv = QtWidgets.QVBoxLayout(fftw)
        self.fig_fft = Figure(figsize=(6,4), facecolor="#2f2f2f")
        self.ax_fft = self.fig_fft.add_subplot(111)
        self.canvas_fft = FigureCanvas(self.fig_fft)
        fv.addWidget(self.canvas_fft, 1)
        # style
        self.ax_fft.set_facecolor("#2f2f2f")
        for s in self.ax_fft.spines.values():
            s.set_linewidth(2.0); s.set_color("white")
        self.ax_fft.tick_params(colors="white")
        self.ax_fft.grid(True, color='white', alpha=0.15, linestyle='--', linewidth=0.7)
        self.ax_fft.set_xlabel("Frequency (Hz)", color="white")
        self.ax_fft.set_ylabel("|V| (RMS)", color="white")
        (self.line_fft,) = self.ax_fft.plot([], [], lw=1.6, color="#00FFFF")  # cyan

    def _connect(self):
        self.btnDetect.clicked.connect(self.find_device)
        self.btnStart.clicked.connect(self.start_acq)
        self.btnStop.clicked.connect(self.stop_acq)
        self.btnPause.clicked.connect(self._toggle_pause)
        self.btnSave.clicked.connect(self._save_csv)
        self.spnRate.valueChanged.connect(self._apply_rate)
        self.spnWin.valueChanged.connect(self._apply_window)
        self.cmbMode.currentIndexChanged.connect(self._apply_mode_range)
        self.cmbRange.currentIndexChanged.connect(self._apply_mode_range)
        self.spnChan.valueChanged.connect(self._apply_channel)

    # ---------- Helpers ----------
    def _alloc_buffers(self):
        N = max(2, int(self.window_s * self.sample_hz))
        self.tbuf = deque([0.0]*N, maxlen=N)
        self.vbuf = deque([0.0]*N, maxlen=N)

    def _apply_rate(self):
        self.sample_hz = float(self.spnRate.value())
        self._alloc_buffers()
        if self.worker:
            self.worker.set_rate(self.sample_hz)

    def _apply_window(self):
        self.window_s = float(self.spnWin.value())
        self._alloc_buffers()
        self.ax_live.set_xlim(-self.window_s, 0.0)
        self.canvas_live.draw_idle()

    def _apply_channel(self):
        ch = int(self.spnChan.value())
        if self.cmbMode.currentText() == "DIFF" and ch > 3:
            ch = 3
            self.spnChan.setValue(3)
        if self.worker:
            self.worker.set_channel(ch)

    def _apply_mode_range(self):
        if not self.worker:
            return
        mode = AnalogInputMode.SE if self.cmbMode.currentText()=="SE" else AnalogInputMode.DIFF
        rng_map = {"±10V": AnalogInputRange.BIP_10V, "±5V": AnalogInputRange.BIP_5V,
                   "±2V": AnalogInputRange.BIP_2V, "±1V": AnalogInputRange.BIP_1V}
        rng = rng_map[self.cmbRange.currentText()]
        ylim_map = {AnalogInputRange.BIP_10V: (-10, 10), AnalogInputRange.BIP_5V: (-5,5),
                    AnalogInputRange.BIP_2V: (-2,2), AnalogInputRange.BIP_1V: (-1,1)}
        self.ax_live.set_ylim(*ylim_map[rng])
        self.canvas_live.draw_idle()
        self.worker.reconfigure(mode, rng)
        self._apply_channel()

    def find_device(self):
        hats = hat_list(HatIDs.MCC_128)
        if not hats:
            self.address = None
            self.lblStatus.setText("Status: no MCC128 found")
        else:
            self.address = hats[0].address
            self.lblStatus.setText(f"Status: MCC128 addr {self.address} ready")

    # ---------- Start/Stop ----------
    def start_acq(self):
        if self.address is None:
            self.find_device()
            if self.address is None:
                return
        mode = AnalogInputMode.SE if self.cmbMode.currentText()=="SE" else AnalogInputMode.DIFF
        rng_map = {"±10V": AnalogInputRange.BIP_10V, "±5V": AnalogInputRange.BIP_5V,
                   "±2V": AnalogInputRange.BIP_2V, "±1V": AnalogInputRange.BIP_1V}
        rng = rng_map[self.cmbRange.currentText()]
        ch = int(self.spnChan.value())
        if mode == AnalogInputMode.DIFF and ch > 3:
            ch = 3
            self.spnChan.setValue(3)

        self._alloc_buffers()

        self.thread = QtCore.QThread(self)
        self.worker = ADCWorker(self.address, ch, mode, rng, self.sample_hz)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.sampleReady.connect(self.on_sample)
        self.worker.error.connect(self.on_error)
        self.worker.stopped.connect(self.on_stopped)
        self.thread.start()

        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.lblStatus.setText(f"Status: running (ch{ch}, {self.sample_hz:.0f} Hz)")
        self.timer_plot.start(); self.timer_fft.start()

    def stop_acq(self):
        self.timer_plot.stop(); self.timer_fft.stop()
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit(); self.thread.wait(1000)
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.lblStatus.setText("Status: stopped")

    @QtCore.pyqtSlot()
    def on_stopped(self):
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.lblStatus.setText("Status: stopped")

    @QtCore.pyqtSlot(str)
    def on_error(self, msg: str):
        self.lblStatus.setText(f"Error: {msg}")

    @QtCore.pyqtSlot(float, float)
    def on_sample(self, t_rel: float, v: float):
        # Always record samples; pause only affects display
        self.tbuf.append(t_rel)
        self.vbuf.append(v)

    # ---------- Actions ----------
    def _toggle_pause(self, checked: bool):
        self.display_paused = checked
        self.btnPause.setText("Resume Display" if checked else "Pause Display")

    def _save_csv(self):
        if len(self.tbuf) < 2:
            return
        t = np.asarray(self.tbuf, dtype=float)
        v = np.asarray(self.vbuf, dtype=float)
        t = t - t[0]
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "capture.csv", "CSV Files (*.csv)")
        if fn:
            try:
                arr = np.column_stack([t, v])
                np.savetxt(fn, arr, delimiter=",", header="time_s,voltage_V", comments="")
                self.lblStatus.setText(f"Saved: {Path(fn).name}")
            except Exception as e:
                self.lblStatus.setText(f"Save failed: {e}")

    # ---------- Plot updates ----------
    def _update_plot(self):
        if self.display_paused or not self.tbuf:
            return
        t = np.asarray(self.tbuf, dtype=float)
        v = np.asarray(self.vbuf, dtype=float)
        if t.size < 2:
            return
        t = t - t[-1]  # slide window to end at 0
        self.line_live.set_data(t, v)
        if self.chkAutorng.isChecked():
            vmin = float(np.nanmin(v)); vmax = float(np.nanmax(v))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                pad = 0.05*(vmax - vmin + 1e-9)
                self.ax_live.set_ylim(vmin - pad, vmax + pad)
        # overlay stats
        rms = float(np.sqrt(np.nanmean(v**2)))
        p2p = float(np.nanmax(v) - np.nanmin(v)) if v.size else 0.0
        self.text_overlay.set_text(f"RMS: {rms:.3f} V\nP-P: {p2p:.3f} V")
        self.ax_live.set_xlim(-self.window_s, 0.0)
        self.canvas_live.draw_idle()

    def _update_fft(self):
        if not self.tbuf:
            return
        t = np.asarray(self.tbuf, dtype=float)
        v = np.asarray(self.vbuf, dtype=float)
        if t.size < 8:
            return
        # convert to uniform grid (best-effort) in case of slight jitter
        dt_est = max(1e-9, np.median(np.diff(t)))
        N = int(min(len(v), self.window_s * self.sample_hz))
        # use last N samples
        tseg = t[-N:]
        vseg = v[-N:]
        # resample to uniform grid with linear interp (optional)
        tu = np.linspace(tseg[0], tseg[-1], N)
        vu = np.interp(tu, tseg, vseg)
        # window + rFFT
        win = np.hanning(N)
        vu_w = vu * win
        V = np.fft.rfft(vu_w)
        # amplitude scaling to RMS-like units (approx):
        # For visualization: magnitude divided by sqrt(2*N) to be modestly scale-stable
        mag = np.abs(V) / np.sqrt(2*N)
        freqs = np.fft.rfftfreq(N, d=dt_est)
        self.line_fft.set_data(freqs, mag)
        # set limits nicely
        self.ax_fft.set_xlim(0, freqs.max())
        # autoscale y each update
        if np.all(np.isfinite(mag)) and mag.size:
            ymax = float(np.nanmax(mag))
            self.ax_fft.set_ylim(0, ymax*1.1 if ymax>0 else 1)
        self.canvas_fft.draw_idle()

    def closeEvent(self, event):
        try:
            self.stop_acq()
        finally:
            super().closeEvent(event)

# ---------------- Main ----------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec_())
