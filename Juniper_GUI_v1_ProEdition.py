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
import queue
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

import GratingMotor4 as gm

# ---------------- Acquisition worker ----------------
class ADCWorker(QtCore.QObject):
    """Legacy streaming worker retained for backwards compatibility."""
    sampleReady = QtCore.pyqtSignal(float, float)
    error = QtCore.pyqtSignal(str)
    stopped = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__()


class StepScanWorker:
    """Perform a step-scan: move the grating and acquire MCC128 samples."""

    def __init__(self, address: int, channel: int, mode: AnalogInputMode,
                 rng: AnalogInputRange, positions, settle_s: float = 0.25,
                 samples_per_step: int = 1, motor_rpm: float = 6.0):
        self._address = address
        self._channel = channel
        self._mode = mode
        self._range = rng
        self._positions = list(positions)
        self._settle_s = max(0.0, float(settle_s))
        self._samples_per_step = max(1, int(samples_per_step))
        self._motor_rpm = float(motor_rpm)

        self.queue = queue.Queue()
        self._ser = None
        self._hat = None
        self._thread = None
        self._run_flag = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._run_flag.set()
        self._thread = threading.Thread(target=self._run, name="StepScanWorker", daemon=True)
        self._thread.start()

    def stop(self):
        self._run_flag.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        current_abs = gm.read_software_abs()
        try:
            self._ser = gm.open_port()
            gm.stop(self._ser)
            gm.set_motor_type(self._ser)
            gm.set_velocity(self._ser, self._motor_rpm)
        except Exception as exc:
            self.queue.put(("error", f"Motor init failed: {exc}"))
            self._cleanup()
            return

        try:
            self._hat = mcc128(self._address)
            self._hat.a_in_mode_write(self._mode)
            self._hat.a_in_range_write(self._range)
        except HatError as exc:
            self.queue.put(("error", f"DAQ init failed: {exc}"))
            self._cleanup()
            return

        try:
            for idx, target in enumerate(self._positions):
                if not self._run_flag.is_set():
                    break

                current_abs = gm.destination(self._ser, current_abs, int(target))
                gm.write_software_abs(current_abs)

                if self._settle_s:
                    time.sleep(self._settle_s)

                total = 0.0
                count = 0
                reading = float("nan")
                for _ in range(self._samples_per_step):
                    if not self._run_flag.is_set():
                        break
                    try:
                        reading = float(self._hat.a_in_read(self._channel, OptionFlags.DEFAULT))
                    except HatError as exc:
                        self.queue.put(("error", f"Read failed at step {idx}: {exc}"))
                        reading = float("nan")
                    total += reading
                    count += 1
                    if self._samples_per_step > 1:
                        time.sleep(0.01)

                if count:
                    value = total / count
                    self.queue.put(("data", current_abs, value))

            self.queue.put(("finished", None))
        finally:
            self._cleanup()

    def _cleanup(self):
        try:
            if self._ser is not None:
                try:
                    gm.stop(self._ser)
                except Exception:
                    pass
                self._ser.close()
        finally:
            self._ser = None

        if self._hat is not None:
            self._hat = None


# ---------------- Matplotlib helpers ----------------
def style_axes_pro(ax, window_s):
    # Dark theme and styling
    ax.set_facecolor("#2f2f2f")  # dark gray
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.grid(True, color='white', alpha=0.15, linestyle='--', linewidth=0.7)
    ax.set_xlabel("Grating position (pulses)", color="white")
    ax.set_ylabel("Signal (V)", color="white")
    ax.set_xlim(0.0, max(1.0, float(window_s)))

# ---------------- Main window ----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCC128 Live Analyzer (Pro Edition)")
        self.resize(1100, 700)

        # Runtime holders
        self.address = None
        self.worker = None

        # Spectrum buffers
        self.window_s = 10.0
        self.sample_hz = 500.0
        self.spectrum_positions = []
        self.spectrum_values = []
        self._plot_dirty = False

        self.display_paused = False

        # Build UI and theme
        self._apply_dark_palette()
        self._build_ui()
        self._connect()
        self.find_device()

        # Worker polling
        self.queue_timer = QtCore.QTimer(self)
        self.queue_timer.setInterval(50)
        self.queue_timer.timeout.connect(self._poll_worker_queue)

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
        self.setStyleSheet(
            "QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {"
            "color: black;"
            "background-color: #f0f0f0;"
            "selection-color: white;"
            "selection-background-color: #3874f2;"
            "border-radius: 4px;"
            "padding: 2px;"
            "}"
            "QComboBox QAbstractItemView {color: black; background-color: white;}"
        )

    # ---------- UI ----------
    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QVBoxLayout(); vbox.addLayout(controls)

        # Row 1: command buttons + status
        row1 = QtWidgets.QHBoxLayout(); controls.addLayout(row1)
        self.btnDetect = QtWidgets.QPushButton("Detect MCC128")
        self.btnStart  = QtWidgets.QPushButton("Start")
        self.btnStop   = QtWidgets.QPushButton("Stop"); self.btnStop.setEnabled(False)
        self.btnPause  = QtWidgets.QPushButton("Pause Display"); self.btnPause.setCheckable(True)
        self.btnSave   = QtWidgets.QPushButton("Save CSV…")

        self.btnStart.setStyleSheet("QPushButton{background-color:#006400;color:white;border-radius:4px;padding:6px;} QPushButton:disabled{background-color:#2a2a2a;color:#777;}")
        self.btnStop.setStyleSheet("QPushButton{background-color:#8B0000;color:white;border-radius:4px;padding:6px;} QPushButton:disabled{background-color:#2a2a2a;color:#777;}")

        for btn in (self.btnDetect, self.btnStart, self.btnStop, self.btnPause, self.btnSave):
            btn.setMinimumWidth(110)

        row1.addWidget(self.btnDetect)
        row1.addWidget(self.btnStart)
        row1.addWidget(self.btnStop)
        row1.addWidget(self.btnPause)
        row1.addWidget(self.btnSave)
        row1.addStretch(1)
        self.lblStatus = QtWidgets.QLabel("Status: idle")
        row1.addWidget(self.lblStatus, alignment=Qt.AlignRight)

        # Row 2: ADC controls
        row2 = QtWidgets.QHBoxLayout(); controls.addLayout(row2)
        row2.addWidget(QtWidgets.QLabel("ADC Mode:"))
        self.cmbMode = QtWidgets.QComboBox(); self.cmbMode.addItems(["SE", "DIFF"])
        row2.addWidget(self.cmbMode)
        row2.addSpacing(6)
        row2.addWidget(QtWidgets.QLabel("Range:"))
        self.cmbRange = QtWidgets.QComboBox(); self.cmbRange.addItems(["±10V", "±5V", "±2V", "±1V"])
        row2.addWidget(self.cmbRange)
        row2.addSpacing(6)
        row2.addWidget(QtWidgets.QLabel("Channel:"))
        self.spnChan = QtWidgets.QSpinBox(); self.spnChan.setRange(0, 7); self.spnChan.setValue(0)
        row2.addWidget(self.spnChan)
        row2.addSpacing(6)
        row2.addWidget(QtWidgets.QLabel("Rate (Hz):"))
        self.spnRate = QtWidgets.QDoubleSpinBox(); self.spnRate.setRange(1, 5000); self.spnRate.setDecimals(0); self.spnRate.setValue(self.sample_hz)
        row2.addWidget(self.spnRate)
        row2.addSpacing(6)
        row2.addWidget(QtWidgets.QLabel("Window (s):"))
        self.spnWin = QtWidgets.QDoubleSpinBox(); self.spnWin.setRange(1, 60); self.spnWin.setDecimals(0); self.spnWin.setValue(self.window_s)
        row2.addWidget(self.spnWin)
        row2.addSpacing(6)
        self.chkAutorng = QtWidgets.QCheckBox("Auto-range")
        controls.addWidget(self.chkAutorng)

        controls.addWidget(QtWidgets.QLabel("Start abs:"))
        self.spnStartAbs = QtWidgets.QSpinBox(); self.spnStartAbs.setRange(-500000, 500000); self.spnStartAbs.setValue(0); controls.addWidget(self.spnStartAbs)
        controls.addWidget(QtWidgets.QLabel("Step pulses:"))
        self.spnStep = QtWidgets.QSpinBox(); self.spnStep.setRange(-20000, 20000); self.spnStep.setValue(500); controls.addWidget(self.spnStep)
        controls.addWidget(QtWidgets.QLabel("Steps:"))
        self.spnSteps = QtWidgets.QSpinBox(); self.spnSteps.setRange(1, 5000); self.spnSteps.setValue(50); controls.addWidget(self.spnSteps)
        controls.addWidget(QtWidgets.QLabel("Settle (s):"))
        self.spnSettle = QtWidgets.QDoubleSpinBox(); self.spnSettle.setRange(0.0, 5.0); self.spnSettle.setDecimals(2); self.spnSettle.setSingleStep(0.05); self.spnSettle.setValue(0.25); controls.addWidget(self.spnSettle)
        controls.addWidget(QtWidgets.QLabel("Avg samples:"))
        self.spnSamples = QtWidgets.QSpinBox(); self.spnSamples.setRange(1, 32); self.spnSamples.setValue(1); controls.addWidget(self.spnSamples)
        controls.addWidget(QtWidgets.QLabel("Motor RPM:"))
        self.spnMotorRPM = QtWidgets.QDoubleSpinBox(); self.spnMotorRPM.setRange(0.1, 30.0); self.spnMotorRPM.setDecimals(1); self.spnMotorRPM.setValue(6.0); controls.addWidget(self.spnMotorRPM)

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
    def _reset_spectrum(self):
        self.spectrum_positions = []
        self.spectrum_values = []
        self._plot_dirty = True
        self.line_live.set_data([], [])
        self.text_overlay.set_text("")
        self.canvas_live.draw_idle()

    def _apply_rate(self):
        self.sample_hz = float(self.spnRate.value())
        if self.worker and hasattr(self.worker, "set_rate"):
            self.worker.set_rate(self.sample_hz)

    def _apply_window(self):
        self.window_s = float(self.spnWin.value())
        self.canvas_live.draw_idle()

    def _apply_channel(self):
        ch = int(self.spnChan.value())
        if self.cmbMode.currentText() == "DIFF" and ch > 3:
            ch = 3
            self.spnChan.setValue(3)
        if self.worker and hasattr(self.worker, "set_channel"):
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
        if self.worker and hasattr(self.worker, "reconfigure"):
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
        start_abs = int(self.spnStartAbs.value())
        step = int(self.spnStep.value())
        steps = int(self.spnSteps.value())
        settle = float(self.spnSettle.value())
        samples_per_step = int(self.spnSamples.value())
        motor_rpm = float(self.spnMotorRPM.value())

        if steps <= 0 or step == 0:
            self.lblStatus.setText("Status: invalid step configuration")
            return

        positions = [start_abs + i * step for i in range(steps)]

        self._reset_spectrum()

        self.worker = StepScanWorker(self.address, ch, mode, rng, positions,
                                     settle_s=settle,
                                     samples_per_step=samples_per_step,
                                     motor_rpm=motor_rpm)
        self.worker.start()

        self.queue_timer.start()
        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.lblStatus.setText(f"Status: scanning {steps} steps from {start_abs}")

    def stop_acq(self):
        if self.worker:
            self.worker.stop()
            self._flush_worker_queue()
            if self.worker:
                self.worker = None
        self.queue_timer.stop()
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        if not self.lblStatus.text().startswith("Error"):
            self.lblStatus.setText("Status: stopped")

    # ---------- Actions ----------
    def _toggle_pause(self, checked: bool):
        self.display_paused = checked
        self.btnPause.setText("Resume Display" if checked else "Pause Display")

    def _save_csv(self):
        if len(self.spectrum_positions) == 0:
            return
        pos = np.asarray(self.spectrum_positions, dtype=float)
        val = np.asarray(self.spectrum_values, dtype=float)
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "capture.csv", "CSV Files (*.csv)")
        if fn:
            try:
                arr = np.column_stack([pos, val])
                np.savetxt(fn, arr, delimiter=",", header="position_pulses,voltage_V", comments="")
                self.lblStatus.setText(f"Saved: {Path(fn).name}")
            except Exception as e:
                self.lblStatus.setText(f"Save failed: {e}")

    # ---------- Plot updates ----------
    def _update_plot(self):
        if self.display_paused or not self._plot_dirty:
            return
        if not self.spectrum_positions:
            self._plot_dirty = False
            return

        pos = np.asarray(self.spectrum_positions, dtype=float)
        val = np.asarray(self.spectrum_values, dtype=float)
        if pos.size == 0:
            self._plot_dirty = False
            return

        self.line_live.set_data(pos, val)

        xmin = float(np.nanmin(pos)) if pos.size else 0.0
        xmax = float(np.nanmax(pos)) if pos.size else 1.0
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            xmin, xmax = 0.0, 1.0
        span = xmax - xmin
        if span <= 0:
            xmin -= 0.5
            xmax += 0.5
        else:
            pad = 0.05 * span
            xmin -= pad
            xmax += pad
        self.ax_live.set_xlim(xmin, xmax)

        if self.chkAutorng.isChecked() and val.size:
            vmin = float(np.nanmin(val))
            vmax = float(np.nanmax(val))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                pad = 0.05 * (vmax - vmin + 1e-9)
                self.ax_live.set_ylim(vmin - pad, vmax + pad)

        if val.size:
            rms = float(np.sqrt(np.nanmean(val**2)))
            self.text_overlay.set_text(
                f"Points: {val.size}\nLast: {val[-1]:.3f} V\nRMS: {rms:.3f} V"
            )
        else:
            self.text_overlay.set_text("Points: 0")

        self.canvas_live.draw_idle()
        self._plot_dirty = False

    def _update_fft(self):
        if not self.spectrum_values:
            return
        val = np.asarray(self.spectrum_values, dtype=float)
        if val.size < 4:
            return
        v = val - np.nanmean(val)
        win = np.hanning(v.size)
        V = np.fft.rfft(v * win)
        mag = np.abs(V)
        freqs = np.fft.rfftfreq(v.size, d=1.0)
        self.line_fft.set_data(freqs, mag)
        if freqs.size:
            self.ax_fft.set_xlim(0, freqs.max())
        if np.all(np.isfinite(mag)) and mag.size:
            ymax = float(np.nanmax(mag))
            self.ax_fft.set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)
        self.canvas_fft.draw_idle()

    def _poll_worker_queue(self):
        worker = self.worker
        if worker is None:
            self.queue_timer.stop()
            return

        try:
            while True:
                item = worker.queue.get_nowait()
                self._handle_worker_message(item, worker)
        except queue.Empty:
            pass

        if self._plot_dirty:
            self._update_plot()

    def _flush_worker_queue(self):
        worker = self.worker
        if worker is None:
            return
        try:
            while True:
                item = worker.queue.get_nowait()
                self._handle_worker_message(item, worker)
        except queue.Empty:
            pass

        if self._plot_dirty:
            self._update_plot()

    def _handle_worker_message(self, item, worker_ref):
        kind = item[0]
        if kind == "data":
            _, pos, val = item
            self.spectrum_positions.append(pos)
            self.spectrum_values.append(val)
            self._plot_dirty = True
            if len(self.spectrum_values) >= 4:
                self._update_fft()
        elif kind == "error":
            msg = item[1] if len(item) > 1 else "unknown"
            self.lblStatus.setText(f"Error: {msg}")
            self._complete_scan(worker_ref)
        elif kind == "finished":
            self._complete_scan(worker_ref)

    def _complete_scan(self, worker_ref):
        if worker_ref is not None:
            try:
                worker_ref.stop()
            except Exception:
                pass
        if self.worker is worker_ref:
            self.worker = None
        self.queue_timer.stop()
        if not self.lblStatus.text().startswith("Error"):
            self.lblStatus.setText("Status: scan complete")
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        if self._plot_dirty:
            self._update_plot()

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
