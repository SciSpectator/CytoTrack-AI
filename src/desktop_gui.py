"""
CytoTrack AI - Native Desktop GUI (PyQt5)
==========================================
Professional desktop application replacing the former pygame UI.
Light theme, native window chrome, menu integration, real file dialogs.

Public API (kept compatible with the legacy fancy_gui module):
    class FancyGUI
        .running: bool
        .screen: QMainWindow           (parent for dialogs / preview windows)
        show_main_menu() -> str | None
        show_message(title, body, buttons) -> str | None
        show_input_dialog(title, prompts, defaults) -> list[str] | None
        show_progress(title, total)     -> callable(step, msg)
        show_folder_dialog(title)       -> str | None
        show_file_dialog(title, extensions=[...]) -> str | None
        cleanup() -> None

    show_splash_screen(duration=3) -> None
    show_image_settings_preview(parent, files, Settings) -> bool
    manual_cell_classification(parent, frame_bgr, detections, type_list) -> dict | None
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

# Import cv2 BEFORE Qt so we can strip its bundled Qt plugin hints. The
# opencv-python wheel writes QT_QPA_PLATFORM_PLUGIN_PATH at import time to
# point at its own Qt plugins, which on Linux frequently can't load
# `libqxcb.so` (missing transitive xcb deps). Removing the var here forces
# PyQt5 to fall back to the system Qt plugins that actually work.
import cv2  # noqa: E402 (must precede Qt imports)
import numpy as np

for _var in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
    _val = os.environ.get(_var, "")
    if "cv2" in _val or "site-packages" in _val:
        os.environ.pop(_var, None)
for _cand in (
    "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms",
    "/usr/lib64/qt5/plugins/platforms",
    "/usr/lib/qt5/plugins/platforms",
):
    if os.path.isdir(_cand):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _cand
        break

from PyQt5.QtCore import (QPoint, QRect, QSize, Qt, QTimer, pyqtSignal)
from PyQt5.QtGui import (QColor, QFont, QIcon, QImage, QPainter, QPen,
                         QPixmap, QBrush, QLinearGradient, QFontDatabase)
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QFileDialog, QFrame, QGraphicsDropShadowEffect,
                             QGridLayout, QHBoxLayout,
                             QInputDialog, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QProgressBar, QProgressDialog,
                             QPushButton, QScrollArea, QSizePolicy, QSlider,
                             QSpacerItem, QSplashScreen, QStyle,
                             QStyleOptionSlider, QVBoxLayout, QWidget)


def _apply_shadow(widget: QWidget, blur: int = 28, dy: int = 4,
                  alpha: int = 40) -> None:
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(blur)
    eff.setOffset(0, dy)
    eff.setColor(QColor(30, 90, 150, alpha))
    widget.setGraphicsEffect(eff)


# -------------------------------------------------------------- branding ----
APP_NAME = "CytoTrack AI"
APP_VERSION = "1.0"
APP_FULL = f"{APP_NAME} v{APP_VERSION}"

# ----- Frutiger Aero palette: glossy white + sky-blue + nature-green -----
COLOR_BG = "#FFFFFF"
COLOR_BG_TOP = "#EAF6FE"       # sky gradient top
COLOR_BG_BOT = "#F4FBF2"       # nature gradient bottom
COLOR_PANEL = "#FFFFFF"
COLOR_PANEL_TOP = "#FCFEFF"    # glossy highlight
COLOR_PANEL_BOT = "#EDF7FF"    # subtle reflected-sky base
COLOR_BORDER = "#C5DAEA"
COLOR_BORDER_SOFT = "#E0EEF7"
COLOR_TEXT = "#0E2A45"
COLOR_MUTED = "#5F7D95"

# Sky blue (dominant)
COLOR_ACCENT = "#1E90E0"
COLOR_ACCENT_DARK = "#0A5B9A"
COLOR_ACCENT_LIGHT = "#B9E3FA"
COLOR_SKY_TOP = "#6DC8F3"
COLOR_SKY_BOT = "#2B8BD6"

# Nature green (secondary)
COLOR_GREEN = "#4CAF50"
COLOR_GREEN_DARK = "#2E7D32"
COLOR_GREEN_LIGHT = "#C9EFC7"
COLOR_LEAF_TOP = "#8FD98F"
COLOR_LEAF_BOT = "#3FAA45"

COLOR_SUCCESS = COLOR_GREEN_DARK
COLOR_DANGER = "#C0392B"
COLOR_GLASS_HILITE = "rgba(255, 255, 255, 0.85)"

_ICON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "assets", "icon.png")


# ----------------------------------------------------------- stylesheet -----
# Frutiger Aero inspired: glossy white surfaces, sky-blue accents,
# nature-green secondary, soft gradients, subtle glass highlights.
_GLOBAL_STYLE = f"""
* {{
    font-family: "Segoe UI", "Inter", "Ubuntu", "Helvetica Neue", sans-serif;
    color: {COLOR_TEXT};
}}
QMainWindow, QDialog {{
    background-color: {COLOR_BG};
}}
QWidget#aeroBackdrop {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLOR_BG_TOP},
        stop:0.55 #FFFFFF,
        stop:1 {COLOR_BG_BOT});
}}
QLabel#title {{
    font-size: 28px;
    font-weight: 300;
    color: {COLOR_TEXT};
    letter-spacing: 0.5px;
}}
QLabel#subtitle {{
    font-size: 13px;
    color: {COLOR_MUTED};
    font-weight: 400;
}}
QLabel#sectionHeader {{
    font-size: 12px;
    font-weight: 700;
    color: {COLOR_ACCENT_DARK};
    letter-spacing: 2px;
}}
QFrame#card {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLOR_PANEL_TOP},
        stop:0.45 {COLOR_PANEL},
        stop:1 {COLOR_PANEL_BOT});
    border: 1px solid {COLOR_BORDER};
    border-radius: 14px;
}}
QFrame#glassCard {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(255,255,255,240),
        stop:0.5 rgba(247,252,255,230),
        stop:1 rgba(226,243,252,230));
    border: 1px solid {COLOR_BORDER};
    border-radius: 16px;
}}
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFFFFF, stop:0.5 #F4FAFE, stop:1 #E2F0FB);
    border: 1px solid {COLOR_BORDER};
    border-radius: 10px;
    padding: 9px 18px;
    font-size: 13px;
    color: {COLOR_TEXT};
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFFFFF, stop:0.5 #E8F5FD, stop:1 #BFE1F6);
    border: 1px solid {COLOR_ACCENT};
}}
QPushButton:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #BFE1F6, stop:1 #9CD1F0);
}}
QPushButton#primary {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLOR_SKY_TOP},
        stop:0.5 {COLOR_ACCENT},
        stop:1 {COLOR_SKY_BOT});
    color: white;
    border: 1px solid {COLOR_ACCENT_DARK};
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #85D3F6, stop:0.5 #2EA0E9, stop:1 {COLOR_ACCENT_DARK});
}}
QPushButton#primary:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLOR_ACCENT_DARK}, stop:1 #073F6B);
}}
QPushButton#success {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLOR_LEAF_TOP},
        stop:0.5 {COLOR_GREEN},
        stop:1 {COLOR_LEAF_BOT});
    color: white;
    border: 1px solid {COLOR_GREEN_DARK};
    font-weight: 600;
}}
QPushButton#success:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #A0E2A0, stop:0.5 #5FC064, stop:1 {COLOR_GREEN_DARK});
}}
QPushButton#danger {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFFFFF, stop:1 #FFECEC);
    color: {COLOR_DANGER};
    border: 1px solid #F5B5B5;
}}
QPushButton#danger:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFE6E6, stop:1 #FFC9C9);
}}
QPushButton#menu {{
    text-align: left;
    padding: 18px 22px;
    font-size: 15px;
    font-weight: 500;
    border-radius: 14px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFFFFF,
        stop:0.45 #F6FBFF,
        stop:1 #E2F0FB);
    border: 1px solid {COLOR_BORDER};
    color: {COLOR_TEXT};
}}
QPushButton#menu:hover {{
    border: 1px solid {COLOR_ACCENT};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFFFFF,
        stop:0.45 #E5F4FC,
        stop:1 #BFE4F8);
}}
QPushButton#menu:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #D2EAF7, stop:1 #AEDAF1);
}}
QLineEdit, QComboBox {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #F8FCFF, stop:1 #FFFFFF);
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: {COLOR_ACCENT_LIGHT};
}}
QLineEdit:focus, QComboBox:focus {{
    border: 1px solid {COLOR_ACCENT};
    background: #FFFFFF;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #D4E7F5, stop:1 #EAF4FC);
    border: 1px solid {COLOR_BORDER_SOFT};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLOR_SKY_TOP}, stop:1 {COLOR_ACCENT});
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFFFFF, stop:0.5 #E6F3FC, stop:1 #A5D6F4);
    border: 2px solid {COLOR_ACCENT};
}}
QSlider::handle:horizontal:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFFFFF, stop:1 #6DC8F3);
}}
QProgressBar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #E5F1FA, stop:1 #F6FBFF);
    border: 1px solid {COLOR_BORDER_SOFT};
    border-radius: 8px;
    text-align: center;
    font-size: 11px;
    color: {COLOR_TEXT};
    min-height: 22px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLOR_LEAF_TOP},
        stop:0.5 {COLOR_GREEN},
        stop:1 {COLOR_GREEN_DARK});
    border-radius: 7px;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 4px;
}}
QScrollBar::handle:vertical {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #9EC7DF, stop:1 #6FAED0);
    border-radius: 5px;
    min-height: 36px;
}}
QScrollBar::handle:vertical:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLOR_ACCENT}, stop:1 {COLOR_ACCENT_DARK});
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QToolTip {{
    background: #FFFFFF;
    border: 1px solid {COLOR_ACCENT};
    color: {COLOR_TEXT};
    padding: 6px 10px;
    border-radius: 6px;
}}
"""


# ---------------------------------------------------------- helpers ---------
_APP_SINGLETON: Optional[QApplication] = None


def _qapp() -> QApplication:
    """Return the active QApplication (creating it if needed)."""
    global _APP_SINGLETON
    app = QApplication.instance()
    if app is None:
        # Pass a stable argv; Qt parses argv for its own options and will
        # sometimes choke on things like `-c` when launched via `python -c`.
        argv = [sys.argv[0] if sys.argv else "CytoTrackAI"]
        app = QApplication(argv)
        app.setApplicationName(APP_NAME)
        app.setOrganizationName("CytoTrack")
        if os.path.exists(_ICON_PATH):
            app.setWindowIcon(QIcon(_ICON_PATH))
        app.setStyleSheet(_GLOBAL_STYLE)
        _APP_SINGLETON = app  # keep a Python reference alive
    return app


def _np_bgr_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# ====================================================== logo widget =========
class LogoWidget(QLabel):
    """Shows the app icon at a fixed size (or a drawn fallback)."""

    def __init__(self, px: int = 56, parent=None):
        super().__init__(parent)
        self.setFixedSize(px, px)
        if os.path.exists(_ICON_PATH):
            pix = QPixmap(_ICON_PATH).scaled(
                px, px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(pix)
        else:
            self.setText("")


# ==================================================== main window ==========
class FancyGUI(QMainWindow):
    """Top-level application window."""

    MENU_ITEMS = [
        ("Track Cells", "Detect, track, and analyse cell migration",
         "TRACK"),
        ("Train Cell Line",
         "Public data search or local files for each requested line",
         "TRAIN_ONLINE"),
        ("Analyze Results", "Load a previous CSV and regenerate plots",
         "ANALYZE"),
        ("Help", "Workflow, outputs, and current feature set", "HELP"),
        ("Exit", "Close the application", "EXIT"),
    ]

    _menu_selection: Optional[str] = None

    def __init__(self):
        _qapp()
        super().__init__()
        self.running = True
        self.setWindowTitle(APP_FULL)
        if os.path.exists(_ICON_PATH):
            self.setWindowIcon(QIcon(_ICON_PATH))
        self.resize(1100, 720)
        self._build_ui()
        self.setStyleSheet(_GLOBAL_STYLE)

    # expose .screen for legacy callers (used as dialog parent)
    @property
    def screen(self) -> "FancyGUI":
        return self

    # ---------------------------------------------------------- ui setup ----
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("aeroBackdrop")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(36, 28, 36, 28)
        root.setSpacing(22)

        # ---- header ----
        header = QHBoxLayout()
        header.setSpacing(14)
        header.addWidget(LogoWidget(64))

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        t = QLabel(APP_NAME)
        t.setObjectName("title")
        st = QLabel(f"Version {APP_VERSION}   \u2022   Cell migration analysis suite")
        st.setObjectName("subtitle")
        title_col.addWidget(t)
        title_col.addWidget(st)
        header.addLayout(title_col)
        header.addStretch(1)

        # Hardware tier badge on the right
        header.addWidget(self._build_hw_badge())
        root.addLayout(header)

        # ---- main menu card ----
        card = QFrame()
        card.setObjectName("glassCard")
        _apply_shadow(card, blur=36, dy=6, alpha=55)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 28, 32, 28)
        card_layout.setSpacing(16)

        header_lbl = QLabel("CHOOSE AN ACTION")
        header_lbl.setObjectName("sectionHeader")
        card_layout.addWidget(header_lbl)

        self._menu_buttons: List[QPushButton] = []
        grid = QGridLayout()
        grid.setSpacing(14)
        for idx, (label, sub, key) in enumerate(self.MENU_ITEMS):
            btn = QPushButton(f"{label}\n{sub}")
            btn.setObjectName("menu")
            btn.setMinimumHeight(88)
            _apply_shadow(btn, blur=14, dy=2, alpha=30)
            btn.clicked.connect(lambda _=False, k=key: self._on_menu(k))
            grid.addWidget(btn, idx // 2, idx % 2)
            self._menu_buttons.append(btn)
        card_layout.addLayout(grid)

        root.addWidget(card)

        # ---- footer ----
        foot = QLabel(
            "SORT-style tracking  \u2022  Multi-strategy detector  "
            "\u2022  DSPy debris reasoning  \u2022  Lost-cell recovery")
        foot.setObjectName("subtitle")
        foot.setAlignment(Qt.AlignCenter)
        root.addWidget(foot)

    # ------------------------------------------------- hardware badge -----
    def _build_hw_badge(self) -> QWidget:
        try:
            from hardware_profile import detect_hardware
            hw = detect_hardware()
            tier_colors = {
                "low": "#B91C1C", "mid": "#CA8A04",
                "high": "#15803D", "extreme": "#0E7490",
            }
            col = tier_colors.get(hw.tier, COLOR_ACCENT)
            w = QFrame()
            w.setObjectName("card")
            w.setStyleSheet(
                f"QFrame#card {{ border:1px solid {col}; border-radius:10px; "
                f"background:#FBFEFF; }}")
            lay = QVBoxLayout(w)
            lay.setContentsMargins(12, 8, 12, 8)
            lay.setSpacing(2)
            t = QLabel(f"HARDWARE  \u2022  {hw.tier.upper()}")
            t.setStyleSheet(
                f"color:{col}; font-size:11px; font-weight:700; letter-spacing:1px;")
            lay.addWidget(t)
            gpu = (hw.gpu_name if hw.has_cuda else "CPU only")
            if hw.has_cuda:
                gpu += f"  \u2022  {hw.vram_gb:.1f} GB"
            s1 = QLabel(gpu)
            s1.setStyleSheet("font-size:12px; font-weight:600;")
            lay.addWidget(s1)
            s2 = QLabel(f"{hw.cpu_count} cores  \u2022  {hw.ram_gb:.1f} GB RAM")
            s2.setStyleSheet(f"color:{COLOR_MUTED}; font-size:11px;")
            lay.addWidget(s2)
            w.setToolTip(hw.long_description())
            return w
        except Exception as e:
            lbl = QLabel(f"HW probe failed: {e}")
            lbl.setStyleSheet(f"color:{COLOR_MUTED}; font-size:11px;")
            return lbl

    # ------------------------------------------------------- menu machinery
    def _on_menu(self, key: str) -> None:
        mapping = {
            "TRACK": "Track Cells",
            "TRAIN_ONLINE": "Train Cell Line",
            "ANALYZE": "Analyze Results",
            "HELP": "Help",
            "EXIT": "Exit",
        }
        self._menu_selection = mapping.get(key, key)
        # Stop the local event loop started in show_main_menu
        if hasattr(self, "_menu_loop") and self._menu_loop is not None:
            self._menu_loop.quit()

    def closeEvent(self, ev) -> None:
        self.running = False
        self._menu_selection = "Exit"
        if hasattr(self, "_menu_loop") and self._menu_loop is not None:
            self._menu_loop.quit()
        super().closeEvent(ev)

    # ============================================ public API (legacy) =====
    def show_main_menu(self) -> Optional[str]:
        from PyQt5.QtCore import QEventLoop
        self.show()
        self.raise_()
        self.activateWindow()
        self._menu_selection = None
        self._menu_loop = QEventLoop()
        self._menu_loop.exec_()
        self._menu_loop = None
        if not self.running:
            return None
        return self._menu_selection

    def show_message(self, title: str, body: str,
                     buttons: List[str]) -> Optional[str]:
        dlg = _MessageDialog(self, title, body, buttons)
        dlg.exec_()
        return dlg.result_button

    def show_input_dialog(self, title: str,
                          prompts: List[str],
                          defaults: List[str]) -> Optional[List[str]]:
        dlg = _InputDialog(self, title, prompts, defaults)
        if dlg.exec_() == QDialog.Accepted:
            return dlg.values
        return None

    def show_progress(self, title: str, total: int):
        dlg = _ProgressDialog(self, title, total)
        dlg.show()
        QApplication.processEvents()

        def update(step: int, msg: str = ""):
            dlg.update_progress(step, msg)
            QApplication.processEvents()
            if dlg.wasCanceled():
                raise RuntimeError("Cancelled by user")
        update._dialog = dlg  # keep a reference alive
        return update

    def show_folder_dialog(self, title: str) -> Optional[str]:
        folder = QFileDialog.getExistingDirectory(
            self, title, os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        return folder or None

    def show_file_dialog(self, title: str,
                         extensions: Optional[List[str]] = None) -> Optional[str]:
        filt = "All files (*.*)"
        if extensions:
            exts = " ".join(f"*{e}" for e in extensions)
            filt = f"Supported ({exts});;All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(
            self, title, os.path.expanduser("~"), filt)
        return path or None

    def cleanup(self) -> None:
        self.running = False
        self.close()


# ================================================ dialog widgets ===========
class _MessageDialog(QDialog):
    def __init__(self, parent, title: str, body: str, buttons: List[str]):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self.result_button: Optional[str] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        head = QHBoxLayout()
        head.addWidget(LogoWidget(36))
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size:17px; font-weight:600;")
        head.addWidget(title_lbl)
        head.addStretch(1)
        root.addLayout(head)

        body_lbl = QLabel(body)
        body_lbl.setWordWrap(True)
        body_lbl.setStyleSheet(
            f"background:{COLOR_PANEL}; border:1px solid {COLOR_BORDER};"
            "border-radius:8px; padding:14px; font-size:13px;")
        body_lbl.setMinimumHeight(80)
        root.addWidget(body_lbl)

        row = QHBoxLayout()
        row.addStretch(1)
        for i, txt in enumerate(buttons):
            b = QPushButton(txt)
            if i == len(buttons) - 1:
                b.setObjectName("primary")
            b.clicked.connect(lambda _=False, t=txt: self._pick(t))
            row.addWidget(b)
        root.addLayout(row)

    def _pick(self, choice: str) -> None:
        self.result_button = choice
        self.accept()


class _InputDialog(QDialog):
    def __init__(self, parent, title: str,
                 prompts: List[str], defaults: List[str]):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self.values: List[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        head = QHBoxLayout()
        head.addWidget(LogoWidget(32))
        t = QLabel(title)
        t.setStyleSheet("font-size:17px; font-weight:600;")
        head.addWidget(t)
        head.addStretch(1)
        root.addLayout(head)

        self._edits: List[QLineEdit] = []
        for prompt, default in zip(prompts, defaults):
            lbl = QLabel(prompt)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{COLOR_MUTED}; font-size:12px;")
            root.addWidget(lbl)
            edit = QLineEdit(default)
            self._edits.append(edit)
            root.addWidget(edit)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("OK")
        ok.setObjectName("primary")
        ok.clicked.connect(self._commit)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        root.addLayout(btns)

        if self._edits:
            self._edits[0].setFocus()
            self._edits[0].selectAll()

    def _commit(self) -> None:
        self.values = [e.text() for e in self._edits]
        self.accept()


class _ProgressDialog(QDialog):
    def __init__(self, parent, title: str, total: int):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self.setModal(True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self._total = max(1, int(total))
        self._cancelled = False

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        head = QHBoxLayout()
        head.addWidget(LogoWidget(36))
        self._title = QLabel(title)
        self._title.setStyleSheet("font-size:17px; font-weight:600;")
        head.addWidget(self._title)
        head.addStretch(1)
        root.addLayout(head)

        self._msg = QLabel("Preparing...")
        self._msg.setStyleSheet(f"color:{COLOR_MUTED}; font-size:12px;")
        root.addWidget(self._msg)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        root.addWidget(self._bar)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel = QPushButton("Cancel")
        self._cancel.setObjectName("danger")
        self._cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel)
        root.addLayout(btn_row)

    def _on_cancel(self) -> None:
        self._cancelled = True

    def wasCanceled(self) -> bool:
        return self._cancelled

    def update_progress(self, step: int, msg: str = "") -> None:
        pct = int(min(100, max(0, step / self._total * 100)))
        self._bar.setValue(pct)
        if msg:
            self._msg.setText(msg)
        if pct >= 100:
            QTimer.singleShot(250, self.accept)


# ==================================================== splash screen ========
def show_splash_screen(duration: float = 2.0) -> None:
    _qapp()
    size = 520
    pix = QPixmap(size, size)
    pix.fill(QColor(COLOR_BG))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    # Gradient background
    g = QLinearGradient(0, 0, 0, size)
    g.setColorAt(0, QColor("#F8FCFF"))
    g.setColorAt(1, QColor("#E1F2F8"))
    p.fillRect(0, 0, size, size, g)
    # Logo
    if os.path.exists(_ICON_PATH):
        icon = QPixmap(_ICON_PATH).scaled(200, 200, Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
        p.drawPixmap((size - 200) // 2, 80, icon)
    # Title
    p.setPen(QColor(COLOR_TEXT))
    f = QFont("Sans", 28, QFont.Bold)
    p.setFont(f)
    p.drawText(QRect(0, 300, size, 40), Qt.AlignCenter, APP_NAME)
    p.setPen(QColor(COLOR_MUTED))
    p.setFont(QFont("Sans", 11))
    p.drawText(QRect(0, 345, size, 30), Qt.AlignCenter,
               f"Version {APP_VERSION}   \u2022   Cell migration analysis")
    p.setPen(QColor(COLOR_ACCENT))
    p.setFont(QFont("Sans", 9))
    p.drawText(QRect(0, 450, size, 30), Qt.AlignCenter,
               "Loading modules...")
    p.end()

    splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
    splash.show()
    app = _qapp()
    from PyQt5.QtCore import QEventLoop
    loop = QEventLoop()
    QTimer.singleShot(int(duration * 1000), loop.quit)
    # Pump events so the splash actually paints
    loop.exec_()
    splash.close()
    app.processEvents()


# ============================================ image settings preview =======
class _ImageSettingsDialog(QDialog):
    """Brightness / contrast / gamma / filter preview on the first frame."""

    def __init__(self, parent, files: List[str], Settings):
        super().__init__(parent)
        self.setWindowTitle("Image Settings Preview")
        self.resize(1100, 720)
        self.Settings = Settings
        self.files = files
        self._frame_idx = 0
        self._frames: List[np.ndarray] = []
        self._confirmed = False

        # Load up to 10 frames for preview scrubbing
        from image_utils import apply_all_adjustments, FILTER_NAMES
        self._apply = apply_all_adjustments
        self._filter_names = FILTER_NAMES

        for f in files[:10]:
            im = cv2.imread(f)
            if im is not None:
                self._frames.append(im)
        if not self._frames:
            self._frames = [np.zeros((400, 600, 3), dtype=np.uint8)]

        root = QHBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)

        # ---- left: preview ----
        left = QVBoxLayout()
        left.setSpacing(10)

        head = QHBoxLayout()
        head.addWidget(LogoWidget(32))
        t = QLabel("Image Settings Preview")
        t.setStyleSheet("font-size:18px; font-weight:600;")
        head.addWidget(t)
        head.addStretch(1)
        left.addLayout(head)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setStyleSheet(
            f"background:#111317; border:1px solid {COLOR_BORDER}; "
            "border-radius:10px;")
        self._preview.setMinimumSize(640, 420)
        left.addWidget(self._preview, stretch=1)

        nav = QHBoxLayout()
        prev_btn = QPushButton("\u25c0  Previous frame")
        next_btn = QPushButton("Next frame  \u25b6")
        prev_btn.clicked.connect(lambda: self._nav(-1))
        next_btn.clicked.connect(lambda: self._nav(1))
        self._frame_lbl = QLabel()
        self._frame_lbl.setAlignment(Qt.AlignCenter)
        self._frame_lbl.setStyleSheet(f"color:{COLOR_MUTED};")
        nav.addWidget(prev_btn)
        nav.addStretch(1)
        nav.addWidget(self._frame_lbl)
        nav.addStretch(1)
        nav.addWidget(next_btn)
        left.addLayout(nav)

        root.addLayout(left, stretch=2)

        # ---- right: controls ----
        right = QVBoxLayout()
        right.setSpacing(12)

        card = QFrame()
        card.setObjectName("card")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(18, 18, 18, 18)
        card_lay.setSpacing(16)

        self._brightness = self._make_slider(
            card_lay, "Brightness", -100, 100, Settings.brightness,
            self._on_bright)
        self._contrast = self._make_slider(
            card_lay, "Contrast", 50, 300, int(Settings.contrast * 100),
            self._on_contrast, divisor=100.0, fmt="{:.2f}")
        self._gamma = self._make_slider(
            card_lay, "Gamma", 30, 300, int(Settings.gamma * 100),
            self._on_gamma, divisor=100.0, fmt="{:.2f}")

        f_lbl = QLabel("Filter")
        f_lbl.setStyleSheet(f"color:{COLOR_MUTED}; font-size:12px;")
        card_lay.addWidget(f_lbl)
        self._filter = QComboBox()
        self._filter.addItems(self._filter_names)
        self._filter.setCurrentIndex(int(Settings.filter_mode))
        self._filter.currentIndexChanged.connect(self._on_filter)
        card_lay.addWidget(self._filter)

        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self._reset)
        card_lay.addWidget(reset)

        right.addWidget(card)
        right.addStretch(1)

        btn_row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        confirm = QPushButton("Start Tracking")
        confirm.setObjectName("primary")
        confirm.clicked.connect(self._accept)
        btn_row.addWidget(cancel)
        btn_row.addStretch(1)
        btn_row.addWidget(confirm)
        right.addLayout(btn_row)

        root.addLayout(right, stretch=1)

        self._refresh()

    def _make_slider(self, parent_lay, label, lo, hi, value,
                     on_change, divisor: float = 1.0, fmt: str = "{:+d}"):
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color:{COLOR_MUTED}; font-size:12px;")
        parent_lay.addWidget(lbl)
        row = QHBoxLayout()
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(value)
        val_lbl = QLabel(fmt.format(value / divisor if divisor != 1.0 else value))
        val_lbl.setMinimumWidth(56)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val_lbl.setStyleSheet("font-weight:600;")

        def _handle(v):
            real = v / divisor if divisor != 1.0 else v
            val_lbl.setText(fmt.format(real))
            on_change(real)

        s.valueChanged.connect(_handle)
        row.addWidget(s, stretch=1)
        row.addWidget(val_lbl)
        parent_lay.addLayout(row)
        return s

    def _on_bright(self, v): self.Settings.brightness = int(v); self._refresh()
    def _on_contrast(self, v): self.Settings.contrast = float(v); self._refresh()
    def _on_gamma(self, v): self.Settings.gamma = float(v); self._refresh()
    def _on_filter(self, idx): self.Settings.filter_mode = int(idx); self._refresh()

    def _reset(self) -> None:
        self.Settings.brightness = 0
        self.Settings.contrast = 1.0
        self.Settings.gamma = 1.0
        self.Settings.filter_mode = 0
        self._brightness.setValue(0)
        self._contrast.setValue(100)
        self._gamma.setValue(100)
        self._filter.setCurrentIndex(0)
        self._refresh()

    def _nav(self, delta: int) -> None:
        self._frame_idx = (self._frame_idx + delta) % len(self._frames)
        self._refresh()

    def _refresh(self) -> None:
        frame = self._frames[self._frame_idx]
        out = self._apply(frame, self.Settings.brightness,
                          self.Settings.contrast, self.Settings.gamma,
                          self.Settings.filter_mode)
        pix = _np_bgr_to_qpixmap(out)
        if not pix.isNull():
            scaled = pix.scaled(self._preview.width(), self._preview.height(),
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._preview.setPixmap(scaled)
        self._frame_lbl.setText(
            f"Frame {self._frame_idx + 1} / {len(self._frames)}")

    def _accept(self) -> None:
        self._confirmed = True
        self.accept()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._refresh()


def show_image_settings_preview(parent, files: List[str], Settings) -> bool:
    _qapp()
    dlg = _ImageSettingsDialog(parent if isinstance(parent, QWidget) else None,
                               files, Settings)
    accepted = dlg.exec_() == QDialog.Accepted
    return accepted and dlg._confirmed


# ============================================ manual classification ========
class _ClickableImage(QLabel):
    clicked_cell = pyqtSignal(int)  # detection index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap = QPixmap()
        self._detections = []
        self._types = {}     # det_idx -> str
        self._palette: dict = {}
        self._scale = 1.0
        self._offset = QPoint(0, 0)

    def set_data(self, frame_bgr: np.ndarray, detections, palette: dict):
        self._pixmap = _np_bgr_to_qpixmap(frame_bgr)
        self._detections = detections
        self._palette = palette
        self._types = {}
        self._repaint()

    def set_type(self, det_idx: int, cell_type: str) -> None:
        self._types[det_idx] = cell_type
        self._repaint()

    def _repaint(self) -> None:
        if self._pixmap.isNull():
            return
        W, H = self.width(), self.height()
        if W < 10 or H < 10:
            return
        scaled = self._pixmap.scaled(W, H, Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation)
        self._scale = scaled.width() / self._pixmap.width()
        self._offset = QPoint((W - scaled.width()) // 2,
                              (H - scaled.height()) // 2)

        canvas = QPixmap(W, H)
        canvas.fill(QColor("#111317"))
        p = QPainter(canvas)
        p.setRenderHint(QPainter.Antialiasing)
        p.drawPixmap(self._offset, scaled)
        for i, d in enumerate(self._detections):
            x = int(d.x * self._scale) + self._offset.x()
            y = int(d.y * self._scale) + self._offset.y()
            w = int(d.w * self._scale)
            h = int(d.h * self._scale)
            ctype = self._types.get(i)
            if ctype is None:
                pen = QPen(QColor("#FACC15"), 2, Qt.DashLine)
            else:
                col = self._palette.get(ctype, QColor(COLOR_ACCENT))
                pen = QPen(col, 2)
            p.setPen(pen)
            p.drawRect(x, y, w, h)
            # label
            tag = f"{i+1}" if ctype is None else f"{i+1}:{ctype}"
            p.setPen(QColor("white"))
            p.setFont(QFont("Sans", 9, QFont.Bold))
            p.drawText(x, max(10, y - 4), tag)
        p.end()
        self.setPixmap(canvas)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._repaint()

    def mousePressEvent(self, ev):
        if self._pixmap.isNull() or not self._detections:
            return
        mx = ev.x() - self._offset.x()
        my = ev.y() - self._offset.y()
        if mx < 0 or my < 0:
            return
        fx = mx / self._scale
        fy = my / self._scale
        # Find nearest detection whose bbox contains the click.
        for i, d in enumerate(self._detections):
            if d.x <= fx <= d.x + d.w and d.y <= fy <= d.y + d.h:
                self.clicked_cell.emit(i)
                return


class _ManualClassifyDialog(QDialog):
    def __init__(self, parent, frame_bgr, detections, type_list):
        super().__init__(parent)
        self.setWindowTitle("Manual Cell Classification")
        self.resize(1200, 780)
        self._types: dict = {}
        self._current = 0
        self._detections = detections
        self._type_list = list(type_list)

        qt_palette = [QColor(c) for c in
                      ["#0891B2", "#15803D", "#B91C1C", "#CA8A04",
                       "#7C3AED", "#DB2777", "#0EA5E9", "#65A30D"]]
        self._palette = {t: qt_palette[i % len(qt_palette)]
                         for i, t in enumerate(self._type_list)}

        root = QHBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(16)

        # ---- left image ----
        left = QVBoxLayout()
        head = QHBoxLayout()
        head.addWidget(LogoWidget(30))
        t = QLabel("Manual Cell Classification")
        t.setStyleSheet("font-size:18px; font-weight:600;")
        head.addWidget(t)
        head.addStretch(1)
        self._progress_lbl = QLabel()
        self._progress_lbl.setStyleSheet(f"color:{COLOR_MUTED};")
        head.addWidget(self._progress_lbl)
        left.addLayout(head)

        self._img = _ClickableImage()
        self._img.setMinimumSize(720, 500)
        self._img.clicked_cell.connect(self._on_cell_click)
        left.addWidget(self._img, stretch=1)

        hint = QLabel(
            "Click any cell in the image to jump to it, or use "
            "\u2190 / \u2192 keys. Number keys 1-8 assign the "
            "corresponding type.")
        hint.setStyleSheet(f"color:{COLOR_MUTED}; font-size:12px;")
        hint.setWordWrap(True)
        left.addWidget(hint)

        root.addLayout(left, stretch=3)

        # ---- right panel ----
        right = QVBoxLayout()
        right.setSpacing(10)
        card = QFrame(); card.setObjectName("card")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(16, 16, 16, 16)
        card_lay.setSpacing(10)

        self._active_lbl = QLabel()
        self._active_lbl.setStyleSheet("font-size:15px; font-weight:600;")
        card_lay.addWidget(self._active_lbl)

        sub = QLabel("Assign a type:")
        sub.setStyleSheet(f"color:{COLOR_MUTED}; font-size:12px;")
        card_lay.addWidget(sub)

        for i, t in enumerate(self._type_list):
            col = self._palette[t].name()
            b = QPushButton(f"{i+1}.  {t}")
            b.setStyleSheet(
                f"QPushButton {{ text-align:left; padding:10px 14px;"
                f"border:1px solid {col}; color:{col}; background:white;"
                f"border-radius:6px; font-weight:600; }}"
                f"QPushButton:hover {{ background:{col}; color:white; }}")
            b.clicked.connect(lambda _=False, tt=t: self._assign(tt))
            card_lay.addWidget(b)

        card_lay.addSpacing(8)
        skip = QPushButton("Skip (leave unclassified)")
        skip.clicked.connect(self._skip)
        card_lay.addWidget(skip)

        nav = QHBoxLayout()
        prev = QPushButton("\u25c0  Previous")
        nxt = QPushButton("Next  \u25b6")
        prev.clicked.connect(lambda: self._move(-1))
        nxt.clicked.connect(lambda: self._move(1))
        nav.addWidget(prev)
        nav.addWidget(nxt)
        card_lay.addLayout(nav)

        right.addWidget(card)
        right.addStretch(1)

        btns = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        done = QPushButton("Done")
        done.setObjectName("primary")
        done.clicked.connect(self._commit)
        btns.addWidget(cancel)
        btns.addStretch(1)
        btns.addWidget(done)
        right.addLayout(btns)

        root.addLayout(right, stretch=1)

        self._img.set_data(frame_bgr, detections, self._palette)
        self._refresh()

    def keyPressEvent(self, ev):
        key = ev.key()
        if Qt.Key_1 <= key <= Qt.Key_8:
            idx = key - Qt.Key_1
            if idx < len(self._type_list):
                self._assign(self._type_list[idx])
                return
        if key == Qt.Key_Left:
            self._move(-1); return
        if key == Qt.Key_Right:
            self._move(1); return
        if key == Qt.Key_Return or key == Qt.Key_Enter:
            self._commit(); return
        super().keyPressEvent(ev)

    def _refresh(self) -> None:
        n = len(self._detections)
        if n == 0:
            return
        self._current = max(0, min(self._current, n - 1))
        done = sum(1 for v in self._types.values() if v)
        self._progress_lbl.setText(f"Classified {done} / {n}")
        # Highlight current by drawing via set_type loop — _ClickableImage
        # already colours by _types; we add a pulsing style via title text.
        cur_type = self._types.get(self._current, "unclassified")
        self._active_lbl.setText(
            f"Cell #{self._current + 1} of {n}   \u2014   {cur_type}")

    def _on_cell_click(self, idx: int) -> None:
        self._current = idx
        self._refresh()

    def _assign(self, cell_type: str) -> None:
        self._types[self._current] = cell_type
        self._img.set_type(self._current, cell_type)
        if self._current < len(self._detections) - 1:
            self._current += 1
        self._refresh()

    def _skip(self) -> None:
        self._types.pop(self._current, None)
        if self._current < len(self._detections) - 1:
            self._current += 1
        self._refresh()

    def _move(self, delta: int) -> None:
        self._current = (self._current + delta) % len(self._detections)
        self._refresh()

    def _commit(self) -> None:
        # All cells need a type.
        if len(self._types) < len(self._detections):
            box = QMessageBox(self)
            box.setWindowTitle("Incomplete")
            box.setText(
                f"Only {len(self._types)} of {len(self._detections)} "
                "cells are classified.\n\nClassify all cells or Cancel.")
            box.setStandardButtons(QMessageBox.Ok)
            box.exec_()
            return
        self.accept()


def manual_cell_classification(parent, frame_bgr, detections,
                               type_list) -> Optional[dict]:
    _qapp()
    dlg = _ManualClassifyDialog(
        parent if isinstance(parent, QWidget) else None,
        frame_bgr, detections, type_list)
    if dlg.exec_() == QDialog.Accepted:
        return dlg._types
    return None


# ==================================================== convenience ==========
def main_menu_only() -> None:
    """Entry point for debugging / standalone launching."""
    app = _qapp()
    win = FancyGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main_menu_only()
