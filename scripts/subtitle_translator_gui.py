#!/usr/bin/env python3
"""Run the screen subtitle OCR + translation GUI.

Usage examples:
    python scripts/subtitle_translator_gui.py
    python scripts/subtitle_translator_gui.py --ocr-lang eng --target zh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.subtitle_translation_tool import (  # noqa: E402
    ScreenOCR,
    StabilityDetector,
    SubtitleTranslatorConfig,
    TencentTranslator,
)

try:
    from PyQt5.QtCore import QPoint, QRect, QThread, Qt, pyqtSignal
    from PyQt5.QtGui import QFont, QPainter, QPen
    from PyQt5.QtWidgets import (
        QApplication,
        QDoubleSpinBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as import_error:  # pragma: no cover - import guard for CLI use
    raise SystemExit(
        "PyQt5 is required for this GUI. Install with: pip install PyQt5"
    ) from import_error


class RegionSelector(QWidget):
    """Transparent full-screen region selector."""

    region_selected = pyqtSignal(tuple)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.3)
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.begin = QPoint()
        self.end = QPoint()
        self.is_drawing = False

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.is_drawing:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
            painter.drawRect(QRect(self.begin, self.end))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            self.begin = event.pos()
            self.end = event.pos()
            self.is_drawing = True
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.is_drawing:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            x1 = min(self.begin.x(), self.end.x())
            y1 = min(self.begin.y(), self.end.y())
            x2 = max(self.begin.x(), self.end.x())
            y2 = max(self.begin.y(), self.end.y())
            if x2 - x1 > 10 and y2 - y1 > 10:
                self.region_selected.emit((x1, y1, x2, y2))
            self.close()


class MonitorThread(QThread):
    """Background OCR stability + translation worker."""

    text_detected = pyqtSignal(str)
    translation_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self, bbox: tuple[int, int, int, int], config: SubtitleTranslatorConfig
    ) -> None:
        super().__init__()
        self.bbox = bbox
        self.config = config
        self.running = False
        self.translator = None
        self.detector = None

    def run(self) -> None:
        self.running = True
        try:
            ocr = ScreenOCR(
                tesseract_cmd=self.config.tesseract_cmd,
                default_lang=self.config.ocr_language,
            )
            self.detector = StabilityDetector(
                stable_duration=self.config.stable_duration,
                capture_interval=self.config.capture_interval,
                ocr=ocr,
            )
            self.translator = TencentTranslator(self.config)
        except Exception as exception:
            self.error_occurred.emit(str(exception))
            return

        self.detector.monitor_region(
            self.bbox,
            self.on_stable_content,
            lang=self.config.ocr_language,
            stop_flag=lambda: not self.running,
        )

    def on_stable_content(self, text: str) -> None:
        if not text.strip():
            return
        self.text_detected.emit(text)
        if self.translator is not None:
            self.translation_completed.emit(self.translator.translate(text))

    def stop(self) -> None:
        self.running = False


class SubtitleTranslatorWindow(QMainWindow):
    """Main subtitle translator window."""

    def __init__(self, config: SubtitleTranslatorConfig) -> None:
        super().__init__()
        self.config = config
        self.bbox: tuple[int, int, int, int] | None = None
        self.monitor_thread: MonitorThread | None = None
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("字幕翻译器")
        self.setGeometry(100, 100, 680, 520)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        region_group = QGroupBox("选择屏幕区域")
        region_layout = QHBoxLayout()
        self.select_btn = QPushButton("选择区域")
        self.select_btn.clicked.connect(self.select_region)
        region_layout.addWidget(self.select_btn)
        self.region_label = QLabel("未选择")
        region_layout.addWidget(self.region_label)
        region_layout.addStretch()
        region_group.setLayout(region_layout)
        main_layout.addWidget(region_group)

        config_group = QGroupBox("配置")
        config_layout = QVBoxLayout()

        stable_layout = QHBoxLayout()
        stable_layout.addWidget(QLabel("稳定持续时间(秒):"))
        self.stable_spin = QDoubleSpinBox()
        self.stable_spin.setRange(0.5, 10.0)
        self.stable_spin.setSingleStep(0.5)
        self.stable_spin.setValue(self.config.stable_duration)
        stable_layout.addWidget(self.stable_spin)
        config_layout.addLayout(stable_layout)

        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("截图间隔(秒):"))
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 5.0)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setValue(self.config.capture_interval)
        interval_layout.addWidget(self.interval_spin)
        config_layout.addLayout(interval_layout)

        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("OCR语言:"))
        self.ocr_lang_input = QLineEdit(self.config.ocr_language)
        lang_layout.addWidget(self.ocr_lang_input)
        lang_layout.addWidget(QLabel("目标语言:"))
        self.target_lang_input = QLineEdit(self.config.target_lang)
        lang_layout.addWidget(self.target_lang_input)
        config_layout.addLayout(lang_layout)

        api_hint = QLabel(
            "腾讯云密钥从环境变量 TENCENT_SECRET_ID / TENCENT_SECRET_KEY 读取"
        )
        api_hint.setStyleSheet("color: #667085;")
        config_layout.addWidget(api_hint)
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始监控")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setEnabled(False)
        self.stop_btn = QPushButton("停止监控")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        main_layout.addLayout(control_layout)

        display_group = QGroupBox("翻译结果")
        display_layout = QVBoxLayout()
        display_layout.addWidget(QLabel("原文:"))
        self.source_text = QTextEdit()
        self.source_text.setReadOnly(True)
        self.source_text.setMaximumHeight(120)
        display_layout.addWidget(self.source_text)
        display_layout.addWidget(QLabel("译文:"))
        self.translation_text = QTextEdit()
        self.translation_text.setReadOnly(True)
        font = QFont()
        font.setPointSize(16)
        self.translation_text.setFont(font)
        display_layout.addWidget(self.translation_text)
        display_group.setLayout(display_layout)
        main_layout.addWidget(display_group)

        self.statusBar().showMessage("就绪")

    def select_region(self) -> None:
        self.hide()
        selector = RegionSelector()
        selector.region_selected.connect(self.on_region_selected)
        selector.show()

    def on_region_selected(self, bbox: tuple[int, int, int, int]) -> None:
        self.bbox = bbox
        self.region_label.setText(f"({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})")
        self.start_btn.setEnabled(True)
        self.show()
        self.statusBar().showMessage(f"已选择区域: {bbox}")

    def _current_config(self) -> SubtitleTranslatorConfig:
        config = SubtitleTranslatorConfig.from_env()
        config.stable_duration = self.stable_spin.value()
        config.capture_interval = self.interval_spin.value()
        config.ocr_language = self.ocr_lang_input.text().strip() or "chi_sim+eng"
        config.target_lang = self.target_lang_input.text().strip() or "zh"
        return config

    def start_monitoring(self) -> None:
        if self.bbox is None:
            QMessageBox.warning(self, "警告", "请先选择屏幕区域")
            return

        self.monitor_thread = MonitorThread(self.bbox, self._current_config())
        self.monitor_thread.text_detected.connect(self.on_text_detected)
        self.monitor_thread.translation_completed.connect(self.on_translation_completed)
        self.monitor_thread.error_occurred.connect(self.on_error)
        self.monitor_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.select_btn.setEnabled(False)
        self.stable_spin.setEnabled(False)
        self.interval_spin.setEnabled(False)
        self.ocr_lang_input.setEnabled(False)
        self.target_lang_input.setEnabled(False)
        self.statusBar().showMessage("监控中...")

    def stop_monitoring(self) -> None:
        if self.monitor_thread is not None:
            self.monitor_thread.stop()
            self.monitor_thread.wait()
            self.monitor_thread = None
        self.start_btn.setEnabled(self.bbox is not None)
        self.stop_btn.setEnabled(False)
        self.select_btn.setEnabled(True)
        self.stable_spin.setEnabled(True)
        self.interval_spin.setEnabled(True)
        self.ocr_lang_input.setEnabled(True)
        self.target_lang_input.setEnabled(True)
        self.statusBar().showMessage("已停止")

    def on_text_detected(self, text: str) -> None:
        self.source_text.setText(text)
        self.statusBar().showMessage("检测到新文本，正在翻译...")

    def on_translation_completed(self, result: dict) -> None:
        if result.get("error"):
            self.translation_text.setText(f"错误: {result['error']}")
            self.statusBar().showMessage("翻译失败")
            return
        self.translation_text.setText(result["target_text"])
        self.statusBar().showMessage(
            f"翻译完成 | {result['source']} -> {result['target']} | 用量: {result['used_amount']} 字符"
        )

    def on_error(self, error_msg: str) -> None:
        QMessageBox.critical(self, "错误", error_msg)
        self.stop_monitoring()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.stop_monitoring()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the subtitle translator GUI.")
    parser.add_argument(
        "--stable-duration",
        type=float,
        default=None,
        help="Initial stable duration seconds",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=None,
        help="Initial capture interval seconds",
    )
    parser.add_argument(
        "--ocr-lang", default=None, help="Initial OCR language, e.g. chi_sim+eng"
    )
    parser.add_argument(
        "--target", default=None, help="Initial target language, e.g. zh"
    )
    args = parser.parse_args()

    config = SubtitleTranslatorConfig.from_env()
    if args.stable_duration is not None:
        config.stable_duration = args.stable_duration
    if args.capture_interval is not None:
        config.capture_interval = args.capture_interval
    if args.ocr_lang:
        config.ocr_language = args.ocr_lang
    if args.target:
        config.target_lang = args.target

    application = QApplication(sys.argv)
    window = SubtitleTranslatorWindow(config)
    window.show()
    return application.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
