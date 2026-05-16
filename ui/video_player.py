from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                   QPushButton, QLabel, QFileDialog)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import os
import shutil
import cv2


class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path = "training_video.mp4"
        self._generated = False
        self._cap = None
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

        layout = QVBoxLayout(self)

        self._status_label = QLabel("No hay video generado")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("color: #888; font-size: 14px; padding: 10px;")
        layout.addWidget(self._status_label)

        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet("background-color: #1e1e28;")
        self._video_label.setMinimumHeight(280)
        self._video_label.hide()
        layout.addWidget(self._video_label)

        btn_layout = QHBoxLayout()

        self._play_btn = QPushButton("Reproducir")
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_play)
        btn_layout.addWidget(self._play_btn)

        self._download_btn = QPushButton("Descargar video")
        self._download_btn.setEnabled(False)
        self._download_btn.clicked.connect(self._download_video)
        btn_layout.addWidget(self._download_btn)

        layout.addLayout(btn_layout)

        self.style_buttons()

    def style_buttons(self):
        style = """
            QPushButton {
                background-color: #6272a4; color: white;
                border: none; padding: 8px 16px; border-radius: 4px;
                font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #7082b4; }
            QPushButton:disabled { background-color: #444; color: #888; }
        """
        self._play_btn.setStyleSheet(style)
        self._download_btn.setStyleSheet(style)

    def load_video(self, path):
        if not os.path.exists(path):
            self._status_label.setText("Video no encontrado")
            self._status_label.show()
            self._video_label.hide()
            self._play_btn.setEnabled(False)
            self._download_btn.setEnabled(False)
            self._generated = False
            return

        self._stop_playback()
        self.video_path = path
        self._generated = True
        self._status_label.hide()
        self._video_label.show()

        self._cap = cv2.VideoCapture(path)
        if self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                self._show_frame(frame)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._play_btn.setEnabled(True)
        self._download_btn.setEnabled(True)
        self._play_btn.setText("Reproducir")

    def _toggle_play(self):
        if self._playing:
            self._stop_playback()
            self._play_btn.setText("Reproducir")
        else:
            self._start_playback()
            self._play_btn.setText("Pausar")

    def _start_playback(self):
        if not self._generated or not os.path.exists(self.video_path):
            return
        self._stop_playback()
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            return
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        interval = int(1000 / fps) if fps > 0 else 100
        self._playing = True
        self._timer.start(interval)

    def _stop_playback(self):
        self._playing = False
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _next_frame(self):
        if self._cap is None or not self._playing:
            return
        ret, frame = self._cap.read()
        if ret:
            self._show_frame(frame)
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            if ret:
                self._show_frame(frame)
            else:
                self._stop_playback()
                self._play_btn.setText("Reproducir")

    def _show_frame(self, frame):
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        scaled = qimg.scaled(
            self._video_label.width(), self._video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._video_label.setPixmap(QPixmap.fromImage(scaled))

    def _download_video(self):
        if not self._generated or not os.path.exists(self.video_path):
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Guardar video", "training_video.mp4",
            "Video MP4 (*.mp4);;All Files (*)"
        )
        if dest:
            shutil.copy2(self.video_path, dest)
            self._status_label.setText(f"Video guardado en: {dest}")
            self._status_label.show()

    def has_video(self):
        return self._generated and os.path.exists(self.video_path)

    def closeEvent(self, event):
        self._stop_playback()
        super().closeEvent(event)
