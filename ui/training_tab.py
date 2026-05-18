from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                                   QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                                   QSlider, QProgressBar, QSplitter, QMessageBox,
                                   QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QMovie
from gym_env import TwoRoomsGymEnv
from ui.grid_widget import GridWidget
from ui.metrics_widget import MetricsWidget
from ui.training_thread import TrainingThread
from ui.video_player import VideoPlayer
import cv2
import numpy as np
import os


DEFAULT_ALPHA = 0.1
DEFAULT_GAMMA = 0.99
DEFAULT_EPSILON = 1.0
DEFAULT_EPSILON_MIN = 0.05
DEFAULT_EPSILON_DECAY = 0.9995
DEFAULT_EPISODES = 500
DEFAULT_SPEED = 50


class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.env = TwoRoomsGymEnv()
        self.thread = None
        self._training = False
        self._paused = False

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        left_panel = QWidget()
        left_panel.setFixedWidth(260)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        params_group = QGroupBox("Hiperparámetros")
        params_layout = QVBoxLayout(params_group)

        self._alpha_spin = self._make_double_spin("Alpha (α)", 0.01, 1.0, DEFAULT_ALPHA, 0.01,
                                                   "Tasa de aprendizaje")
        self._gamma_spin = self._make_double_spin("Gamma (γ)", 0.5, 1.0, DEFAULT_GAMMA, 0.01,
                                                   "Factor de descuento")
        self._epsilon_spin = self._make_double_spin("Epsilon (ε)", 0.01, 1.0, DEFAULT_EPSILON, 0.01,
                                                     "Exploración inicial")
        self._eps_min_spin = self._make_double_spin("Epsilon min", 0.0, 0.5, DEFAULT_EPSILON_MIN, 0.01,
                                                     "Exploración mínima")
        self._eps_decay_spin = self._make_double_spin("Epsilon decay", 0.9, 0.9999, DEFAULT_EPSILON_DECAY, 0.0001,
                                                       "Decaimiento por episodio")

        params_layout.addWidget(self._alpha_spin[0])
        params_layout.addWidget(self._gamma_spin[0])
        params_layout.addWidget(self._epsilon_spin[0])
        params_layout.addWidget(self._eps_min_spin[0])
        params_layout.addWidget(self._eps_decay_spin[0])
        left_layout.addWidget(params_group)

        train_group = QGroupBox("Entrenamiento")
        train_layout = QVBoxLayout(train_group)

        ep_layout = QHBoxLayout()
        ep_layout.addWidget(QLabel("Episodios:"))
        self._episodes_spin = QSpinBox()
        self._episodes_spin.setRange(1, 50000)
        self._episodes_spin.setValue(DEFAULT_EPISODES)
        self._episodes_spin.setSingleStep(100)
        ep_layout.addWidget(self._episodes_spin)
        train_layout.addLayout(ep_layout)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Velocidad:"))
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(0, 200)
        self._speed_slider.setValue(200 - DEFAULT_SPEED)
        self._speed_slider.setToolTip("ms entre pasos")
        speed_layout.addWidget(self._speed_slider)
        self._speed_label = QLabel(f"{DEFAULT_SPEED}ms")
        speed_layout.addWidget(self._speed_label)
        train_layout.addLayout(speed_layout)

        self._stop_on_goal_check = QCheckBox("Detener al alcanzar la meta")
        self._stop_on_goal_check.setChecked(True)
        train_layout.addWidget(self._stop_on_goal_check)

        left_layout.addWidget(train_group)

        btn_layout = QVBoxLayout()

        self._start_btn = QPushButton("Iniciar entrenamiento")
        self._start_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self._start_btn)

        self._pause_btn = QPushButton("Pausar")
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._pause_training)
        btn_layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("Detener")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self._stop_btn)

        left_layout.addLayout(btn_layout)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        left_layout.addWidget(self._progress)

        self._ep_label = QLabel("")
        self._ep_label.setStyleSheet("color: #888; font-size: 12px;")
        left_layout.addWidget(self._ep_label)

        post_group = QGroupBox("Post-entrenamiento")
        post_layout = QVBoxLayout(post_group)

        self._eval_btn = QPushButton("Evaluar agente")
        self._eval_btn.setEnabled(False)
        self._eval_btn.clicked.connect(self._run_evaluate)
        post_layout.addWidget(self._eval_btn)

        self._plot_btn = QPushButton("Curvas de aprendizaje (.png)")
        self._plot_btn.setEnabled(False)
        self._plot_btn.clicked.connect(self._run_plot)
        post_layout.addWidget(self._plot_btn)

        self._demo_btn = QPushButton("Generar demo GIF")
        self._demo_btn.setEnabled(False)
        self._demo_btn.clicked.connect(self._run_demo)
        post_layout.addWidget(self._demo_btn)

        self._report_btn = QPushButton("Generar reporte (.md)")
        self._report_btn.setEnabled(False)
        self._report_btn.clicked.connect(self._run_report)
        post_layout.addWidget(self._report_btn)

        self._post_status = QLabel("")
        self._post_status.setStyleSheet("color: #50fa7b; font-size: 11px;")
        post_layout.addWidget(self._post_status)

        left_layout.addWidget(post_group)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        top_right = QWidget()
        top_layout = QVBoxLayout(top_right)
        top_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_widget = GridWidget(self.env)
        top_layout.addWidget(self._grid_widget)
        right_splitter.addWidget(top_right)

        bot_right = QWidget()
        bot_layout = QVBoxLayout(bot_right)
        bot_layout.setContentsMargins(0, 0, 0, 0)

        self._video_player = VideoPlayer()
        bot_layout.addWidget(self._video_player)

        self._demo_viewer = QLabel("Demo GIF / Curvas PNG")
        self._demo_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._demo_viewer.setStyleSheet("background-color: #1e1e28; color: #888;")
        self._demo_viewer.setMinimumHeight(100)
        self._demo_viewer.hide()
        bot_layout.addWidget(self._demo_viewer)

        right_splitter.addWidget(bot_right)
        right_splitter.setSizes([420, 300])

        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.addWidget(right_splitter)

        self._metrics_widget = MetricsWidget()
        right_panel_layout.addWidget(self._metrics_widget)

        main_layout.addWidget(right_panel, 1)

        self._style_buttons()

    def _make_double_spin(self, label, min_val, max_val, default, step, tooltip):
        w = QWidget()
        w.setStyleSheet("""
            QWidget {
                background-color: #2a2d3a; border-radius: 4px;
            }
        """)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)
        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 12px; color: #bd93f9; font-weight: bold; background: transparent;")
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setDecimals(4)
        spin.setToolTip(tooltip)
        spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #1e1e2a; color: #f8f8f2;
                border: 1px solid #6272a4; padding: 4px 6px;
                border-radius: 4px; font-size: 13px;
            }
            QDoubleSpinBox:focus { border: 1px solid #bd93f9; }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #44475a; border: none; width: 16px;
                border-radius: 2px;
            }
        """)
        layout.addWidget(lbl)
        layout.addWidget(spin)
        return w, spin

    def _style_buttons(self):
        style_start = """
            QPushButton {
                background-color: #50fa7b; color: #282a36;
                border: none; padding: 10px; border-radius: 4px;
                font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #69ff94; }
            QPushButton:disabled { background-color: #444; color: #888; }
        """
        style_ctrl = """
            QPushButton {
                background-color: #6272a4; color: white;
                border: none; padding: 10px; border-radius: 4px;
                font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #7082b4; }
            QPushButton:disabled { background-color: #444; color: #888; }
        """
        style_stop = """
            QPushButton {
                background-color: #ff5555; color: white;
                border: none; padding: 10px; border-radius: 4px;
                font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #ff7777; }
            QPushButton:disabled { background-color: #444; color: #888; }
        """
        style_post = """
            QPushButton {
                background-color: #bd93f9; color: #282a36;
                border: none; padding: 8px; border-radius: 4px;
                font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #d4aaff; }
            QPushButton:disabled { background-color: #444; color: #888; }
        """
        self._start_btn.setStyleSheet(style_start)
        self._pause_btn.setStyleSheet(style_ctrl)
        self._stop_btn.setStyleSheet(style_stop)
        for btn in [self._eval_btn, self._plot_btn, self._demo_btn, self._report_btn]:
            btn.setStyleSheet(style_post)

    def _connect_signals(self):
        self._speed_slider.valueChanged.connect(self._on_speed_changed)

    def _on_speed_changed(self, value):
        ms = 200 - value
        self._speed_label.setText(f"{ms}ms")

    @property
    def speed_ms(self):
        return 200 - self._speed_slider.value()

    @property
    def alpha(self):
        return self._alpha_spin[1].value()

    @property
    def gamma(self):
        return self._gamma_spin[1].value()

    @property
    def epsilon(self):
        return self._epsilon_spin[1].value()

    @property
    def epsilon_min(self):
        return self._eps_min_spin[1].value()

    @property
    def epsilon_decay(self):
        return self._eps_decay_spin[1].value()

    @property
    def n_episodes(self):
        return self._episodes_spin.value()

    def _start_training(self):
        self._metrics_widget.reset()

        self._training = True
        self._paused = False
        self._start_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._pause_btn.setText("Pausar")
        self._progress.setVisible(True)
        self._progress.setRange(0, self.n_episodes)
        self._progress.setValue(0)

        self.env.reset()

        self.thread = TrainingThread()
        self.thread.configure(
            n_episodes=self.n_episodes,
            speed_ms=self.speed_ms,
            alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon,
            epsilon_min=self.epsilon_min, epsilon_decay=self.epsilon_decay,
            stop_on_goal=self._stop_on_goal_check.isChecked(),
        )

        self.thread.step_signal.connect(self._on_step)
        self.thread.episode_signal.connect(self._on_episode)
        self.thread.progress_signal.connect(self._on_progress)
        self.thread.finished_signal.connect(self._on_finished)
        self.thread.goal_reached_signal.connect(self._on_goal_reached)
        self.thread.start()

    def _pause_training(self):
        if self.thread is None:
            return
        if self._paused:
            self.thread.resume()
            self._paused = False
            self._pause_btn.setText("Pausar")
        else:
            self.thread.pause()
            self._paused = True
            self._pause_btn.setText("Reanudar")

    def _stop_training(self):
        if self.thread is None:
            return
        self.thread.stop()
        self.thread.wait(2000)
        self._reset_ui()

    def _on_step(self, state, action, reward, done, info):
        self._grid_widget.update_step(state, action, reward, done, info)

    def _on_episode(self, episode, reward, steps, epsilon):
        self._metrics_widget.add_episode(episode, reward, steps, epsilon)

    def _on_progress(self, current, total):
        self._progress.setValue(current)
        self._ep_label.setText(f"Episodio {current}/{total}")

    def _on_goal_reached(self, episode):
        self._ep_label.setText(f"Meta alcanzada en episodio {episode}!")
        if self._stop_on_goal_check.isChecked():
            self._generate_video()
        self._enable_post_actions()

    def _on_finished(self):
        self._reset_ui()
        if not self._stop_on_goal_check.isChecked():
            self._generate_video()
        self._enable_post_actions()

        if self.thread:
            agent = self.thread.get_agent()
            main_win = self._find_main_window()
            if main_win:
                main_win.on_training_finished(agent)

    def _enable_post_actions(self):
        has_log = os.path.exists("training_log.json") or (
            self.thread and self.thread._rewards_h
        )
        has_qtable = os.path.exists("qtable.pkl")
        self._eval_btn.setEnabled(has_qtable)
        self._plot_btn.setEnabled(has_log)
        self._demo_btn.setEnabled(has_qtable)
        self._report_btn.setEnabled(has_qtable and has_log)
        self._post_status.setText("Q-table y log guardados. Acciones disponibles." 
                                  if has_qtable else "Esperando archivos...")

    def _run_evaluate(self):
        from evaluate import evaluate
        msg = QMessageBox(self)
        msg.setWindowTitle("Evaluando agente...")
        msg.setText("Ejecutando evaluación...")
        msg.show()
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, lambda: self._do_evaluate(msg))

    def _do_evaluate(self, msg):
        from evaluate import evaluate
        try:
            all_r, all_s, all_ok = evaluate("qtable.pkl", n_episodes=10, render=False)
            msg.close()
            text = (f"Tasa de éxito: {np.mean(all_ok)*100:.1f}%\n"
                    f"Recompensa prom: {np.mean(all_r):.2f} ± {np.std(all_r):.2f}\n"
                    f"Pasos promedio: {np.mean(all_s):.1f}")
            QMessageBox.information(self, "Resultado de evaluación", text)
        except Exception as e:
            msg.close()
            QMessageBox.warning(self, "Error", str(e))

    def _run_plot(self):
        from plot_results import plot
        try:
            plot("training_log.json", "learning_curves.png")
            pix = QPixmap("learning_curves.png")
            self._demo_viewer.show()
            self._demo_viewer.setPixmap(pix.scaled(
                self._demo_viewer.width(), self._demo_viewer.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            self._post_status.setText("Curvas generadas: learning_curves.png")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _run_demo(self):
        self._post_status.setText("Generando demo GIF...")
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._do_demo)

    def _do_demo(self):
        from visualize import generate_demo
        try:
            generate_demo("qtable.pkl", "demo_agente.gif", save_frames=False, fps=4)
            movie = QMovie("demo_agente.gif")
            self._demo_viewer.show()
            self._demo_viewer.setMovie(movie)
            movie.start()
            self._post_status.setText("Demo GIF generado: demo_agente.gif")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _run_report(self):
        from ui.report_generator import generate_report
        params = {
            "alpha": self.alpha, "gamma": self.gamma,
            "epsilon": self.epsilon, "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        try:
            generate_report(agent_params=params)
            self._post_status.setText("Reporte generado: reporte.md")
            QMessageBox.information(self, "Reporte", "reporte.md generado exitosamente.")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _generate_video(self):
        if not self.thread or not self.thread.frames:
            return

        frames = self.thread.frames
        if not frames:
            return

        h, w = frames[0].height(), frames[0].width()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("training_video.mp4", fourcc, 10, (w, h))

        for qimg in frames:
            ptr = qimg.bits()
            ptr.setsize(qimg.height() * qimg.bytesPerLine())
            arr = np.array(ptr).reshape(qimg.height(), qimg.bytesPerLine() // 4, 4)
            arr = arr[..., :3]
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            out.write(bgr)

        out.release()
        self._video_player.load_video("training_video.mp4")

    def _reset_ui(self):
        self._training = False
        self._paused = False
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._pause_btn.setText("Pausar")
        if self.thread:
            self.thread.step_signal.disconnect()
            self.thread.episode_signal.disconnect()
            self.thread.progress_signal.disconnect()
            self.thread.finished_signal.disconnect()
            self.thread.goal_reached_signal.disconnect()

    def _find_main_window(self):
        p = self.parent()
        while p is not None:
            from ui.main_window import MainWindow
            if isinstance(p, MainWindow):
                return p
            p = p.parent()
        return None
