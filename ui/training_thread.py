from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
import numpy as np
from environment import TwoRoomsEnv
from agent import QLearningAgent


class TrainingThread(QThread):
    step_signal = pyqtSignal(tuple, int, float, bool, dict)
    episode_signal = pyqtSignal(int, float, int, float)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal()
    goal_reached_signal = pyqtSignal(int)
    frame_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.env = TwoRoomsEnv()
        self.agent = QLearningAgent()
        self.n_episodes = 500
        self.speed_ms = 50
        self.record_video = False
        self.frames = []
        self._paused = False
        self._stopped = False
        self._mutex = QMutex()
        self._cond = QWaitCondition()

    def configure(self, n_episodes, speed_ms, record_video,
                  alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.n_episodes = n_episodes
        self.speed_ms = speed_ms
        self.record_video = record_video
        self.agent = QLearningAgent(
            alpha=alpha, gamma=gamma, epsilon=epsilon,
            epsilon_min=epsilon_min, epsilon_decay=epsilon_decay
        )
        self.frames = []
        self._goal_reached = False

    def pause(self):
        self._mutex.lock()
        self._paused = True
        self._mutex.unlock()

    def resume(self):
        self._mutex.lock()
        self._paused = False
        self._cond.wakeAll()
        self._mutex.unlock()

    def stop(self):
        self._mutex.lock()
        self._stopped = True
        self._paused = False
        self._cond.wakeAll()
        self._mutex.unlock()

    def _check_paused(self):
        self._mutex.lock()
        while self._paused and not self._stopped:
            self._cond.wait(self._mutex)
        self._mutex.unlock()

    def run(self):
        self._stopped = False
        self._paused = False
        self._goal_reached = False
        self.frames = []

        for ep in range(self.n_episodes):
            if self._stopped:
                break

            state = self.env.reset()
            done = False
            total_reward = 0
            step_count = 0

            grid = self._render_grid_image()
            self.step_signal.emit(state, -1, 0, False, {"event": None})
            if self.record_video and grid is not None:
                self.frames.append(grid)

            while not done and not self._stopped:
                self._check_paused()
                if self._stopped:
                    break

                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.update(state, action, reward, next_state, done)

                total_reward += reward
                step_count += 1
                state = next_state

                if info.get("event") == "GOAL_REACHED":
                    self._goal_reached = True

                grid = self._render_grid_image()
                self.step_signal.emit(state, action, reward, done, info)

                if self.record_video and grid is not None:
                    self.frames.append(grid)

                if self.speed_ms > 0:
                    self.msleep(self.speed_ms)

            if self._stopped:
                break

            self.agent.decay_epsilon()
            self.episode_signal.emit(ep + 1, total_reward, step_count, self.agent.epsilon)
            self.progress_signal.emit(ep + 1, self.n_episodes)

            if self._goal_reached:
                self.goal_reached_signal.emit(ep + 1)
                break

        self.finished_signal.emit()

    def _render_grid_image(self):
        from PyQt6.QtGui import QImage, QPainter, QColor, QFont, QPen
        from PyQt6.QtCore import Qt
        from environment import EMPTY, WALL, KEY, BALL, DOOR_CLOSED, DOOR_OPEN, GOAL

        env = self.env
        cell_size = 48
        margin = 4
        img_w = env.WIDTH * cell_size + margin * 2
        img_h = env.HEIGHT * cell_size + margin * 2 + 28

        img = QImage(img_w, img_h, QImage.Format.Format_ARGB32)
        img.fill(QColor(40, 42, 54))

        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("monospace", 10)
        painter.setFont(font)

        for y in range(env.HEIGHT):
            for x in range(env.WIDTH):
                rx = margin + x * cell_size
                ry = margin + y * cell_size

                if env._is_wall(x, y):
                    color = QColor(80, 80, 100)
                    painter.fillRect(rx, ry, cell_size, cell_size, color)
                    painter.setPen(QPen(QColor(120, 120, 140), 1))
                    painter.drawRect(rx, ry, cell_size, cell_size)
                else:
                    painter.fillRect(rx, ry, cell_size, cell_size, QColor(55, 55, 70))
                    painter.setPen(QPen(QColor(90, 90, 100), 1))
                    painter.drawRect(rx, ry, cell_size, cell_size)

                content = env._cell_content(x, y)
                cx, cy = rx + cell_size // 2, ry + cell_size // 2

                if content == GOAL:
                    painter.setBrush(QColor(80, 250, 123))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(cx - 14, cy - 14, 28, 28)
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(rx, ry, cell_size, cell_size, Qt.AlignmentFlag.AlignCenter, "G")
                elif content == KEY:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(255, 184, 108))
                    painter.drawRect(cx - 8, cy - 14, 6, 18)
                    painter.drawEllipse(cx - 10, cy - 18, 16, 10)
                    painter.drawRect(cx + 2, cy + 4, 8, 4)
                elif content == BALL:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(255, 85, 85))
                    painter.drawEllipse(cx - 11, cy - 11, 22, 22)
                elif content == DOOR_CLOSED:
                    painter.fillRect(rx + 4, ry + 4, cell_size - 8, cell_size - 8, QColor(189, 147, 249))
                    painter.setPen(QPen(QColor(150, 110, 210), 2))
                    painter.drawLine(cx, ry + 6, cx, ry + cell_size - 6)
                elif content == DOOR_OPEN:
                    painter.fillRect(rx + 2, ry + 2, cell_size - 4, cell_size - 4, QColor(100, 200, 150))

                if [x, y] == env.agent_pos:
                    agent_color = QColor(255, 85, 85) if not env.has_key else QColor(255, 184, 108)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(agent_color)
                    painter.drawEllipse(cx - 12, cy - 12, 24, 24)
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(rx, ry, cell_size, cell_size, Qt.AlignmentFlag.AlignCenter,
                                     "Ak" if env.has_key else "A")

        status_y = img_h - 24
        painter.setPen(QColor(200, 200, 200))
        status = (f"Paso={env.steps} | llave={env.has_key} "
                  f"| puerta={env.door_open} | bola={env.ball_removed}")
        painter.drawText(margin, status_y, img_w - margin * 2, 20,
                         Qt.AlignmentFlag.AlignLeft, status)

        painter.end()
        return img

    def get_agent_qtable(self):
        return self.agent.Q

    def get_agent(self):
        return self.agent
