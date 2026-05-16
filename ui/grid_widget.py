from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QFont, QPen
from PyQt6.QtCore import Qt
from environment import EMPTY, WALL, KEY, BALL, DOOR_CLOSED, DOOR_OPEN, GOAL


class GridWidget(QWidget):
    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env = env
        self.cell_size = 56
        self.margin = 8
        self._last_state = None
        self._last_action = -1
        self._last_reward = 0.0
        self._last_info = {}
        self.setMinimumSize(
            env.WIDTH * self.cell_size + self.margin * 2,
            env.HEIGHT * self.cell_size + self.margin * 2 + 50,
        )

    def update_step(self, state, action, reward, done, info):
        self._last_state = state
        self._last_action = action
        self._last_reward = reward
        self._last_info = info
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        env = self.env
        cell = self.cell_size
        mg = self.margin

        painter.fillRect(self.rect(), QColor(40, 42, 54))

        font = QFont("monospace", 9)
        painter.setFont(font)

        for y in range(env.HEIGHT):
            for x in range(env.WIDTH):
                rx = mg + x * cell
                ry = mg + y * cell

                if env._is_wall(x, y):
                    painter.fillRect(rx, ry, cell, cell, QColor(80, 80, 100))
                    painter.setPen(QPen(QColor(120, 120, 140), 1))
                    painter.drawRect(rx, ry, cell, cell)
                else:
                    painter.fillRect(rx, ry, cell, cell, QColor(55, 55, 70))
                    painter.setPen(QPen(QColor(90, 90, 100), 1))
                    painter.drawRect(rx, ry, cell, cell)

                content = env._cell_content(x, y)
                cx, cy = rx + cell // 2, ry + cell // 2

                if content == GOAL:
                    painter.setBrush(QColor(80, 250, 123))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(cx - 16, cy - 16, 32, 32)
                    painter.setPen(QColor(0, 0, 0))
                    font_g = QFont("monospace", 11, QFont.Weight.Bold)
                    painter.setFont(font_g)
                    painter.drawText(rx, ry, cell, cell, Qt.AlignmentFlag.AlignCenter, "G")
                    painter.setFont(font)
                elif content == KEY:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(255, 184, 108))
                    painter.drawRect(cx - 9, cy - 15, 7, 20)
                    painter.drawEllipse(cx - 11, cy - 20, 18, 11)
                    painter.drawRect(cx, cy + 5, 9, 4)
                elif content == BALL:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(255, 85, 85))
                    painter.drawEllipse(cx - 13, cy - 13, 26, 26)
                elif content == DOOR_CLOSED:
                    painter.fillRect(rx + 4, ry + 4, cell - 8, cell - 8, QColor(189, 147, 249))
                    painter.setPen(QPen(QColor(150, 110, 210), 3))
                    painter.drawLine(cx, ry + 6, cx, ry + cell - 6)
                elif content == DOOR_OPEN:
                    painter.fillRect(rx + 2, ry + 2, cell - 4, cell - 4, QColor(100, 200, 150))

                if [x, y] == env.agent_pos:
                    ac = QColor(80, 180, 255) if not env.has_key else QColor(255, 184, 108)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(ac)
                    painter.drawEllipse(cx - 14, cy - 14, 28, 28)
                    painter.setPen(QColor(0, 0, 0))
                    font_a = QFont("monospace", 9, QFont.Weight.Bold)
                    painter.setFont(font_a)
                    painter.drawText(rx, ry, cell, cell, Qt.AlignmentFlag.AlignCenter,
                                     "Ak" if env.has_key else "A")
                    painter.setFont(font)

        status_y = env.HEIGHT * cell + mg + 8
        painter.setPen(QColor(200, 200, 200))

        action_names = ["UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "TOGGLE"]
        action_str = action_names[self._last_action] if 0 <= self._last_action < 6 else "-"
        status = (f"Paso={env.steps} | llave={env.has_key} "
                  f"| puerta={env.door_open} | bola={env.ball_removed} | "
                  f"Acción={action_str} | R={self._last_reward:+.1f}")

        painter.drawText(mg, status_y, self.width() - mg * 2, 20,
                         Qt.AlignmentFlag.AlignLeft, status)

        status_y2 = status_y + 22
        event_str = self._last_info.get("event", "-") if self._last_info else "-"
        painter.setPen(QColor(150, 200, 150))
        painter.drawText(mg, status_y2, self.width() - mg * 2, 20,
                         Qt.AlignmentFlag.AlignLeft, f"Evento: {event_str}")

        painter.end()
