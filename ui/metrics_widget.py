import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from PyQt6.QtCore import pyqtSignal
from collections import deque
import numpy as np


class MetricsWidget(QWidget):
    episode_updated = pyqtSignal(int, float, int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilons = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        self.reward_plot = self.graph_widget.addPlot(title="Recompensa por episodio", row=0, col=0)
        self.reward_plot.setLabel("left", "Recompensa")
        self.reward_plot.setLabel("bottom", "Episodio")
        self.reward_plot.showGrid(x=True, y=True, alpha=0.3)
        self.reward_curve = self.reward_plot.plot(pen=pg.mkPen(color=(80, 250, 123), width=2))

        self.epsilon_plot = self.graph_widget.addPlot(title="Epsilon decay", row=1, col=0)
        self.epsilon_plot.setLabel("left", "Epsilon")
        self.epsilon_plot.setLabel("bottom", "Episodio")
        self.epsilon_plot.showGrid(x=True, y=True, alpha=0.3)
        self.epsilon_curve = self.epsilon_plot.plot(pen=pg.mkPen(color=(255, 184, 108), width=2))

        self.steps_plot = self.graph_widget.addPlot(title="Pasos por episodio", row=2, col=0)
        self.steps_plot.setLabel("left", "Pasos")
        self.steps_plot.setLabel("bottom", "Episodio")
        self.steps_plot.showGrid(x=True, y=True, alpha=0.3)
        self.steps_curve = self.steps_plot.plot(pen=pg.mkPen(color=(137, 180, 250), width=2))

    def add_episode(self, episode, reward, steps, epsilon):
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.epsilons.append(epsilon)

        x = np.arange(1, len(self.episode_rewards) + 1)

        self.reward_curve.setData(x, np.array(self.episode_rewards))
        self.epsilon_curve.setData(x, np.array(self.epsilons))
        self.steps_curve.setData(x, np.array(self.episode_steps))

    def reset(self):
        self.episode_rewards.clear()
        self.episode_steps.clear()
        self.epsilons.clear()
        self.reward_curve.setData([], [])
        self.epsilon_curve.setData([], [])
        self.steps_curve.setData([], [])
