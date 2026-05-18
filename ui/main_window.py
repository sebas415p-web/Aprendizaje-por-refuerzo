from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor
from ui.training_tab import TrainingTab
from ui.qtable_tab import QTableTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Proyecto Final - Aprendizaje por Refuerzo")
        self.resize(1100, 800)
        self._apply_dark_theme()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = self._build_header()
        layout.addWidget(header)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self.setCentralWidget(central)

        self._training_tab = TrainingTab()
        self._tabs.addTab(self._training_tab, "Entrenamiento")

        self._qtable_tab = QTableTab()
        self._tabs.addTab(self._qtable_tab, "Q-Table")

    def _build_header(self):
        w = QWidget()
        w.setStyleSheet("""
            background-color: #1e1e2a; border-bottom: 2px solid #6272a4;
        """)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 10, 16, 8)
        layout.setSpacing(2)

        title = QLabel("Proyecto Final - Aprendizaje por Refuerzo")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #bd93f9; font-size: 18px; font-weight: bold; border: none;")
        layout.addWidget(title)

        subtitle = QLabel("Entrega final")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #6272a4; font-size: 12px; border: none;")
        layout.addWidget(subtitle)

        meta = QLabel("Materia: Aprendizaje por Refuerzo  |  Mayo 2026  |  "
                      "Juan Sebastian Paez Cortes - Jean Betancourt Padilla")
        meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meta.setStyleSheet("color: #888; font-size: 11px; border: none;")
        layout.addWidget(meta)

        return w

    def on_training_finished(self, agent):
        self._qtable_tab.set_agent(agent)
        self._tabs.setCurrentIndex(1)

    def _apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(40, 42, 54))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(248, 248, 242))
        palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 40))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 60))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(40, 42, 54))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(248, 248, 242))
        palette.setColor(QPalette.ColorRole.Text, QColor(248, 248, 242))
        palette.setColor(QPalette.ColorRole.Button, QColor(68, 71, 90))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(248, 248, 242))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 85, 85))
        palette.setColor(QPalette.ColorRole.Link, QColor(139, 233, 253))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(98, 114, 164))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(248, 248, 242))
        self.setPalette(palette)

        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold; border: 1px solid #555; border-radius: 6px;
                margin-top: 12px; padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 4px;
                color: #bd93f9;
            }
            QTabWidget::pane { border: 1px solid #555; background: #282a36; }
            QTabBar::tab {
                background: #44475a; color: #ccc; padding: 8px 20px;
                border: 1px solid #555; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #6272a4; color: white; }
            QDoubleSpinBox, QSpinBox {
                background: #282a36; color: #f8f8f2; border: 1px solid #555;
                padding: 4px; border-radius: 3px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555; height: 6px; background: #444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6272a4; width: 16px; margin: -5px 0;
                border-radius: 8px;
            }
            QProgressBar {
                border: 1px solid #555; border-radius: 3px; text-align: center;
                background: #282a36; color: #f8f8f2;
            }
            QProgressBar::chunk { background: #50fa7b; border-radius: 2px; }
            QCheckBox { color: #f8f8f2; }
            QTableWidget {
                background: #282a36; color: #f8f8f2; gridline-color: #555;
                border: 1px solid #555;
            }
            QTableWidget::item:selected { background: #6272a4; }
            QHeaderView::section {
                background: #44475a; color: #f8f8f2; border: 1px solid #555;
                padding: 4px; font-weight: bold;
            }
            QSplitter::handle { background: #555; width: 2px; }
        """)
