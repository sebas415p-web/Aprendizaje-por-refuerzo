from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                   QPushButton, QTableWidget, QTableWidgetItem,
                                   QFileDialog, QLabel, QHeaderView, QMessageBox)
from PyQt6.QtCore import Qt
from environment import ACTION_NAMES


class QTableTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._agent = None

        layout = QVBoxLayout(self)

        self._status = QLabel("Sin Q-table cargada. Ejecute un entrenamiento o cargue una existente.")
        self._status.setStyleSheet("color: #888; font-size: 13px; padding: 6px;")
        layout.addWidget(self._status)

        btn_layout = QHBoxLayout()

        self._export_txt_btn = QPushButton("Exportar .txt")
        self._export_txt_btn.setEnabled(False)
        self._export_txt_btn.clicked.connect(self._export_txt)
        btn_layout.addWidget(self._export_txt_btn)

        self._load_btn = QPushButton("Cargar Q-table (.pkl)")
        self._load_btn.clicked.connect(self._load_qtable)
        btn_layout.addWidget(self._load_btn)

        self._refresh_btn = QPushButton("Refrescar desde agente")
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.clicked.connect(self._refresh_from_agent)
        btn_layout.addWidget(self._refresh_btn)

        layout.addLayout(btn_layout)

        self._table = QTableWidget()
        self._table.setColumnCount(12)
        self._table.setHorizontalHeaderLabels(
            ["x", "y", "llave", "puerta", "bola", "estados_visitados"]
            + ACTION_NAMES
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        self._info = QLabel("")
        self._info.setStyleSheet("color: #aaa; font-size: 12px; padding: 4px;")
        layout.addWidget(self._info)

        self.style_buttons()

    def style_buttons(self):
        style = """
            QPushButton {
                background-color: #6272a4; color: white;
                border: none; padding: 8px 14px; border-radius: 4px;
                font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #7082b4; }
            QPushButton:disabled { background-color: #444; color: #888; }
        """
        for btn in [self._export_txt_btn, self._load_btn, self._refresh_btn]:
            btn.setStyleSheet(style)

    def set_agent(self, agent):
        self._agent = agent
        has_q = agent is not None and len(agent.Q) > 0
        self._export_txt_btn.setEnabled(has_q)
        self._refresh_btn.setEnabled(agent is not None)

        if agent is not None and len(agent.Q) > 0:
            self._status.setText(f"Q-table activa: {len(agent.Q)} estados, "
                                 f"epsilon={agent.epsilon:.4f}")
        else:
            self._status.setText("Sin Q-table. Inicie un entrenamiento.")

    def _refresh_from_agent(self):
        if self._agent is None:
            return
        self._populate_table(self._agent.Q)

    def _populate_table(self, q_dict):
        if not q_dict:
            self._table.setRowCount(0)
            self._info.setText("Q-table vacía.")
            return

        items = sorted(q_dict.items())
        self._table.setRowCount(len(items))

        for i, (state, values) in enumerate(items):
            self._table.setItem(i, 0, QTableWidgetItem(str(state[0])))
            self._table.setItem(i, 1, QTableWidgetItem(str(state[1])))
            self._table.setItem(i, 2, QTableWidgetItem(str(state[2])))
            self._table.setItem(i, 3, QTableWidgetItem(str(state[3])))
            self._table.setItem(i, 4, QTableWidgetItem(str(state[4])))
            self._table.setItem(i, 5, QTableWidgetItem(str(int(sum(1 for v in values if v != 0)))))

            for a in range(6):
                val = float(values[a])
                item = QTableWidgetItem(f"{val:.3f}")
                if val == max(values):
                    item.setBackground(Qt.GlobalColor.darkGreen)
                self._table.setItem(i, 6 + a, item)

        self._info.setText(f"{len(items)} estados en la Q-table")

    def _export_txt(self):
        if self._agent is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Q-table como texto", "qtable.txt",
            "Texto (*.txt);;All Files (*)"
        )
        if not path:
            return
        lines = []
        header = f"{'x':>3} {'y':>3} {'llave':>6} {'puerta':>7} {'bola':>5} |"
        for a_name in ACTION_NAMES:
            header += f" {a_name:>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for state, values in sorted(self._agent.Q.items()):
            row = f"{state[0]:3d} {state[1]:3d} {state[2]:6d} {state[3]:7d} {state[4]:5d} |"
            for a in range(len(ACTION_NAMES)):
                row += f" {float(values[a]):8.3f}"
            lines.append(row)

        lines.append(f"\nTotal estados: {len(self._agent.Q)}")
        lines.append(f"Alpha: {self._agent.alpha}, Gamma: {self._agent.gamma}")
        lines.append(f"Epsilon: {self._agent.epsilon:.6f}")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        QMessageBox.information(self, "Exportado", f"Q-table guardada en:\n{path}")

    def _load_qtable(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Q-table", "", "Pickle (*.pkl);;All Files (*)"
        )
        if not path:
            return
        from agent import QLearningAgent
        agent = QLearningAgent()
        agent.load_qtable(path)
        self._agent = agent
        self._populate_table(agent.Q)
        self._export_txt_btn.setEnabled(True)
        self._refresh_btn.setEnabled(True)
        self._status.setText(f"Q-table cargada: {len(agent.Q)} estados, epsilon={agent.epsilon:.4f}")
        QMessageBox.information(self, "Cargado", f"Q-table cargada desde:\n{path}")
