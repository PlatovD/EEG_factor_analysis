import sys
from dataclasses import dataclass

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QApplication, QPushButton, QMessageBox

from gui.containers import FactorAnalysisScene, RawSignalScene, CustomPanel
from gui.controller import Controller, TypeFactor

pathToCss = "static/style.qss"


@dataclass
class GUIStats:
    is_lsl_enabled = False
    active_panel: CustomPanel = None
    type_analysis = TypeFactor.RAW


class GUI(QMainWindow):
    def readCss(self, app):
        with open(pathToCss, "r") as f:
            app.setStyleSheet(f.read())

    def __init__(self):
        self.controller = Controller()
        self.controller.data_received.connect(self.update_active_panel)
        self.stats = GUIStats()

        super().__init__()
        self.setWindowTitle("EEG LSL Analyzer")
        self.resize(800, 600)
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.raw_scene = RawSignalScene()
        self.factor_scene = FactorAnalysisScene()
        self.stacked_widget.addWidget(self.raw_scene)
        self.stacked_widget.addWidget(self.factor_scene)
        self._create_menu()

        self.stats.active_panel = self.raw_scene

    def update_active_panel(self, matrix):
        active_panel = self.stats.active_panel
        active_panel.update_data(matrix)

    def handler_lsl_button(self):
        if self.stats.is_lsl_enabled:
            self.controller.stop()
            self.stats.is_lsl_enabled = False
            self._apply_button_style()
        else:
            try:
                self.controller.connect_LSL()
                self.stats.is_lsl_enabled = True
                self._apply_button_style()
            except Exception as e:
                self._show_error(e)

            self.controller.start()

    def handler_row_data(self):
        self.handler_factor_raw_data()
        if self.stats.active_panel == self.raw_scene:
            return
        self.stacked_widget.setCurrentIndex(0)
        self.stats.active_panel = self.raw_scene
        if self.stats.is_lsl_enabled:
            self.handler_lsl_button()
            self.handler_lsl_button()

    def handler_factor_data(self):
        if self.stats.active_panel == self.factor_scene:
            return
        self.stacked_widget.setCurrentIndex(1)
        self.stats.active_panel = self.factor_scene
        if self.stats.is_lsl_enabled:
            self.handler_lsl_button()
            self.handler_lsl_button()

    def handler_factor_linear_data(self):
        self.stats.type_analysis = TypeFactor.LINEAR
        self.controller.current_type = self.stats.type_analysis

    def handler_factor_nonlinear_data(self):
        self.stats.type_analysis = TypeFactor.NON_LINEAR
        self.controller.current_type = self.stats.type_analysis

    def handler_factor_raw_data(self):
        self.stats.type_analysis = TypeFactor.RAW
        self.controller.current_type = self.stats.type_analysis

    def _create_menu(self):
        menu_bar = self.menuBar()
        view_menu = menu_bar.addMenu("Window")
        raw_action = QAction("Initial EEG data", self)
        raw_action.triggered.connect(self.handler_row_data)
        self.stats.active_panel = raw_action
        view_menu.addAction(raw_action)

        factor_action = QAction("Factor Analysis", self)
        factor_action.triggered.connect(self.handler_factor_data)
        view_menu.addAction(factor_action)

        factor_linear_action = QAction("Linear Factor Analysis", self)
        factor_linear_action.triggered.connect(self.handler_factor_linear_data)
        menu_bar.addAction(factor_linear_action)

        factor_non_linear_action = QAction("Non Linear Factor Analysis", self)
        factor_non_linear_action.triggered.connect(self.handler_factor_nonlinear_data)
        menu_bar.addAction(factor_non_linear_action)

        raw_data_action = QAction("Raw data", self)
        raw_data_action.triggered.connect(self.handler_row_data)
        menu_bar.addAction(raw_data_action)

        self.lsl_btn = QPushButton("Start LSL")
        self._apply_button_style()
        self.lsl_btn.clicked.connect(self.handler_lsl_button)
        menu_bar.setCornerWidget(self.lsl_btn)

    def _apply_button_style(self):
        if self.stats.is_lsl_enabled:
            self.lsl_btn.setText("Stop LSL")
        else:
            self.lsl_btn.setText("Start LSL")

    def _show_error(self, message):
        QMessageBox.critical(self, "Ошибка LSL", message)


def main_gui():
    app = QApplication(sys.argv)
    window = GUI()
    window.readCss(app)
    window.show()
    sys.exit(app.exec())
