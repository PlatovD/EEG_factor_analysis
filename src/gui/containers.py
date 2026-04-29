import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
import pyqtgraph as pg


class CustomPanel(QWidget):
    def update_data(self, data):
        pass


class RawSignalScene(CustomPanel):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.label = QLabel("Initial EEG data")
        layout.addWidget(self.label)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.curves = []
        self.offset = 100

    def update_data(self, data):
        if not data: return

        matrix = np.array(data)
        n_samples, n_channels = matrix.shape

        if not self.curves:
            for i in range(n_channels):
                pen = pg.mkPen(color=pg.intColor(i, n_channels), width=1)
                curve = self.plot_widget.plot(pen=pen)
                self.curves.append(curve)

        for i in range(n_channels):
            y_data = matrix[:, i] + (i * self.offset)
            self.curves[i].setData(y_data)


class FactorAnalysisScene(CustomPanel):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.label = QLabel("Factor Analysis")
        layout.addWidget(self.label)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#121212')
        layout.addWidget(self.plot_widget)

        self.curves = []
        self.offset = 5

    def update_data(self, data):
        if not data: return

        matrix = np.array(data)
        n_samples, n_factors = matrix.shape

        if not self.curves:
            for i in range(n_factors):
                pen = pg.mkPen(color=pg.intColor(i, n_factors), width=2)
                curve = self.plot_widget.plot(pen=pen)
                self.curves.append(curve)

        for i in range(n_factors):
            y_data = matrix[:, i] + (i * self.offset)
            self.curves[i].setData(y_data)
