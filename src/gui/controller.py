from PySide6.QtCore import QThread, Signal

from lib.factor_analysis import NonLinearFactorAnalyzer
from util.logger import app_logger
from util.reader import EEGStream, MockStream
from enum import Enum


class TypeFactor(Enum):
    RAW = 1
    NON_LINEAR = 2
    LINEAR = 3


class Controller(QThread):
    data_received = Signal(list)

    def __init__(self):
        super().__init__()
        self.stream = MockStream()
        self.running = False
        self.current_type = TypeFactor.RAW
        self.factor_nonlinear_analyser = NonLinearFactorAnalyzer()

    def connect_LSL(self):
        try:
            self.stream.connect()
        except Exception as e:
            raise e

    def run(self):
        self.running = True
        while self.running:
            try:
                matrix = self.stream.get_next_amplitudes(20)
                if self.current_type == TypeFactor.NON_LINEAR:
                    matrix = self.factor_nonlinear_analyser.fit_transform(matrix)

                self.data_received.emit(matrix)
            except Exception:
                app_logger.warn('EEGStream error')
                self.running = False

    def stop(self):
        self.running = False
        self.wait()
