from pylsl import StreamInlet
from pylsl.resolve import resolve_stream
from logger import app_logger


class EEGStream:
    def __init__(self):
        self.stream = None
        self.inlet: StreamInlet = None

    def connect(self):
        try:
            self.stream = resolve_stream('type', 'EEG')
            self.inlet = StreamInlet(self.stream)
        except Exception as e:
            app_logger.critical(f"Unable to connect. Error {e}")
            raise e

    def next_chunk(self) -> ([], float):
        sample, timestamp = self.inlet.pull_sample()
        return sample, timestamp

    def get_next_amplitudes(self, n: int):
        if n <= 0:
            app_logger.warn(f"Wrong amplitudes cnt - {n}")
            return [[]]
        matrix = []
        for _ in range(n):
            sample, timestamp = self.next_chunk()
            matrix.append(sample)
        return matrix
