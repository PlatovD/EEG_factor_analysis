import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


class EEGSimulator:
    def __init__(
            self,
            channels: int = 32,
            sfreq: int = 250,
            duration_sec: float = 60.0,
            seed: int = 42
    ):
        self.channels = channels
        self.sfreq = sfreq
        self.duration_sec = duration_sec
        self.seed = seed
        self.n_samples = int(duration_sec * sfreq)
        self.t = np.arange(self.n_samples) / sfreq
        np.random.seed(seed)

    def generate_background(self, amplitude: float = 10.0) -> np.ndarray:
        return np.random.randn(self.channels, self.n_samples) * amplitude

    def add_alpha_rhythm(
            self,
            signal: np.ndarray,
            channels: list,
            frequency: float = 10.0,
            amplitude: float = 20.0
    ) -> np.ndarray:
        """
        Добавляет альфа-ритм (8-13 Гц)
        """
        alpha = amplitude * np.sin(2 * np.pi * frequency * self.t)
        for ch in channels:
            signal[ch, :] += alpha
        return signal

    def add_beta_rhythm(
            self,
            signal: np.ndarray,
            channels: list,
            frequency: float = 20.0,
            amplitude: float = 15.0
    ) -> np.ndarray:
        """
        Добавляет бета-ритм (13-30 Гц)
        """
        beta = amplitude * np.sin(2 * np.pi * frequency * self.t)
        for ch in channels:
            signal[ch, :] += beta
        return signal

    def add_theta_rhythm(
            self,
            signal: np.ndarray,
            channels: list,
            frequency: float = 6.0,
            amplitude: float = 25.0
    ) -> np.ndarray:
        """
        Добавляет тета-ритм (4-8 Гц)
        """
        theta = amplitude * np.sin(2 * np.pi * frequency * self.t)
        for ch in channels:
            signal[ch, :] += theta
        return signal

    def add_delta_rhythm(
            self,
            signal: np.ndarray,
            channels: list,
            frequency: float = 2.0,
            amplitude: float = 30.0
    ) -> np.ndarray:
        """
        Добавляет дельта-ритм (0.5-4 Гц)
        """
        delta = amplitude * np.sin(2 * np.pi * frequency * self.t)
        for ch in channels:
            signal[ch, :] += delta
        return signal

    def add_gamma_rhythm(
            self,
            signal: np.ndarray,
            channels: list,
            frequency: float = 40.0,
            amplitude: float = 5.0
    ) -> np.ndarray:
        """
        Добавляет гамма-ритм (30-45 Гц)
        """
        gamma = amplitude * np.sin(2 * np.pi * frequency * self.t)
        for ch in channels:
            signal[ch, :] += gamma
        return signal

    def add_background_rhythms(
            self,
            signal: np.ndarray,
            background_rhythm_amplitude: float = 3.0
    ) -> ndarray:
        all_channels = [_ for _ in range(0, self.channels)]
        signal = self.add_alpha_rhythm(signal, all_channels, amplitude=background_rhythm_amplitude)
        signal = self.add_beta_rhythm(signal, all_channels, amplitude=background_rhythm_amplitude)
        signal = self.add_theta_rhythm(signal, all_channels, amplitude=background_rhythm_amplitude)
        signal = self.add_gamma_rhythm(signal, all_channels, amplitude=background_rhythm_amplitude)
        signal = self.add_delta_rhythm(signal, all_channels, amplitude=background_rhythm_amplitude)
        return signal

    def generate_simple_eeg(
            self,
            alpha_channels: list = None,
            beta_channels: list = None,
            background_amplitude: float = 10.0,
            alpha_amplitude: float = 20.0,
            beta_amplitude: float = 15.0
    ) -> np.ndarray:
        if alpha_channels is None:
            alpha_channels = list(range(min(4, self.channels)))
        if beta_channels is None:
            beta_channels = list(range(min(4, self.channels), min(8, self.channels)))

        signal = self.generate_background(background_amplitude)
        signal = self.add_background_rhythms(signal)

        signal = self.add_alpha_rhythm(signal, alpha_channels, amplitude=alpha_amplitude)
        signal = self.add_beta_rhythm(signal, beta_channels, amplitude=beta_amplitude)

        return signal

    def generate_full_eeg(
            self,
            rhythm_config: dict = None
    ) -> np.ndarray:
        signal = self.generate_background(10.0)
        signal = self.add_background_rhythms(signal)

        if rhythm_config is None:
            rhythm_config = {
                'alpha': {'channels': list(range(min(4, self.channels))), 'frequency': 10, 'amplitude': 20},
                'beta': {'channels': list(range(min(4, self.channels), min(8, self.channels))), 'frequency': 20,
                         'amplitude': 15}
            }

        for rhythm, params in rhythm_config.items():
            if rhythm == 'alpha':
                signal = self.add_alpha_rhythm(signal, params['channels'], params.get('frequency', 10),
                                               params.get('amplitude', 20))
            elif rhythm == 'beta':
                signal = self.add_beta_rhythm(signal, params['channels'], params.get('frequency', 20),
                                              params.get('amplitude', 15))
            elif rhythm == 'theta':
                signal = self.add_theta_rhythm(signal, params['channels'], params.get('frequency', 6),
                                               params.get('amplitude', 25))
            elif rhythm == 'delta':
                signal = self.add_delta_rhythm(signal, params['channels'], params.get('frequency', 2),
                                               params.get('amplitude', 30))
            elif rhythm == 'gamma':
                signal = self.add_gamma_rhythm(signal, params['channels'], params.get('frequency', 40),
                                               params.get('amplitude', 5))

        return signal

    def plot_signal(self, signal: np.ndarray, channels: list = None, duration: float = 5.0):
        if channels is None:
            channels = list(range(min(4, self.channels)))

        samples_to_plot = int(duration * self.sfreq)
        t_plot = self.t[:samples_to_plot]

        plt.figure(figsize=(12, 6))
        for i, ch in enumerate(channels):
            plt.subplot(len(channels), 1, i + 1)
            plt.plot(t_plot, signal[ch, :samples_to_plot])
            plt.ylabel(f'Канал {ch}. мкВ')
            if i == len(channels) - 1:
                plt.xlabel('Время (сек)')
            plt.grid(True, alpha=0.3)

        plt.suptitle('ЭЭГ сигнал')
        plt.tight_layout()
        plt.show()

    def get_info(self) -> dict:
        return {
            'channels': self.channels,
            'sfreq': self.sfreq,
            'duration_sec': self.duration_sec,
            'n_samples': self.n_samples,
            'seed': self.seed
        }


if __name__ == "__main__":
    eeg = EEGSimulator(channels=32, sfreq=250, duration_sec=10, seed=42)
    print(eeg.get_info())

    # случайное ЭЭГ
    signal1 = eeg.generate_simple_eeg()
    eeg.plot_signal(signal1, channels=[0, 4], duration=2)

    # заданное ЭЭГ
    config = {
        'alpha': {'channels': [0, 1, 2, 3], 'frequency': 10, 'amplitude': 25},
        'beta': {'channels': [4, 5, 6, 7], 'frequency': 20, 'amplitude': 20},
        'theta': {'channels': [8, 9], 'frequency': 6, 'amplitude': 30}
    }
    signal2 = eeg.generate_full_eeg(config)
    eeg.plot_signal(signal2, channels=[0, 4, 8], duration=2)
