from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np


class SpectralDataGenerator:
    def __init__(
            self,
            n_frequencies: int = 50,
            freq_min: float = 0.5,
            freq_max: float = 45.0,
            seed: int = 42
    ):
        self.n_frequencies = n_frequencies
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.seed = seed

        self.frequencies = np.linspace(freq_min, freq_max, n_frequencies)

        np.random.seed(seed)

    def _gaussian_peak(
            self,
            center: float,
            amplitude: float,
            sigma: float = 2.0
    ) -> np.ndarray:
        return amplitude * np.exp(-((self.frequencies - center) ** 2) / (2 * sigma ** 2))

    def generate_full_random(self, n_samples: int):
        np.random.seed(82)
        features = []
        for _ in range(n_samples):
            features.append(np.random.uniform(-100, 100, n_samples))
        return np.stack(features)

    def generate_non_linear_data(self, n_samples: int):
        if n_samples < 2:
            raise ValueError('Нужно минимум 2 базовых признака')

        np.random.seed(42)

        latent_factors = []
        for i in range(n_samples):
            factor = np.random.uniform(-2, 2, n_samples)
            latent_factors.append(factor)

        features = []
        feature1 = 2.5 * latent_factors[0] + 1.5 * latent_factors[1]
        features.append(feature1)
        feature2 = 1.8 * latent_factors[1] - 2.0 * latent_factors[2] if n_samples > 2 else 1.8 * latent_factors[1]
        features.append(feature2)
        feature3 = latent_factors[0] * latent_factors[1]
        features.append(feature3)

        feature4 = latent_factors[0] ** 2 + 0.5 * latent_factors[1] * latent_factors[2] if n_samples > 2 else \
            latent_factors[0] ** 2
        features.append(feature4)

        feature5 = np.sin(latent_factors[0] + latent_factors[1])
        features.append(feature5)

        for i in range(len(features)):
            features[i] += np.random.normal(0, 0, n_samples)

        X = np.column_stack(features)
        self.latent_factors_ = np.column_stack(latent_factors)
        return X

    def generate_alpha_dominant(
            self,
            n_samples: int,
            alpha_center: float = 10.0,
            alpha_amplitude: float = 10.0,
            alpha_sigma: float = 2.0,
            noise_level: float = 1.0,
            baseline: float = 5.0
    ) -> np.ndarray:
        X = np.zeros((n_samples, self.n_frequencies))

        for i in range(n_samples):
            alpha_peak = self._gaussian_peak(alpha_center, alpha_amplitude, alpha_sigma)
            noise = np.random.randn(self.n_frequencies) * noise_level
            X[i, :] = alpha_peak + noise + baseline

        return X

    def generate_beta_dominant(
            self,
            n_samples: int,
            beta_center: float = 20.0,
            beta_amplitude: float = 12.0,
            beta_sigma: float = 2.0,
            noise_level: float = 1.0,
            baseline: float = 5.0
    ) -> np.ndarray:
        X = np.zeros((n_samples, self.n_frequencies))

        for i in range(n_samples):
            beta_peak = self._gaussian_peak(beta_center, beta_amplitude, beta_sigma)
            noise = np.random.randn(self.n_frequencies) * noise_level
            X[i, :] = beta_peak + noise + baseline

        return X

    def generate_mixed(
            self,
            n_samples: int,
            alpha_center: float = 10.0,
            beta_center: float = 20.0,
            alpha_amplitude: float = 8.0,
            beta_amplitude: float = 8.0,
            alpha_sigma: float = 2.0,
            beta_sigma: float = 2.0,
            noise_level: float = 1.0,
            baseline: float = 5.0
    ) -> np.ndarray:
        X = np.zeros((n_samples, self.n_frequencies))

        for i in range(n_samples):
            alpha_peak = self._gaussian_peak(alpha_center, alpha_amplitude, alpha_sigma)
            beta_peak = self._gaussian_peak(beta_center, beta_amplitude, beta_sigma)
            noise = np.random.randn(self.n_frequencies) * noise_level
            X[i, :] = alpha_peak + beta_peak + noise + baseline

        return X

    def generate_theta_dominant(
            self,
            n_samples: int,
            theta_center: float = 6.0,
            theta_amplitude: float = 12.0,
            theta_sigma: float = 1.5,
            noise_level: float = 1.0,
            baseline: float = 5.0
    ) -> np.ndarray:
        X = np.zeros((n_samples, self.n_frequencies))

        for i in range(n_samples):
            theta_peak = self._gaussian_peak(theta_center, theta_amplitude, theta_sigma)
            noise = np.random.randn(self.n_frequencies) * noise_level
            X[i, :] = theta_peak + noise + baseline

        return X

    def generate_delta_dominant(
            self,
            n_samples: int,
            delta_center: float = 2.0,
            delta_amplitude: float = 15.0,
            delta_sigma: float = 1.0,
            noise_level: float = 1.0,
            baseline: float = 5.0
    ) -> np.ndarray:
        X = np.zeros((n_samples, self.n_frequencies))

        for i in range(n_samples):
            delta_peak = self._gaussian_peak(delta_center, delta_amplitude, delta_sigma)
            noise = np.random.randn(self.n_frequencies) * noise_level
            X[i, :] = delta_peak + noise + baseline

        return X

    def generate_gamma_dominant(
            self,
            n_samples: int,
            gamma_center: float = 40.0,
            gamma_amplitude: float = 5.0,
            gamma_sigma: float = 3.0,
            noise_level: float = 1.0,
            baseline: float = 5.0
    ) -> np.ndarray:
        X = np.zeros((n_samples, self.n_frequencies))

        for i in range(n_samples):
            gamma_peak = self._gaussian_peak(gamma_center, gamma_amplitude, gamma_sigma)
            noise = np.random.randn(self.n_frequencies) * noise_level
            X[i, :] = gamma_peak + noise + baseline

        return X

    def generate_test_data(
            self,
            n_subjects: int = 100,
            groups: List[str] = None,
            group_sizes: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if groups is None:
            groups = ['alpha', 'beta', 'mixed']

        n_groups = len(groups)

        if group_sizes is None:
            base_size = n_subjects // n_groups
            remainder = n_subjects % n_groups
            group_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_groups)]

        X_list = []
        labels_list = []

        generators = {
            'alpha': self.generate_alpha_dominant,
            'beta': self.generate_beta_dominant,
            'mixed': self.generate_mixed,
            'theta': self.generate_theta_dominant,
            'delta': self.generate_delta_dominant,
            'gamma': self.generate_gamma_dominant
        }

        for group_idx, (group_type, size) in enumerate(zip(groups, group_sizes)):
            if group_type not in generators:
                raise ValueError(f"Неизвестный тип группы: {group_type}")

            X_group = generators[group_type](n_samples=size)
            X_list.append(X_group)
            labels_list.append(np.full(size, group_idx))

        X = np.vstack(X_list)
        labels = np.concatenate(labels_list)
        X += np.random.randn(*X.shape) * 2
        X = np.abs(X)

        return X, labels, self.frequencies

    def plot_spectral_profiles(
            self,
            X: np.ndarray,
            labels: np.ndarray,
            n_samples: int = 9,
            group_names: dict = None
    ):
        if group_names is None:
            group_names = {0: 'Альфа', 1: 'Бета', 2: 'Смешанные',
                           3: 'Тета', 4: 'Дельта', 5: 'Гамма'}

        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()

        indices = []
        indices.extend(range(3))
        indices.extend(range(X.shape[0] // 2, X.shape[0] // 2 + 3))
        indices.extend(range(X.shape[0] - 3, X.shape[0]))
        for idx, ax in zip(indices, axes):
            if idx < n_samples and idx < X.shape[0]:
                subject = idx
                label = labels[subject]
                ax.plot(self.frequencies, X[subject, :],
                        color=colors[label % len(colors)], linewidth=1.5)
                ax.set_title(f'Испытуемый {subject} ({group_names.get(label, label)})', fontsize=10)
                ax.set_xlabel('Частота (Гц)')
                ax.set_ylabel('Мощность')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(self.freq_min, self.freq_max)

        plt.suptitle('Спектральные профили мощности', fontsize=14)
        plt.tight_layout()
        plt.show()

    def get_info(self) -> dict:
        return {
            'n_frequencies': self.n_frequencies,
            'freq_range': (self.freq_min, self.freq_max),
            'seed': self.seed
        }


if __name__ == "__main__":
    generator = SpectralDataGenerator(
        n_frequencies=50,
        freq_min=0.5,
        freq_max=45.0,
        seed=42
    )

    X1, labels1, freqs = generator.generate_test_data(
        n_subjects=150,
        groups=['alpha', 'beta', 'mixed']
    )

    generator.plot_spectral_profiles(X1, labels1, n_samples=9, group_names={})
