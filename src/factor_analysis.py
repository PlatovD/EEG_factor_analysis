import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.table import Table
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


class NonLinearFactorAnalyzer:
    def __init__(self, n_factors: int = 5, kernel: str = 'rbf', gamma: float = 0.1):
        self.n_factors = n_factors
        self.kernel = kernel
        self.gamma = gamma
        self.scaler = None
        self.kpca = None
        self.factors_ = None
        self.data_ = None

    def fit(self, data: np.ndarray):
        self.data_ = data
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data)

        self.kpca = KernelPCA(
            n_components=self.n_factors,
            kernel=self.kernel,
            gamma=self.gamma
        )
        self.kpca.fit(data_scaled)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.kpca is None:
            raise ValueError("Model not fitted. You need to call fit first")

        data_scaled = self.scaler.transform(data)
        self.factors_ = self.kpca.transform(data_scaled)
        return self.factors_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def get_factors(self) -> np.ndarray:
        if self.factors_ is None:
            raise ValueError("No factors available. Call transform() or fit_transform() first")
        return self.factors_

    def get_explained_variance(self) -> np.ndarray:
        if self.kpca is None:
            raise ValueError("Model not fitted")

        if hasattr(self.kpca, 'eigenvalues_') and self.kpca.eigenvalues_ is not None:
            ev = self.kpca.eigenvalues_
            explained = ev / np.sum(ev)
            return explained[:self.n_factors]
        else:
            return np.array([0.0] * self.n_factors)

    def inverse_transform(self, factors: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.kpca is None:
            raise ValueError("Model not fitted")

        reconstructed_scaled = self.kpca.inverse_transform(factors)
        reconstructed = self.scaler.inverse_transform(reconstructed_scaled)
        return reconstructed

    def plot_factors_table(self, labels=None, group_names=None):
        if self.factors_ is None:
            raise ValueError("No factors available. Run fit_transform first")

        df = pd.DataFrame({
            'Subject': range(1, len(self.factors_) + 1),
            'F1': self.factors_[:, 0].round(3),
            'F2': self.factors_[:, 1].round(3),
        })

        if labels is not None:
            if group_names is None:
                group_names = {0: 'Alpha', 1: 'Beta', 2: 'Mixed'}
            df['Group'] = [group_names.get(l, str(l)) for l in labels]

        group_colors = {
            'Alpha': '#3498db',
            'Beta': '#e74c3c',
            'Mixed': '#2ecc71'
        }

        n = len(df)

        if n <= 15:
            indices = list(range(n))
        else:
            indices = []
            indices.extend(range(5))
            indices.extend(range(n // 2 - 2, n // 2 + 3))
            indices.extend(range(n - 5, n))
            indices = sorted(set(indices))

        df_display = df.iloc[indices]

        fig, ax = plt.subplots(figsize=(12, len(df_display) * 0.4 + 2))
        ax.axis('off')

        table = Table(ax, bbox=[0, 0, 1, 1])

        for col, col_name in enumerate(df_display.columns):
            cell = table.add_cell(0, col, width=0.15, height=0.1,
                                  text=col_name, loc='center',
                                  facecolor='#2c3e50', edgecolor='white')
            cell.set_fontsize(11)
            cell.set_text_props(fontweight='bold', color='white')

        for i, (idx, row) in enumerate(df_display.iterrows()):
            for j, val in enumerate(row):
                if labels is not None and j == len(row) - 1:
                    group = str(val)
                    bg = group_colors.get(group, '#95a5a6')
                    text_color = 'white'
                else:
                    bg = '#f8f9fa' if i % 2 == 0 else 'white'
                    text_color = 'black'

                cell = table.add_cell(i + 1, j, width=0.15, height=0.08,
                                      text=str(val), loc='center',
                                      facecolor=bg, edgecolor='lightgray')
                cell.set_fontsize(9)
                cell.set_text_props(color=text_color)

        table.auto_set_font_size(False)
        ax.add_table(table)

        plt.title(f'Factor analysis results',
                  fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()

        return df

    def plot_factors_scatter(self, labels=None, group_names=None):
        if self.factors_ is None:
            raise ValueError("No factors available. Run fit_transform first")

        plt.figure(figsize=(10, 8))

        if labels is not None:
            colors = {0: '#3498db', 1: '#e74c3c', 2: '#2ecc71'}
            if group_names is None:
                group_names = {0: 'Alpha', 1: 'Beta', 2: 'Mixed'}

            for label in np.unique(labels):
                mask = labels == label
                plt.scatter(self.factors_[mask, 0], self.factors_[mask, 1],
                            c=colors[label], label=group_names[label],
                            alpha=0.7, s=80, edgecolors='black', linewidth=1)
        else:
            plt.scatter(self.factors_[:, 0], self.factors_[:, 1],
                        alpha=0.7, s=80, c='steelblue', edgecolors='black')

        plt.xlabel('Factor 1', fontsize=14)
        plt.ylabel('Factor 2', fontsize=14)
        plt.title('Subjects in Factor Space', fontsize=16)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
