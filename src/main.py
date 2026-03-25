import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from factor_analysis import NonLinearFactorAnalyzer
from spectral_generator import SpectralDataGenerator

if __name__ == '__main__':
    generator = SpectralDataGenerator(
        n_frequencies=3,
        freq_min=0.5,
        freq_max=45.0,
        seed=42
    )

    X, labels, freqs = generator.generate_test_data(
        n_subjects=3,
        groups=['alpha', 'beta', 'mixed']
    )
    X = np.array([
        [6.6, 3.9, 4.7],
        [7, 5.6, 1.3],
        [5.4, 6.8, 5.1]
    ])

    analyzer = NonLinearFactorAnalyzer(
        n_factors=5,
        kernel='rbf',
        gamma=0.1
    )
    analyzer.fit_transform(X)
    print('Факторы на выходе')
    print(pd.DataFrame(analyzer.get_factors()))
    analyzer.plot_factors_scatter(labels=labels)
    analyzer.plot_factors_table(labels=labels, group_names={0: 'Alpha', 1: 'Beta', 2: 'Mixed'})
