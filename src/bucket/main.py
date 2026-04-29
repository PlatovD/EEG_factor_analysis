import numpy as np
import pandas as pd

from lib.factor_analysis import NonLinearFactorAnalyzer
from spectral_generator import SpectralDataGenerator

if __name__ == '__main__':
    generator = SpectralDataGenerator(
        n_frequencies=3,
        freq_min=0.5,
        freq_max=45.0,
        seed=42
    )

    X = generator.generate_non_linear_data(100)
    Y = generator.generate_full_random(100)
    print(pd.DataFrame(X))

    analyzer = NonLinearFactorAnalyzer(
        n_factors=2,
        kernel='rbf',
        gamma=0.1
    )
    analyzer_on_random = NonLinearFactorAnalyzer(
        n_factors=2,
        kernel='rbf',
        gamma=0.1
    )
    analyzer.fit_transform(X)
    analyzer_on_random.fit_transform(Y)
    print('Факторы на выходе')
    print(pd.DataFrame(analyzer.get_factors()))
    print('Факторы для рандома')
    print(pd.DataFrame(analyzer_on_random.factors_))

    X_transformed = analyzer.get_factors()
    dot_products = np.dot(X_transformed.T, X_transformed)
    print(pd.DataFrame(dot_products))
    np.fill_diagonal(dot_products, 0)
    max_correlation = np.max(np.abs(dot_products))

    print(f"Максимальная корреляция между компонентами: {max_correlation:.10f}")
    analyzer.plot_factors_scatter()
    analyzer.plot_factors_table()
    analyzer_on_random.plot_factors_scatter()

    print(f'Объясненная дисперсия {analyzer.get_explained_variance()}')
    print(analyzer.get_eigenvalues())
    print(analyzer.get_eigenvectors())
    print(f'Объясненная дисперсия {analyzer_on_random.get_explained_variance()}')
    print(analyzer_on_random.get_eigenvalues())
    print(analyzer_on_random.get_eigenvectors())
