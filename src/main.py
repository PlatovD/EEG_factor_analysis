from factor_analysis import NonLinearFactorAnalyzer
from spectral_generator import SpectralDataGenerator

if __name__ == '__main__':
    generator = SpectralDataGenerator(
        n_frequencies=50,
        freq_min=0.5,
        freq_max=45.0,
        seed=42
    )

    X, labels, freqs = generator.generate_test_data(
        n_subjects=150,
        groups=['alpha', 'beta', 'mixed']
    )

    generator.plot_spectral_profiles(X, labels, n_samples=150)

    analyzer = NonLinearFactorAnalyzer(
        n_factors=5,
        kernel='rbf',
        gamma=0.1
    )

    analyzer.fit_transform(X)
    analyzer.plot_factors_scatter(labels=labels)
    analyzer.plot_factors_table(labels=labels)
