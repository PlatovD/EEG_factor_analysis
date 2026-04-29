import numpy as np
from pylsl import StreamInlet, resolve_streams


def compute_wavelet(data, fs, freqs):
    w = 5.0
    N = len(data)
    t = np.arange(-N // 2, N - N // 2) / fs
    cwt_matrix = np.zeros((len(freqs), N), dtype=complex)

    for i, f in enumerate(freqs):
        s = w / (2 * np.pi * f)
        wavelet = np.exp(2j * np.pi * f * t) * np.exp(-(t ** 2) / (2 * s ** 2))
        cwt_matrix[i, :] = np.convolve(data, wavelet, mode='same')

    return np.abs(cwt_matrix)


def get_power_matrix(data_dict, fs, f_min=1, f_max=50):
    freqs = np.linspace(f_min, f_max, f_max - f_min + 1)

    channels = sorted(data_dict.keys())
    power_matrix = []

    for ch_idx in channels:
        data = data_dict[ch_idx]
        data = data - np.mean(data)
        cwt = compute_wavelet(data, fs, freqs)
        spectrum = np.mean(cwt, axis=1)
        power_matrix.append(spectrum)

    return np.array(power_matrix), freqs


def main():
    analyzer = NonLinearFactorAnalyzer()
    streams = resolve_streams()

    if not streams:
        return

    if len(streams) > 1:
        try:
            idx = int(input(f"\nВыберите поток (0-{len(streams) - 1}): "))
        except:
            idx = 0

    info = streams[idx]
    fs = int(info.nominal_srate())
    if fs <= 0:
        fs = 250

    inlet = StreamInlet(info)

    buffer_size = fs * 2
    data_buffers = {}
    sample_count = 0

    try:
        while True:
            chunk, timestamps = inlet.pull_chunk(timeout=0.02, max_samples=256)

            if not chunk:
                continue

            for sample, ts in zip(chunk, timestamps):
                for ch_idx, value in enumerate(sample):
                    if ch_idx not in data_buffers:
                        data_buffers[ch_idx] = []
                    data_buffers[ch_idx].append(value)

                    if len(data_buffers[ch_idx]) > buffer_size:
                        data_buffers[ch_idx] = data_buffers[ch_idx][-buffer_size:]

                sample_count += 1

                if sample_count >= fs // 2:
                    sample_count = 0

                    data_copy = {}
                    for ch_idx, buf in data_buffers.items():
                        if len(buf) >= buffer_size:
                            data_copy[ch_idx] = np.array(buf[-buffer_size:])

                    if data_copy:
                        power_matrix, freqs = get_power_matrix(data_copy, fs)
                        analyzer.fit_transform(power_matrix)
                        print(analyzer.get_factors())


    except KeyboardInterrupt:
        print("\nОстановлено пользователем")


if __name__ == "__main__":
    main()
