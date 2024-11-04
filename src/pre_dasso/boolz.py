import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import svd


def hist(data, hdatlen):
    H, B = np.histogram(data, bins=hdatlen, density=False)
    return H, np.array([(B[i] + B[i + 1]) / 2 for i in range(len(B) - 1)])


def ecdf(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    CDF = np.arange(1, n + 1) / n
    return CDF, sorted_data


def randgen(H, B, size):
    # Generate random samples from the histogram (assuming PDF-like behavior)
    cdf = np.cumsum(H) / np.sum(H)
    random_values = np.random.rand(*size)
    indices = np.searchsorted(cdf, random_values)
    return B[indices]


def split_even_chunks(signal, n_ts, chunk_size):
    chunks = []
    start = 0

    # Loop until we reach the last chunk
    while (n_ts - start) > chunk_size:
        # Ensure the chunk is even-length, but not larger than max_chunk_size
        chunk_size = chunk_size if chunk_size % 2 == 0 else chunk_size - 1
        chunks.append(signal[:, start : start + chunk_size])
        start += chunk_size

    # Append the last chunk (which can be odd length)
    if start < n_ts:
        chunks.append(signal[:, start:])

    return chunks


def fft_sample(
    signal,
    n_simu,
):
    n_chnl, n_ts = signal.shape
    len_signal = n_chnl // 100
    n_fft = 2 * int(np.ceil(n_ts / 2))
    idx_nyquist = n_fft // 2
    fft_signal = fft(signal, n_fft, axis=1)
    # simu_magnitude = np.abs(fft_signal)
    # simu_phase = np.angle(fft_signal)
    mag_fft = np.abs(fft_signal[:, : idx_nyquist + 1])
    phs_fft = np.angle(fft_signal[:, : idx_nyquist + 1])

    simu_magnitude = np.zeros((n_chnl, n_fft, n_simu))
    simu_phase = np.zeros((n_chnl, n_fft, n_simu))

    for f in range(idx_nyquist):
        Hmag, Bmag = hist(mag_fft[:, f], len_signal)
        Hphs, Bphs = hist(phs_fft[:, f], len_signal)
        simu_magnitude[:, f, :] = randgen(Hmag, Bmag, (n_chnl, n_simu))
        simu_phase[:, f, :] = randgen(Hphs, Bphs, (n_chnl, n_simu))

    # print(SURR[0][:,:,0])
    simu_magnitude[:, idx_nyquist + 1 :, :] = simu_magnitude[
        :, idx_nyquist - 1 : 0 : -1, :
    ]
    simu_phase[:, idx_nyquist + 1 :, :] = -simu_phase[:, idx_nyquist - 1 : 0 : -1, :]
    complex_simu = simu_magnitude * np.exp(1j * simu_phase)
    simu = ifft(x=complex_simu, n=n_ts, axis=1).real
    return simu


def booleanization(segment, R, list_alpha, n_simu, need_norm, chunk_size):
    n_chnl, n_ts = segment.shape
    # * split chunks
    if n_ts <= chunk_size:
        simu = fft_sample(signal=segment, n_simu=n_simu)
    else:
        simu_chunk = []
        for chunk in split_even_chunks(
            signal=segment, n_ts=n_ts, chunk_size=chunk_size
        ):
            simu_chunk.append(fft_sample(signal=chunk, n_simu=n_simu))

        simu = np.concatenate(simu_chunk, axis=1)

    simu -= np.mean(simu, axis=1, keepdims=True)
    CMsurr = np.zeros((n_chnl, R, n_simu))

    for s in range(n_simu):
        U, S, _ = svd(simu[:, :, s], full_matrices=False)
        CMsurr[:, :, s] = np.sign(U[:, :R]) * (U[:, :R] * S[:R]) ** 2 / n_ts

    if need_norm:
        vrtxen = np.sum(simu**2, axis=1) / n_ts
        CMsurr /= vrtxen[:, np.newaxis, np.newaxis]

    nalpha = len(list_alpha)
    thresh = np.zeros((nalpha, R))
    # print(CMsurr[:,:,0])
    for c in range(R):
        AbsCMsurr = np.abs(CMsurr[:, c, :])
        CDF, bins = ecdf(AbsCMsurr.flatten())

        for a in range(nalpha):
            idx = np.searchsorted(CDF, 1 - list_alpha[a])
            adj = (1 - list_alpha[a] - CDF[idx - 1]) / (CDF[idx] - CDF[idx - 1])
            thresh[a, c] = bins[idx - 1] + adj * (bins[idx] - bins[idx - 1])

    return thresh
