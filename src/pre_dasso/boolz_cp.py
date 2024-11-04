import cupy as cp
from cupy.fft import fft, ifft
from cupy.linalg import svd
from loguru import logger


def hist(data, hdatlen, gpu_id):
    with cp.cuda.Device(gpu_id):
        H, B = cp.histogram(data, bins=hdatlen, density=False)
        return H, cp.array([(B[i] + B[i + 1]) / 2 for i in range(len(B) - 1)])


def ecdf(data, gpu_id):
    with cp.cuda.Device(gpu_id):
        sorted_data = cp.sort(data)
        n = len(sorted_data)
        CDF = cp.arange(1, n + 1) / n
        return CDF, sorted_data


def randgen(H, B, size, gpu_id):
    with cp.cuda.Device(gpu_id):
        # Generate random samples from the histogram (assuming PDF-like behavior)
        cdf = cp.cumsum(H) / cp.sum(H)
        random_values = cp.random.rand(*size)
        indices = cp.searchsorted(cdf, random_values)
        return B[indices]


def split_even_chunks(signal, n_ts, chunk_size, gpu_id):
    with cp.cuda.Device(gpu_id):
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


def fft_sample(signal, n_simu, gpu_id):
    with cp.cuda.Device(gpu_id):
        n_chnl, n_ts = signal.shape
        len_signal = n_chnl // 100
        n_fft = 2 * int(cp.ceil(n_ts / 2))
        idx_nyquist = n_fft // 2
        fft_signal = fft(signal, n_fft, axis=1)
        mag_fft = cp.abs(fft_signal[:, : idx_nyquist + 1])
        phs_fft = cp.angle(fft_signal[:, : idx_nyquist + 1])

        simu_magnitude = cp.zeros((n_chnl, n_fft, n_simu))
        simu_phase = cp.zeros((n_chnl, n_fft, n_simu))

        for f in range(idx_nyquist):
            Hmag, Bmag = hist(mag_fft[:, f], len_signal, gpu_id)
            Hphs, Bphs = hist(phs_fft[:, f], len_signal, gpu_id)
            simu_magnitude[:, f, :] = randgen(Hmag, Bmag, (n_chnl, n_simu), gpu_id)
            simu_phase[:, f, :] = randgen(Hphs, Bphs, (n_chnl, n_simu), gpu_id)

        simu_magnitude[:, idx_nyquist + 1 :, :] = simu_magnitude[
            :, idx_nyquist - 1 : 0 : -1, :
        ]
        simu_phase[:, idx_nyquist + 1 :, :] = -simu_phase[
            :, idx_nyquist - 1 : 0 : -1, :
        ]
        complex_simu = simu_magnitude * cp.exp(1j * simu_phase)
        simu = ifft(complex_simu, n=n_ts, axis=1).real
        return simu


def booleanization(segment, R, list_alpha, n_simu, need_norm, chunk_size, gpu_id):
    with cp.cuda.Device(gpu_id):
        segment = cp.asarray(segment)
        list_alpha = cp.asarray(list_alpha)
        n_chnl, n_ts = segment.shape
        logger.debug(f"boolz the segment: {segment.shape}")
        # * split chunks
        if n_ts <= chunk_size:
            simu = fft_sample(signal=segment, n_simu=n_simu, gpu_id=gpu_id)
        else:
            simu_chunk = []
            for chunk in split_even_chunks(
                signal=segment, n_ts=n_ts, chunk_size=chunk_size, gpu_id=gpu_id
            ):
                simu_chunk.append(
                    fft_sample(signal=chunk, n_simu=n_simu, gpu_id=gpu_id)
                )

            simu = cp.concatenate(simu_chunk, axis=1)

        simu -= cp.mean(simu, axis=1, keepdims=True)
        CMsurr = cp.zeros((n_chnl, R, n_simu))

        for s in range(n_simu):
            U, S, _ = svd(simu[:, :, s], full_matrices=False)
            CMsurr[:, :, s] = cp.sign(U[:, :R]) * (U[:, :R] * S[:R]) ** 2 / n_ts

        if need_norm:
            vrtxen = cp.sum(simu**2, axis=1) / n_ts
            CMsurr /= vrtxen[:, cp.newaxis, cp.newaxis]

        nalpha = len(list_alpha)
        thresh = cp.zeros((nalpha, R))

        for c in range(R):
            AbsCMsurr = cp.abs(CMsurr[:, c, :])
            CDF, bins = ecdf(AbsCMsurr.flatten(), gpu_id)
            for a in range(nalpha):
                idx = cp.searchsorted(CDF, 1 - list_alpha[a])
                adj = (1 - list_alpha[a] - CDF[idx - 1]) / (CDF[idx] - CDF[idx - 1])
                thresh[a, c] = bins[idx - 1] + adj * (bins[idx] - bins[idx - 1])

        return cp.asnumpy(thresh)
