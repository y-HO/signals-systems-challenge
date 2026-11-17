"""
통계적 음성 향상 필터 (머신러닝 미사용)

한 채널 처리 순서
1. 100 Hz 고역통과 필터 (4차 Butterworth, 양방향 → 위상 왜곡 없음)
2. STFT 후, 에너지가 가장 작은 10% 프레임의 평균으로 잡음 스펙트럼(PSD) 추정
3. decision-directed 방식으로 사전 SNR을 구하고 log-MMSE 이득 적용   [Ephraim & Malah 1984, 1985]
4. 하모닉 재생성: 1차 결과를 반파 정류해 음성 하모닉을 되살린 뒤 사전 SNR을 다시 계산
                                                                     [Plapous, Marro, Scalart 2006]
5. 창 길이 1024 / 2048 두 가지로 2~4를 수행하고 결과를 평균
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.special import exp1

# ---------------- 파라미터 (train으로 고르고 validation으로 확인한 값) ----------------
SAMPLE_RATE = 16000  # 입력 샘플링 주파수 (Hz)
HOP = 128  # STFT 프레임 간격 (8 ms)
FFT_SIZES = (1024, 2048)  # 평균할 두 가지 창 길이 (64 ms, 128 ms)
NOISE_FRAME_FRACTION = 0.10  # 잡음 추정에 쓸 "조용한 프레임" 비율
DD_ALPHA = 0.98  # decision-directed 평활 계수
FIRST_GAIN_FLOOR_DB = -20.0  # 1차 이득의 최소값 (너무 깎지 않도록)
FINAL_GAIN_FLOOR_DB = -30.0  # 최종 이득의 최소값
MIN_PRIOR_SNR_DB = -25.0  # 사전 SNR 하한
HARMONIC_RHO = 0.3  # 하모닉 재생성 섞는 비율 (0.3 = 1차 결과 30%, 재생성 70%)
HIGHPASS_HZ = 100.0  # 고역통과 차단 주파수 (바람·엔진 저역 잡음 제거)


def db_to_amplitude(db):
    """dB → 진폭 비율"""
    return 10.0 ** (db / 20.0)


def db_to_power(db):
    """dB → 파워 비율"""
    return 10.0 ** (db / 10.0)


def stft(x, n_fft):
    """단시간 푸리에 변환 (Hann 창)"""
    return signal.stft(
        x,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - HOP,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )[2]


def istft(spec, length, n_fft):
    """역 STFT (overlap-add), 원래 길이로 자름"""
    y = signal.istft(
        spec,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - HOP,
        nfft=n_fft,
        input_onesided=True,
        boundary=True,
    )[1]
    return y[:length]


def highpass(x):
    """100 Hz 아래 저역 잡음 제거 (양방향 필터링이라 위상 지연 없음)"""
    if len(x) <= 30:
        return x
    sos = signal.butter(4, HIGHPASS_HZ, "highpass", fs=SAMPLE_RATE, output="sos")
    return signal.sosfiltfilt(sos, x)


def estimate_noise_psd(power, n_fft):
    """가장 조용한 프레임들의 평균 파워 = 잡음 스펙트럼.
    파일 양 끝의 zero-padding 프레임은 인위적으로 조용하므로 제외한다."""
    edge = n_fft // (2 * HOP)
    n_frames = power.shape[1]
    frames = (
        np.arange(edge, n_frames - edge)
        if n_frames > 2 * edge + 1
        else np.arange(n_frames)
    )
    frame_energy = power[:, frames].mean(axis=0)
    count = max(1, int(np.ceil(NOISE_FRAME_FRACTION * len(frames))))
    quiet = frames[np.argsort(frame_energy, kind="stable")[:count]]
    noise = uniform_filter1d(
        power[:, quiet].mean(axis=1), 3, mode="nearest"
    )  # 주파수 방향 3-bin 평활
    return np.maximum(noise, 1e-12)


def log_mmse_gain(prior_snr, posterior_snr):
    """log-MMSE 이득: G = ξ/(1+ξ) · exp(½·E1(v)),  v = ξ/(1+ξ)·γ"""
    v = np.maximum(prior_snr / (1 + prior_snr) * posterior_snr, 1e-8)
    return prior_snr / (1 + prior_snr) * np.exp(0.5 * exp1(v))


def decision_directed_gain(power, noise):
    """프레임마다 사전 SNR(ξ)을 '이전 프레임 결과'와 '현재 관측'으로 섞어 추정하고 이득 계산"""
    min_prior = db_to_power(MIN_PRIOR_SNR_DB)
    floor = db_to_amplitude(FIRST_GAIN_FLOOR_DB)
    posterior = np.minimum(power / noise[:, None], 1e4)  # 사후 SNR γ = |Y|² / 잡음
    gain = np.empty_like(power)
    previous_speech = np.zeros_like(noise)
    for t in range(power.shape[1]):
        ml = np.maximum(posterior[:, t] - 1, 0)  # 현재 프레임만 본 SNR 추정
        prior = (
            ml if t == 0 else DD_ALPHA * previous_speech / noise + (1 - DD_ALPHA) * ml
        )
        g = np.clip(
            log_mmse_gain(np.maximum(prior, min_prior), posterior[:, t]), floor, 1
        )
        gain[:, t] = g
        previous_speech = g**2 * power[:, t]  # 다음 프레임에서 쓸 음성 파워 추정
    return gain, posterior


def enhance_one_resolution(x, n_fft):
    """창 길이 하나로 log-MMSE + 하모닉 재생성 수행"""
    length = len(x)
    spec = stft(x, n_fft)
    power = np.abs(spec) ** 2
    noise = estimate_noise_psd(power, n_fft)
    gain, posterior = decision_directed_gain(power, noise)

    # 하모닉 재생성: 1차 결과를 반파 정류(max(s,0))하면 비선형성 때문에
    # 기본 주파수의 정수배(하모닉)가 다시 생긴다 → 깎여 나간 하모닉 복원
    first = istft(gain * spec, length, n_fft)
    harmonics = stft(np.maximum(first, 0), n_fft)[:, : spec.shape[1]]
    prior = (
        HARMONIC_RHO * np.abs(gain * spec) ** 2
        + (1 - HARMONIC_RHO) * np.abs(harmonics) ** 2
    ) / noise[:, None]
    prior = np.maximum(prior, db_to_power(MIN_PRIOR_SNR_DB))
    final_gain = np.clip(
        log_mmse_gain(prior, posterior), db_to_amplitude(FINAL_GAIN_FLOOR_DB), 1
    )
    return istft(final_gain * spec, length, n_fft)


def enhance(noisy):
    """모노 16 kHz 신호의 잡음 제거. 출력 길이는 입력과 정확히 같다."""
    x = highpass(np.asarray(noisy, dtype=np.float64))
    length = len(x)
    if length < max(FFT_SIZES):  # 아주 짧은 입력 대비
        x = np.pad(x, (0, max(FFT_SIZES) - length))
    outputs = [enhance_one_resolution(x, n) for n in FFT_SIZES]
    return np.mean(outputs, axis=0)[:length]
