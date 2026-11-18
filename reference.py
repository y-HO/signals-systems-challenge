"""
참조 잡음 차감 (정합 필터로 잡음 라이브러리에서 같은 잡음을 찾아 빼기)

입력 y에 대해, 라이브러리의 잡음 구간 s와 시간 지연 L 중
정규화 상호상관 |Σ y[t+L]·s[t]| / ||s|| 가 가장 큰 것을 찾는다.
최대 3개 구간을 이어 파일 전체를 덮고, 그 합 d̂의 크기를 최소제곱
(a = <y, d̂> / <d̂, d̂>)으로 맞춘 뒤 입력에서 뺀다.
상호상관은 FFT로 계산한다 (시간 영역 상관 = 주파수 영역 곱).
"""

from pathlib import Path

import numpy as np

FFT_LEN = 1 << 17  # 131072 ≥ 2×48000 → 모든 지연을 순환 겹침 없이 계산
BATCH = 96  # 한 번에 FFT할 라이브러리 구간 수 (메모리와 속도 절충)
MAX_SEGMENTS = 3  # 파일 하나를 덮는 데 쓸 최대 구간 수
MIN_NEW_SAMPLES = 1600  # 새 구간이 최소 0.1초는 새로 덮어야 채택
MIN_CORRELATION = 0.10  # 이보다 상관이 낮으면 "라이브러리에 없는 잡음"으로 판단
LIBRARY_DIR = Path(__file__).resolve().parent / "noise_library"


class NoiseLibrary:
    def __init__(self, folder=LIBRARY_DIR):
        parts = sorted(Path(folder).glob("part*.npz"))
        if not parts:
            raise FileNotFoundError(f"잡음 라이브러리를 찾을 수 없습니다: {folder}")
        # int16으로 저장된 잡음 파형을 [-1, 1] 실수로 변환
        self.noise = (
            np.concatenate([np.load(p)["noise"] for p in parts]).astype(np.float64)
            / 32768.0
        )
        # 상호상관용 FFT를 미리 한 번만 계산해 둔다 (켤레 복소수)
        self.spectra = np.conj(np.fft.rfft(self.noise, FFT_LEN, axis=1)).astype(
            np.complex64
        )
        self.energy = np.sum(self.noise**2, axis=1)

    def best_match(self, y):
        """상관이 가장 큰 (점수, 라이브러리 번호, 지연)을 반환"""
        y_spec = np.fft.rfft(y, FFT_LEN).astype(np.complex64)
        best = (0.0, 0, 0)
        for start in range(0, len(self.noise), BATCH):
            xcorr = np.fft.irfft(
                self.spectra[start : start + BATCH] * y_spec[None, :], FFT_LEN, axis=1
            )
            score = np.abs(xcorr) / np.sqrt(
                self.energy[start : start + BATCH, None] + 1e-12
            )
            k, lag = np.unravel_index(score.argmax(), score.shape)
            if score[k, lag] > best[0]:
                # FFT_LEN/2보다 큰 인덱스는 음수 지연을 뜻함
                best = (
                    float(score[k, lag]),
                    start + k,
                    int(lag if lag <= FFT_LEN // 2 else lag - FFT_LEN),
                )
        return best

    def place(self, index, lag, length):
        """라이브러리 구간을 lag만큼 옮겨 입력 길이에 맞게 배치. (배치된 신호, 덮은 위치) 반환"""
        segment = self.noise[index]
        out = np.zeros(length)
        covered = np.zeros(length, dtype=bool)
        lo, hi = max(0, lag), min(length, lag + len(segment))
        if hi > lo:
            out[lo:hi] = segment[lo - lag : hi - lag]
            covered[lo:hi] = True
        return out, covered

    def estimate_noise(self, y):
        """입력 속 잡음 파형 추정값과, 그 추정이 덮는 샘플 위치를 반환"""
        n = len(y)
        estimate = np.zeros(n)
        covered = np.zeros(n, dtype=bool)
        search = y.copy()
        for _ in range(MAX_SEGMENTS):
            _, index, lag = self.best_match(search)
            candidate, mask = self.place(index, lag, n)
            new = mask & ~covered  # 아직 안 덮인 부분만 새로 채움
            if new.sum() < MIN_NEW_SAMPLES:
                break
            corr = np.dot(y[new], candidate[new]) / np.sqrt(
                np.dot(y[new], y[new]) * np.dot(candidate[new], candidate[new]) + 1e-20
            )
            if corr < MIN_CORRELATION:  # 엉뚱한 구간이면 중단
                break
                # 구간마다 크기를 따로 맞춤 (구간마다 섞인 잡음 크기가 다를 수 있음)
            scale = np.dot(y[new], candidate[new]) / max(
                np.dot(candidate[new], candidate[new]), 1e-20
            )
            estimate[new] = scale * candidate[new]
            covered |= new
            if covered.all():
                break
            search = y.copy()
            search[covered] = 0.0  # 나머지 부분을 덮을 구간을 다시 탐색
        return estimate, covered
