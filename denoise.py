"""
input 폴더의 모든 WAV를 잡음 제거해 output 폴더에 같은 이름으로 저장

파일마다
1. 잡음 라이브러리에서 같은 잡음을 찾아 뺀다 (reference.py)
2. 차감 전후 '잡음 바닥'(가장 조용한 10% 프레임의 평균 파워)이 얼마나 줄었는지로 판단
   - 3 dB 넘게 줄고 파일의 90% 이상을 덮음 → 차감 결과 그대로 사용
   - 1 dB 넘게 줄어듦                       → 차감 후 남은 잡음을 통계적 필터로 정리
   - 그 외 (라이브러리에 없는 잡음)          → 통계적 필터만 사용 (algorithm.py)

사용법:  python denoise.py --input input --output output
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from algorithm import SAMPLE_RATE, enhance
from reference import NoiseLibrary

MATCH_DB = 1.0  # 차감 후 잡음 바닥이 이만큼 넘게 줄면 "잘 맞음"
CROSSFADE = 801  # 차감 구간과 필터 구간 경계를 부드럽게 잇는 창 길이 (50 ms)


def noise_floor(x, frame=512, hop=256):
    """가장 조용한 10% 프레임(말 쉬는 구간)의 평균 파워 = 잡음 크기의 대략적인 값"""
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    power = np.sort(np.mean(frames**2, axis=1))
    return power[: max(1, len(power) // 10)].mean()


def denoise_channel(x, library):
    """한 채널 잡음 제거"""
    noise, covered = library.estimate_noise(x)
    residual = x - noise
    floor_drop_db = 10 * np.log10(noise_floor(x) / max(noise_floor(residual), 1e-20))
    if floor_drop_db > MATCH_DB:
        # 잘 맞음: 잡음을 찾은 구간은 차감 결과, 못 찾은 구간은 통계적 필터 결과를 쓰고
        # 경계는 Hann 창으로 부드럽게 섞는다
        window = np.hanning(CROSSFADE)
        weight = np.convolve(covered.astype(float), window / window.sum(), mode="same")
        return weight * residual + (1 - weight) * enhance(x)
    if floor_drop_db > 0:
        return enhance(residual)  # 조금 맞음: 차감 후 남은 잡음을 필터로 정리
    return enhance(x)  # 안 맞음 (라이브러리에 없는 잡음): 필터만 사용


def denoise_file(src, dst, library):
    """WAV 하나 처리. 샘플링 주파수·채널 수·샘플 수·비트 형식을 그대로 유지"""
    audio, sr = sf.read(src, dtype="float64", always_2d=True)  # (샘플 수, 채널 수)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{src}: {SAMPLE_RATE} Hz 입력이 필요합니다 (현재 {sr} Hz)")
    out = np.stack(
        [denoise_channel(audio[:, ch], library) for ch in range(audio.shape[1])], axis=1
    )
    out = np.clip(out, -1.0, 1.0)  # 정수 WAV 저장 시 클리핑 방지
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        dst, out if out.shape[1] > 1 else out[:, 0], sr, subtype=sf.info(src).subtype
    )


def main():
    here = Path(__file__).resolve().parent  # 절대 경로 대신 이 파일 위치 기준
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=here / "input")
    parser.add_argument("--output", type=Path, default=here / "output")
    args = parser.parse_args()

    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() == ".wav")
    if not files:
        raise SystemExit(f"WAV 파일이 없습니다: {args.input}")
    library = NoiseLibrary()  # 한 번만 불러와서 모든 파일에 사용
    for src in tqdm(files, desc="denoise"):
        denoise_file(src, args.output / src.relative_to(args.input), library)
    print(f"{len(files)}개 파일 저장 완료 → {args.output}")


if __name__ == "__main__":
    main()
