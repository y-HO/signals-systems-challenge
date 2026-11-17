"""
input 폴더의 모든 WAV를 잡음 제거해 output 폴더에 같은 이름으로 저장 (통계적 필터만 사용)

사용법:  python denoise.py --input input --output output
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from algorithm import SAMPLE_RATE, enhance


def denoise_channel(x):
    """한 채널 잡음 제거"""
    return enhance(x)


def denoise_file(src, dst):
    """WAV 하나 처리. 샘플링 주파수·채널 수·샘플 수·비트 형식을 그대로 유지"""
    audio, sr = sf.read(src, dtype="float64", always_2d=True)  # (샘플 수, 채널 수)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{src}: {SAMPLE_RATE} Hz 입력이 필요합니다 (현재 {sr} Hz)")
    out = np.stack(
        [denoise_channel(audio[:, ch]) for ch in range(audio.shape[1])], axis=1
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
    for src in tqdm(files, desc="denoise"):
        denoise_file(src, args.output / src.relative_to(args.input))
    print(f"{len(files)}개 파일 저장 완료 → {args.output}")


if __name__ == "__main__":
    main()
