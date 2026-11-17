"""
한 번에 실행: validation(또는 train) noisy 전부 잡음 제거 → WAV 저장 → PESQ·SNR 계산 → CSV 저장

결과
  output_<split>/                 처리된 WAV (noisy 파일과 같은 이름)
  <split>_scores.csv              파일별 PESQ·SNR
  <split>_summary.csv             전체 평균
  터미널                           SNR별 / 잡음 종류별 표, 전체 PESQ 평균

사용법:  python main.py                (이 폴더 안에 train/, validation/ 이 있을 때)
         python main.py --limit 10     (앞 10개만 빠르게 시험)
         python main.py --data 다른경로 (데이터가 다른 곳에 있을 때)
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from pesq import pesq

from algorithm import SAMPLE_RATE
from denoise import denoise_channel

HERE = Path(__file__).resolve().parent
DATA = None
OUT_DIR = None


def snr_db(clean, estimate):
    """SNR = 10·log10( Σclean² / Σ(clean − estimate)² )"""
    return 10 * np.log10(np.sum(clean**2) / max(np.sum((clean - estimate) ** 2), 1e-20))


def init(data, out_dir):
    """병렬 작업자마다 경로 준비"""
    global DATA, OUT_DIR
    DATA, OUT_DIR = data, out_dir


def process(row):
    """파일 하나: 잡음 제거 → WAV 저장 → 점수 계산 (clean은 채점에만 사용)"""
    noisy_path = DATA / row["noisy_path"]
    clean, _ = sf.read(DATA / row["clean_path"], dtype="float64")
    noisy, sr = sf.read(noisy_path, dtype="float64")

    out = np.clip(denoise_channel(noisy), -1.0, 1.0)
    sf.write(OUT_DIR / noisy_path.name, out, sr, subtype=sf.info(noisy_path).subtype)

    return {
        "segment_id": row["segment_id"],
        "noise_category": row["noise_category"],
        "target_snr_db": row["target_snr_db"],
        "pesq_noisy": pesq(SAMPLE_RATE, clean, noisy, "wb"),
        "pesq_out": pesq(SAMPLE_RATE, clean, out, "wb"),
        "snr_noisy": snr_db(clean, noisy),
        "snr_out": snr_db(clean, out),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=HERE
    )  # 기본: 이 폴더 안의 train/, validation/
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--limit", type=int, default=None
    )  # 앞 N개만 처리 (빠른 시험용)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rows = pd.read_csv(args.data / args.split / "samples.csv")
    if args.limit:
        rows = rows.head(args.limit)
    out_dir = HERE / f"output_{args.split}"
    out_dir.mkdir(exist_ok=True)

    print(f"{len(rows)}개 처리 시작...")
    with Pool(args.workers, initializer=init, initargs=(args.data, out_dir)) as pool:
        result = pd.DataFrame(pool.map(process, rows.to_dict("records"), chunksize=4))

    cols = ["pesq_noisy", "pesq_out", "snr_noisy", "snr_out"]
    result.to_csv(HERE / f"{args.split}_scores.csv", index=False)
    result[cols].mean().round(4).to_frame("mean").T.to_csv(
        HERE / f"{args.split}_summary.csv", index=False
    )

    print("\n[SNR별]")
    print(result.groupby("target_snr_db")[cols].mean().round(3).to_string())
    print("\n[잡음 종류별]")
    print(result.groupby("noise_category")[cols].mean().round(3).to_string())
    print(
        f"\n저장: {out_dir.name}/ (WAV {len(result)}개), {args.split}_scores.csv, {args.split}_summary.csv"
    )
    print(
        f"\n★ PESQ 평균  noisy {result.pesq_noisy.mean():.3f}  →  처리 후 {result.pesq_out.mean():.3f}"
    )


if __name__ == "__main__":
    main()
