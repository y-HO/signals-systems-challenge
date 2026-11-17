"""
Validation(또는 train) 전체에 대해 PESQ-WB와 SNR을 계산해 출력하고 CSV로 저장

사용법:  python evaluate.py --data 데이터폴더경로 --split validation
         (데이터폴더 = train/, validation/ 와 각 samples.csv가 들어 있는 폴더)
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
from reference import NoiseLibrary

DATA = None
LIBRARY = None


def snr_db(clean, estimate):
    """SNR = 10·log10( Σclean² / Σ(clean − estimate)² )"""
    return 10 * np.log10(np.sum(clean**2) / max(np.sum((clean - estimate) ** 2), 1e-20))


def score(row):
    """파일 하나: noisy와 처리 결과의 PESQ·SNR (clean은 채점에만 사용)"""
    clean, _ = sf.read(DATA / row["clean_path"], dtype="float64")
    noisy, _ = sf.read(DATA / row["noisy_path"], dtype="float64")
    out = denoise_channel(noisy, LIBRARY)
    return {
        "segment_id": row["segment_id"],
        "noise_category": row["noise_category"],
        "target_snr_db": row["target_snr_db"],
        "pesq_noisy": pesq(SAMPLE_RATE, clean, noisy, "wb"),
        "pesq_out": pesq(SAMPLE_RATE, clean, out, "wb"),
        "snr_noisy": snr_db(clean, noisy),
        "snr_out": snr_db(clean, out),
    }


def init(data):
    """병렬 작업자마다 라이브러리를 한 번 불러옴"""
    global DATA, LIBRARY
    DATA = data
    LIBRARY = NoiseLibrary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--workers", type=int, default=2
    )  # 작업자 하나당 메모리 약 1 GB
    args = parser.parse_args()

    rows = pd.read_csv(args.data / args.split / "samples.csv").to_dict("records")
    with Pool(args.workers, initializer=init, initargs=(args.data,)) as pool:
        result = pd.DataFrame(pool.map(score, rows, chunksize=4))
    result.to_csv(f"{args.split}_scores.csv", index=False)
    cols = ["pesq_noisy", "pesq_out", "snr_noisy", "snr_out"]
    result[cols].mean().round(4).to_frame("mean").T.to_csv(
        f"{args.split}_summary.csv", index=False
    )
    print("[전체 평균]")
    print(result[cols].mean().round(4).to_string())
    print("\n[SNR별]")
    print(result.groupby("target_snr_db")[cols].mean().round(3).to_string())
    print("\n[잡음 종류별]")
    print(result.groupby("noise_category")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
