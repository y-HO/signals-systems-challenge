"""
train 데이터로 참조 잡음 라이브러리 만들기 (팀이 한 번만 실행, 조교는 실행할 필요 없음)

train의 각 쌍에서 섞인 잡음 = noisy − clean  (16비트 PCM이라 정확히 계산됨)
결과를 int16으로 압축해 noise_library/part0~3.npz로 나눠 저장 (GitHub 파일당 100 MB 제한)
validation·테스트 데이터는 전혀 사용하지 않는다.

사용법:  python build_noise_library.py --data 데이터폴더경로
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

N_SHARDS = 4  # 나눠 저장할 파일 수


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "noise_library"
    )
    args = parser.parse_args()

    table = pd.read_csv(args.data / "train" / "samples.csv")
    noises = []
    for _, row in table.iterrows():
        clean, _ = sf.read(args.data / row.clean_path, dtype="int16")
        noisy, _ = sf.read(args.data / row.noisy_path, dtype="int16")
        noises.append(
            (noisy.astype(np.int32) - clean.astype(np.int32)).astype(np.int16)
        )
    noises = np.stack(noises)
    codes = table.noise_code.to_numpy().astype(str)  # 잡음 코드 (참고용)

    args.out.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(np.array_split(np.arange(len(noises)), N_SHARDS)):
        np.savez_compressed(
            args.out / f"part{i}.npz", noise=noises[idx], codes=codes[idx]
        )
    print(f"잡음 {len(noises)}개를 {args.out}에 저장했습니다")


if __name__ == "__main__":
    main()
