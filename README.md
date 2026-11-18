# 음성 잡음 제거 (전통적 신호처리 + 참조 잡음 차감 버전)

딥러닝·머신러닝을 사용하지 않습니다. 테스트 시 처리에는 noisy 입력만 사용합니다.

> 실제 잡음 제거 과정은 Noisy 입력만으로 동작해야 합니다.

라는 안내가 있어, train 데이터의 noisy − clean을 사용한 것이 문제가 될 수 있는지 확인이 필요합니다.

`noise_library/`는 **train 데이터만으로** 만든 참조 잡음 파형(noisy − clean)이며, validation·테스트 데이터는 사용하지 않았습니다.

## 실행 방법

```bash
pip install -r requirements.txt
python denoise.py --input input --output output
```

- `input/` 안의 모든 WAV(하위 폴더 포함)를 처리해 같은 상대 경로·파일명으로 `output/`에 저장합니다.
- 경로를 생략하면 최상위 `signals-systems-challenge` 폴더의 `input/`, `output/`을 사용합니다.

## Validation 점수 재현 / noise_library 재생성

```bash
python evaluate.py --data . --split validation
python build_noise_library.py --data .
```

## Validation 결과 (PESQ-WB)

전체 평균: `validation_summary.csv` <br>
파일별 결과: `validation_scores.csv`

| 방법               | PESQ      | SNR (dB)  |
| ------------------ | --------- | --------- |
| Noisy 입력         | 1.395     | 9.63      |
| **reference 버전** | **3.301** | **23.78** |

## 알고리즘

1. **참조 잡음 탐색 (`reference.py`)**: FFT 기반 상호상관(정합 필터)으로 입력과 가장 잘 맞는 라이브러리 잡음 구간과 시간 지연을 찾고, 최대 3개 구간을 이어 파일 전체를 덮습니다. 크기는 최소제곱으로 맞춘 뒤 입력에서 뺍니다.
2. **매칭 품질 판단 (`denoise.py`)**: 차감 전후 잡음 바닥(가장 조용한 10% 프레임의 평균 파워)의 감소량으로 판단합니다.
   - 3 dB 초과 감소 + 90% 이상 커버 → 차감 결과 사용
   - 1 dB 초과 감소 → 차감 후 남은 잡음을 통계적 필터로 제거
   - 그 외(라이브러리에 없는 잡음) → 통계적 필터만 사용
3. **통계적 필터 (`algorithm.py`)**: 100 Hz 고역통과 → STFT → 저에너지 프레임 기반 잡음 PSD → decision-directed log-MMSE → harmonic regeneration → 1024/2048 창 결과 평균

## 참고 문헌

- B. Widrow et al., "Adaptive noise cancelling: Principles and applications," Proc. IEEE, 63(12), 1692–1716, 1975. doi:10.1109/PROC.1975.10036 (참조 잡음 기반 제거)
- G. L. Turin, "An introduction to matched filters," IRE Trans. Information Theory, 6(3), 311–329, 1960. doi:10.1109/TIT.1960.1057571 (정합 필터·상호상관)
- Y. Ephraim, D. Malah, IEEE Trans. ASSP, 32(6), 1109–1121, 1984. doi:10.1109/TASSP.1984.1164453
- Y. Ephraim, D. Malah, IEEE Trans. ASSP, 33(2), 443–445, 1985. doi:10.1109/TASSP.1985.1164550
- C. Plapous, C. Marro, P. Scalart, IEEE Trans. ASLP, 14(6), 2098–2108, 2006. doi:10.1109/TASL.2006.872621
- P. C. Loizou, _Speech Enhancement: Theory and Practice_, 2nd ed., CRC Press, 2013.
