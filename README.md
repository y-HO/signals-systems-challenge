# 음성 잡음 제거 (전통적 신호처리, pure 버전)

딥러닝·머신러닝을 사용하지 않습니다. 잡음 제거에는 **noisy 입력만** 사용하며, train·clean 데이터나 잡음 라이브러리를 전혀 사용하지 않습니다.

## 실행 방법

```bash
pip install -r requirements.txt
python denoise.py --input input --output output
```

- `input/` 안의 모든 WAV(하위 폴더 포함)를 처리해 같은 상대 경로·파일명으로 `output/`에 저장합니다.
- 출력은 입력과 같은 샘플링 주파수(16 kHz), 채널 수, 샘플 수, 비트 형식을 유지합니다.
- 경로를 생략하면 최상위 `signals-systems-challenge` 폴더의 `input/`, `output/`을 사용합니다.

## Validation 점수 재현

최상위 폴더에 `validation/` 폴더를 넣은 뒤 실행합니다.

```bash
python main.py
```

잡음 제거한 WAV를 `output_validation/`에 저장하고, PESQ·SNR을 계산해 CSV로 저장합니다.

## Validation 결과 (PESQ-WB)

전체 평균: `validation_summary.csv`
파일별 결과: `validation_scores.csv`

| 방법          | PESQ      | SNR (dB)  |
| ------------- | --------- | --------- |
| Noisy 입력    | 1.395     | 9.63      |
| **pure 버전** | **1.968** | **12.73** |

## 알고리즘 (`algorithm.py`)

1. **고역통과 필터**: 100 Hz 4차 Butterworth, 양방향 필터링(위상 왜곡 없음)으로 바람·엔진 저역 잡음 제거
2. **잡음 스펙트럼 추정**: STFT(Hann, hop 128) 후 에너지가 가장 작은 10% 프레임의 평균 파워
3. **log-MMSE 이득**: decision-directed 방식으로 사전 SNR을 추정해 이득 계산
4. **하모닉 재생성**: 1차 결과를 반파 정류해 깎여 나간 음성 하모닉을 복원한 뒤 사전 SNR을 다시 계산
5. **다중 해상도 평균**: 창 길이 1024 / 2048 두 가지로 2~4를 수행하고 결과 평균
6. **다중대역 스펙트럼 차감**: 대역별 SNR에 따라 빼는 양을 다르게 한 결과를 30% 섞음

## 참고 문헌

- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator," IEEE Trans. ASSP, 32(6), 1109–1121, 1984. doi:10.1109/TASSP.1984.1164453
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator," IEEE Trans. ASSP, 33(2), 443–445, 1985. doi:10.1109/TASSP.1985.1164550
- C. Plapous, C. Marro, P. Scalart, "Improved signal-to-noise ratio estimation for speech enhancement," IEEE Trans. ASLP, 14(6), 2098–2108, 2006. doi:10.1109/TASL.2006.872621
- S. Kamath, P. Loizou, "A multi-band spectral subtraction method for enhancing speech corrupted by colored noise," Proc. IEEE ICASSP, 2002, p. IV-4164.
- P. C. Loizou, _Speech Enhancement: Theory and Practice_, 2nd ed., CRC Press, 2013.
