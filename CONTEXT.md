# Core-Clustering — Context

이 문서는 "지금 이 프로젝트가 왜 이렇게 생겼는지"를 빠르게 파악하기 위한 문서. 사용법은
`README.md` 참고, 여긴 배경/설계 이유/최근 변경사항/미해결 이슈 위주.

## 이 프로젝트가 왜 존재하나

[RedLamp](https://github.com/skimrepo/RedLamp_Check)를 "데이터 준비"([AnomSim](https://github.com/skimrepo/AnomSim))와
"학습/추론"(이 프로젝트)으로 쪼갠 것 중 후자. RedLamp의 `ConvAEC` 모델(CNN 오토인코더 + 12종
pseudo-anomaly 분류기)을 AnomSim이 만든 합성 데이터로 학습시키고, 도메인 일반화(특정 waveform은
학습에서 빼고 그 도메인에 대한 성능 확인) 평가까지 담당함.

**최종 목적**: 여기서 학습한 모델을 RedLamp_Check의 `scripts/simulation_cross_domain_metrics.py`로
가져가서, 실제 UCR/KPI 데이터에 대해 RedLamp 논문과 같은 TSB-UAD 지표(VUS_ROC, VUS_PR 등)로
채점 — "시뮬레이션만으로 학습해도 실제 이상 탐지에 얼마나 통하는지" 확인하는 것.

## RedLamp와의 관계 — 포팅 vs 재사용

- `models.py`의 `ConvEncoder`/`ConvDecoder`/`NonLinClassifier`/`MetaAEC`/`ConvAEC`는 RedLamp의
  `models/cnn.py`, `models/classifier.py`, `models/meta.py`를 **레이어 이름까지 동일하게** 포팅함
  — 그래서 여기서 학습한 `bestmodel.pkl`(state_dict)이 RedLamp의 진짜 `ConvAEC`에
  `load_state_dict(strict=True)`로 그대로 로드됨 (실제로 여러 번 검증함).
- **딱 하나 다른 점**: RedLamp의 `NonLinClassifier`는 `CrossEntropyLoss` 직전에 `Softmax`를 한 번
  더 씌우는 버그가 있었음(이미 내부에서 `log_softmax`를 하는데 이중으로 눌러버림). 여기 포팅본은
  이 `Softmax` 레이어를 뺐음 — 파라미터가 없는 레이어라 state_dict 호환성엔 영향 없음, 대신
  `predict_proba()` 헬퍼를 따로 둠.
- `redlamp_compat.py`의 `REDLAMP_ANOMALY_TYPES`는 RedLamp `main.py`가 쓰는 12종 이상치 순서를
  그대로 박아둔 상수 — RedLamp의 스코어링 코드(`anomaly_scoreing`)가 "분류기 출력의 인덱스
  0번 = normal"이라고 하드코딩하고 있어서, 여기서 학습할 때도 이 순서를 반드시 맞춰야 함
  (`--class_list redlamp`). 안 맞추면 에러 없이 조용히 결과가 틀어짐.

## 두 가지 학습 경로

| | 진입점 | 데이터 소스 | 특징 |
|---|---|---|---|
| 미리 굽기 (레거시) | `cli.py` | AnomSim의 `windowed_dataset_cli` 결과 | 매 에폭 똑같은 데이터 재사용, 소규모/디버깅용 |
| **온라인 (권장)** | `online_cli.py` | AnomSim의 `base_pool_dataset_cli` 결과 | 윈도우 자르기+이상치 주입을 학습 중 그때그때, 실전 스케일용 |

`online_dataset.py`가 AnomSim 패키지를 **직접 import**해서 이상치 주입 클래스를 재사용함
(`anomsim.anomalies.base.get_anomaly`, `anomsim.windowing.window_positions`) — 코드 복사가 아니라
단일 소스 유지가 목적. **그래서 이 프로젝트를 쓰려면 AnomSim이 형제 폴더로 있어야 함**
(`online_dataset.py` 상단에서 자동으로 `../AnomSim`을 찾아 `sys.path`에 넣는 fallback이 있음).

`splits.py`의 `make_cross_domain_split()`은 두 경로 모두에서 **코드 변경 없이 그대로 재사용됨** —
온라인 경로의 `BasePool`이 `LoadedDataset`과 똑같은 `.domain`/`.group_key()` 인터페이스를
duck-typing으로 제공하기 때문.

## ⚠️ 아직 해결 안 된 것: "매 에폭 재주입" 여부가 RedLamp과 다름

온라인 경로(`OnlineWindowedDataset`)는 `set_epoch()`으로 **매 에폭 새로운 랜덤 시드로 이상치를
다시 주입**함. 이건 처음에 "RedLamp도 매 에폭 새로 주입할 것"이라는 가정 하에 설계한 건데,
**나중에 `loaders/loader_aug.py`를 직접 다시 읽어보니 이 가정이 틀렸다는 게 밝혀짐**:

RedLamp의 `Loader_aug`는 **객체 생성 시점에 딱 한 번**(`__init__` → `_inject_anomalies()`) 모든
윈도우×이상치 조합을 주입해서 고정시키고, 이후 매 에폭 `__iter__`은 그 고정된 텐서의 **순서만
셔플**함. 즉 RedLamp은 "한 번 주입한 고정 데이터셋을 N 에폭 반복 학습"하는 거고, 우리 온라인
경로는 "매 에폭 완전히 새로운 augmentation"이라 실질적으로 훨씬 어려운 학습 문제임 — 같은
에폭 수로는 훨씬 덜 수렴할 수 있음.

**지금 상태**: 이 차이는 아직 안 고쳤음. 대신 진단 과정에서 발견한 또 다른 문제(AnomSim 쪽
정규화 누락)만 먼저 고쳐서 작은 데이터셋으로 실험해보는 중 (`AnomSim_v1`, 아래 참고). 만약
정규화 수정만으로 정확도가 여전히 낮으면, 다음 후보는 이 "매 에폭 재주입" 방식을 RedLamp처럼
"한 번만 주입하고 고정" 방식으로 바꾸는 것 — `set_epoch()`을 아예 안 부르거나 옵션으로 끌 수
있게 하면 됨.

## 최근 작업

- **OpenBLAS 스레드 크래시 수정**: 코어 수 많은 서버에서 numpy/scikit-learn이 기본으로 코어 수만큼
  스레드를 잡으려다 OpenBLAS의 빌드 시점 상한(128)을 넘겨서 segfault 나던 문제.
  `online_cli.py` 맨 위(numpy/torch import 전)에서 `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/
  `MKL_NUM_THREADS`를 4로 제한하도록 고침.
- **재학습 방지(resumable)**: 실제로 23 에폭(약 11시간) 학습이 끝난 뒤 평가/플롯 단계에서
  위 크래시로 죽는 사고가 있었음 — `output_dir/bestmodel.pkl`이 이미 있으면 학습을 건너뛰고
  바로 평가로 넘어가도록 수정 (`--force`로 강제 재학습 가능). `Trainer`도 매 에폭
  `epoch_history.json`을 저장해서, 나중에 재개해도 `run_summary.json`의 에폭별 기록이
  안 끊기게 함.
- **AnomSim 쪽 정규화 버그 진단 및 수정**: 위 "아직 해결 안 된 것" 섹션과 함께, 학습 정확도가
  논문 대비 너무 낮게 나온 원인을 `loaders/loader_aug.py`/`loaders/load.py`와 직접 대조해서
  찾음. RedLamp은 실제 데이터를 항상 `[0,1]`로 정규화하는데 AnomSim은 그러지 않았던 게 큰
  원인 — AnomSim 쪽에서 수정함(이 프로젝트는 그 수정된 데이터를 그대로 받아 씀).
- **`AnomSim_v1` 데이터셋으로 검증**: AnomSim의 `length_tiers` 기능으로 만든, UCR/KPI 구조를
  절반씩 섞은 144개 시계열 데이터셋(9도메인×16개)을 실제로 로드 + train/val 분할까지
  돌려서 정상 동작 확인함 (총 474,252 윈도우/에폭).

## 테스트

`pytest` — 현재 59개, 전부 통과. `test_online_cli.py`의 엔드투엔드 테스트는 재학습 건너뛰기
(`--force` 유무)까지 실제로 검증함.
