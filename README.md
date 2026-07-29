# Core-Clustering

[AnomSim](https://github.com/skimrepo/AnomSim)이 만든 데이터로 RedLamp의 모델(`ConvAEC`,
CNN 오토인코더 + 12종 이상치 분류기)을 학습시키는 도구. 학습이 끝나면 나온 체크포인트를
RedLamp_Check에 그대로 갖다 써서 실제 데이터(UCR/KPI)와 교차 테스트할 수 있음.

## 설치

```bash
pip install numpy matplotlib scikit-learn pytest
```

`torch`는 일부러 뺐음 — 서버에 CUDA 버전 맞춰서 이미 깔려있는 torch를 그대로 써야 함.
`requirements.txt`를 그대로 `pip install -r`로 돌리면 torch가 엉뚱한 버전으로 재설치되면서
CUDA 세팅이 깨질 수 있으니 주의.

저장소 루트 폴더(`Core-Clustering/`)에서 실행해야 함.

## 우리 목적에 맞는 실행 예시 (이것만 돌리면 됨)

```bash
cd Core-Clustering
python -m core_clustering.cli \
  --dataset_dir /경로/AnomSim/data/windowed_v1 \
  --val_fraction 0.1 \
  --output_dir ./outputs \
  --run_id sim_v1 \
  --epochs 100 \
  --batch_size 128 \
  --gpu 0 \
  --embedding_dim 128 \
  --class_list redlamp
```

`--dataset_dir`만 AnomSim에서 방금 만든 폴더 경로로 바꿔주면 됨. 나머지는 그대로 써도 됨.

**`--class_list redlamp`는 절대 빼면 안 됨.** 나중에 이 모델을 RedLamp 쪽에서 불러다 쓰려면
이상치 12종의 순서가 RedLamp과 똑같아야 하는데(0번이 "normal"), 이 옵션 없이 학습하면
순서가 알파벳순으로 제멋대로 정해져서 — 에러는 안 나는데 결과가 조용히 다 틀어짐. 반드시 넣을 것.

### 파라미터 설명

| 파라미터 | 의미 | 기본값 | 예시에서 쓴 값 |
|---|---|---|---|
| `--dataset_dir` | AnomSim이 만든 데이터셋 폴더 경로 | (필수) | AnomSim에서 만든 `windowed_v1` 경로 |
| `--output_dir` | 학습 결과가 저장될 상위 폴더 | `./outputs` | `./outputs` |
| `--run_id` | 이번 학습 결과에 붙일 이름(하위 폴더명) | 자동 생성(시각 기준) | `sim_v1` (원하는 이름으로) |
| `--epochs` | 몇 번 반복 학습할지 | 100 | 100 |
| `--val_fraction` | 전체 중 검증용으로 뗄 비율 | 0.2 | 0.1 |
| `--batch_size` | 한 번에 몇 개씩 묶어서 학습할지 | 128 | 128 |
| `--gpu` | 사용할 GPU 번호. `-1`이면 CPU만 사용 | 0 | 0 (서버는 GPU 있으니 그대로) |
| `--embedding_dim` | 모델 내부 표현 크기 | 128 | 128 (RedLamp과 맞춰야 하므로 고정) |
| `--class_list` | 이상치 12종의 순서를 고정할지 | 없음(알파벳순) | `redlamp` (**반드시 넣기**) |
| `--held_out_domains` | 특정 파형 도메인을 학습에서 아예 제외 (예: `sine trend`) | 없음(전부 학습) | 필요할 때만 |
| `--seed` | 재현성을 위한 시드값 | 0 | 0 |

### 학습이 끝나면 `outputs/sim_v1/` 안에 뭐가 생기나

| 파일/폴더 | 내용 |
|---|---|
| `bestmodel.pkl` | 학습된 모델 가중치. `--class_list redlamp`로 학습했으면 RedLamp의 `ConvAEC`에 바로 로드 가능 |
| `run_summary.json` | 학습 전체 요약 (epoch별 loss, 데이터 분할 정보, 도메인별 윈도우 개수, held-out 정확도, 모델 하이퍼파라미터 전부) |
| `classification_accuracy.csv` | 도메인별 분류 정확도 |
| `plots/tsne_by_class.png`, `plots/tsne_by_domain.png` | 임베딩 시각화 |
| `plots/samples/` | 대표 샘플 윈도우 그래프 |

## (선택) 도메인별 자세한 분석

핵심 학습 코드/결과와는 폴더를 분리해서 관리함:

```bash
python -m research.analyze \
  --run_dir ./outputs/sim_v1 \
  --domains sine trend \
  --research_root ./research \
  --n_examples 10
```

| 파라미터 | 의미 | 기본값 |
|---|---|---|
| `--run_dir` | 분석할 학습 결과 폴더 | (필수) |
| `--domains` | 자세히 볼 도메인들 (비우면 학습 때 held-out 했던 도메인) | 없음 |
| `--research_root` | 분석 결과 저장 폴더 | `./research` |
| `--n_examples` | 맞춘/틀린 예시를 각각 몇 개씩 뽑을지 | 10 |

`research/sim_v1/<도메인>/accuracy.json` (정확히 몇 개 맞고 틀렸는지) +
`correct_examples.pdf`/`incorrect_examples.pdf` (맞은/틀린 샘플 그래프)가 도메인별로 생김.

## 테스트

```bash
pytest
```
