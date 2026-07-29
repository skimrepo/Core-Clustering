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

## 설치 (추가로 필요한 것)

**AnomSim 저장소가 Core-Clustering과 형제 폴더로 있어야 함** (`.../Git/AnomSim`,
`.../Git/Core-Clustering`처럼). 온라인 학습이 실제로 윈도우를 자르고 이상치를 주입할 때
AnomSim의 이상치 코드를 그대로 가져다 쓰기 때문 — 복사해서 중복 관리하는 대신, 같은 코드를
직접 재사용해서 두 저장소의 로직이 절대 어긋나지 않게 한 것. 형제 폴더가 아니면
`PYTHONPATH`에 AnomSim 경로를 추가하면 됨.

## 우리 목적에 맞는 실행 예시 (이것만 돌리면 됨)

**추천 방식: 온라인 학습** — AnomSim의 `base_pool_dataset_cli`가 만든 "원본 시계열만 있는"
폴더를 넣으면, 윈도우 자르기 + 12종 이상치 주입을 **학습 중에 매 에폭 그때그때** 함
(RedLamp의 `Loader_aug`와 동일한 방식 — 매 에폭 새로운 랜덤 주입이라 다양성이 더 좋고,
디스크에 미리 구워둘 필요가 없어서 훨씬 가벼움).

```bash
cd Core-Clustering
python -m core_clustering.online_cli \
  --dataset_dir /경로/AnomSim/data/base_pool_v1 \
  --val_fraction 0.1 \
  --window_size 100 --window_step 10 \
  --output_dir ./outputs \
  --run_id sim_v1 \
  --epochs 100 \
  --batch_size 128 \
  --gpu 0 \
  --embedding_dim 128 \
  --class_list redlamp
```

`--dataset_dir`만 AnomSim의 `base_pool_dataset_cli`로 방금 만든 폴더 경로로 바꿔주면 됨.

**`--class_list redlamp`는 절대 빼면 안 됨.** 나중에 이 모델을 RedLamp 쪽에서 불러다 쓰려면
이상치 12종의 순서가 RedLamp과 똑같아야 하는데(0번이 "normal"), 이 옵션 없이 학습하면
순서가 알파벳순으로 제멋대로 정해져서 — 에러는 안 나는데 결과가 조용히 다 틀어짐. 반드시 넣을 것.

### 파라미터 설명

| 파라미터 | 의미 | 기본값 | 예시에서 쓴 값 |
|---|---|---|---|
| `--dataset_dir` | AnomSim `base_pool_dataset_cli`가 만든 폴더 경로 | (필수) | AnomSim에서 만든 `base_pool_v1` 경로 |
| `--window_size` / `--window_step` | 학습 중 잘라낼 윈도우 크기/보폭 | 100 / 10 | 100 / 10 (RedLamp과 맞춤, 바꾸지 말 것) |
| `--output_dir` | 학습 결과가 저장될 상위 폴더 | `./outputs` | `./outputs` |
| `--run_id` | 이번 학습 결과에 붙일 이름(하위 폴더명) | 자동 생성(시각 기준) | `sim_v1` (원하는 이름으로) |
| `--epochs` | 몇 번 반복 학습할지 | 100 | 100 |
| `--val_fraction` | 전체 중 검증용으로 뗄 비율 | 0.2 | 0.1 |
| `--batch_size` | 한 번에 몇 개씩 묶어서 학습할지 | 128 | 128 |
| `--gpu` | 사용할 GPU 번호. `-1`이면 CPU만 사용 | 0 | 0 (서버는 GPU 있으니 그대로) |
| `--embedding_dim` | 모델 내부 표현 크기 | 128 | 128 (RedLamp과 맞춰야 하므로 고정) |
| `--class_list` | 이상치 12종의 순서를 고정할지 | 없음(알파벳순) | `redlamp` (**반드시 넣기**) |
| `--held_out_domains` | 특정 파형 도메인을 학습에서 아예 제외 (예: `sine trend`) | 없음(전부 학습) | 필요할 때만 |
| `--num_workers` | 데이터 로딩 병렬 프로세스 수 | 0 | 스케일 커지면 4~8 정도로 올리는 걸 추천 (윈도우 자르기+주입을 매번 그때그때 하다보니 CPU 코어를 여러 개 쓰면 훨씬 빨라짐) |
| `--eval_max_samples` | 정확도/시각화용으로 도메인당 최대 몇 개 윈도우만 뽑아볼지 (학습 자체와는 무관, 보고용) | 5000 | 5000 |
| `--seed` | 재현성을 위한 시드값 | 0 | 0 |

### 학습이 끝나면 `outputs/sim_v1/` 안에 뭐가 생기나

| 파일/폴더 | 내용 |
|---|---|
| `bestmodel.pkl` | 학습된 모델 가중치. `--class_list redlamp`로 학습했으면 RedLamp의 `ConvAEC`에 바로 로드 가능 |
| `run_summary.json` | 학습 전체 요약 (epoch별 loss, 데이터 분할 정보, 도메인별 윈도우 개수, held-out 정확도, 모델 하이퍼파라미터 전부) |
| `classification_accuracy.csv` | 도메인별 분류 정확도 |
| `plots/tsne_by_class.png`, `plots/tsne_by_domain.png` | 임베딩 시각화 |
| `plots/samples/` | 대표 샘플 윈도우 그래프 |

## 소규모 실험/디버깅용 (옛날 방식)

AnomSim의 `windowed_dataset_cli`로 미리 구워둔 데이터셋을 쓰는 방식도 남아있음:

```bash
python -m core_clustering.cli --dataset_dir /경로/windowed_v1 --class_list redlamp ...
```

파라미터/출력은 위 온라인 방식과 거의 같은데, 데이터가 이미 디스크에 다 구워져 있어서
매 에폭 항상 똑같은 걸 재사용함 (RedLamp처럼 에폭마다 새로 주입되지 않음). 아주 작은
규모의 빠른 실험/디버깅에는 여전히 편하지만, 실전 학습에는 위 `online_cli`를 쓸 것.

> ⚠️ `research.analyze`(아래)는 아직 이 옛날 방식(`cli.py`)의 결과만 지원함 — `online_cli`로
> 학습한 결과에는 아직 못 씀 (다음에 필요하면 확장 가능).

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
