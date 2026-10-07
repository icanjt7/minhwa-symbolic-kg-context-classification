# Reproducibility Guide

## 1. 재현 범위

두 수준의 검증을 구분합니다.

1. **결과 무결성 검증**: 공개된 최소 OOF, fold, 집계 지표와 SHA-256를 검사합니다. 원천 데이터가 필요 없습니다.
2. **전체 분석 재실행**: AI Hub에서 직접 받은 데이터로 A–D 예측과 민감도 분석을 다시 생성합니다.

## 2. 결과 무결성 검증

```bash
python3 src/verify_results.py
python3 -m unittest discover -s tests -v
```

검증기는 다음을 확인합니다.

- OOF 34,522행, 3,779작품, 8클래스, 5개 fold
- 한 작품이 복수 fold에 들어가지 않음
- A/B/C/D pooled OOF 지표
- D−C 2,000회 작품군 부트스트랩 요약
- 공개 결과 파일의 SHA-256

## 3. 원천 데이터 준비

AI Hub 데이터셋 71920, 버전 1.1을 이용합니다. 이 저장소는 원본을 다운로드하거나 재배포하지 않습니다.

분석용 로컬 표는 UTF-8 CSV로 준비합니다.

```text
data/private/minhwa_raw_combined.csv
data/private/manifest.csv
```

`minhwa_raw_combined.csv`의 필수 열은 다음과 같습니다.

```text
file_name, subject_type, theme_type, painting_type, ground_material,
purpose, objects, keywords, era
```

`manifest.csv`의 필수 열은 `image_id,image_stem,file_path,rel_path`입니다. `file_path`는 로컬 절대경로이며 Git에 추가하지 않습니다. 세부 정의는 [metadata/data_dictionary.md](metadata/data_dictionary.md)를 참조하세요.

입력을 미리 검증하려면:

```bash
python3 src/01_validate_source.py \
  --metadata /path/to/minhwa_raw_combined.csv \
  --manifest /path/to/manifest.csv
```

논문 분석에 사용한 정규화 파일의 기대 SHA-256는 [metadata/source_hashes.sha256](metadata/source_hashes.sha256)에 기록되어 있습니다. 해시가 다르면 데이터 버전·압축 해제·정규화 절차를 먼저 확인해야 합니다.

## 4. 고정 환경

A–C fold를 생성한 검증 환경은 Python 3.9.6, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.6.1입니다. 특히 `StratifiedGroupKFold` 배정이 버전에 따라 달라졌으므로 저장된 fold를 우선해야 합니다.

Conda:

```bash
conda env create -f environment.yml
conda activate minhwa-repro
```

venv/pip:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

결정적 실행을 위해 다음을 권장합니다.

```bash
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## 5. 전체 재실행

```bash
python src/02_run_abc_ablation.py
python src/03_run_spectral_embedding.py
python src/04_run_other_sensitivity.py
python src/05_run_agent_evaluation.py --csv
python src/06_make_tables_figures.py
python src/verify_results.py --generated
```

순서는 의존성을 따릅니다. D는 A–C에서 저장한 OOF/fold를 사용하며, 기타 범주 민감도도 A–C OOF를 사용합니다. 에이전트 평가는 로컬 `object_symbol_mapping.csv`, `subject_symbol_correlation.csv`, KG pair 표가 추가로 필요합니다.

## 6. 재현성 한계

- A–C의 최종 fold는 scikit-learn 1.6.1 환경에서 생성되었습니다. 1.9.1 시험에서 3,779작품 중 3,010개의 fold 배정이 달라졌습니다.
- D 확장 산출물은 저장된 1.6.1 fold를 고정한 후 별도 환경에서 계산되었습니다. 수치 대조에서는 `results/reference/` 값을 기준으로 삼습니다.
- 이 분석은 기존 15,901행·5클래스 표의 동일 표본 재현이 아니라, 9,119건 메타데이터에서 재구성한 보충 분석입니다.
