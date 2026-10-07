# Data Dictionary

## 로컬 입력(비공개)

### `data/private/minhwa_raw_combined.csv`

| 열 | 설명 |
|---|---|
| `file_name` | 이미지 stem. 상세묘사 파일은 작품 그룹 ID로 정규화됨 |
| `subject_type` | 화목/genre |
| `theme_type` | 주제 유형 |
| `painting_type` | 표현 기법 |
| `ground_material` | 바탕 재료 |
| `purpose` | 용도 |
| `objects` | 쉼표로 구분한 시각 소재 |
| `keywords` | 쉼표로 구분한 키워드. 상위 상징 레이블의 준정답 생성에 사용 |
| `era` | 시대 |

### `data/private/manifest.csv`

| 열 | 설명 |
|---|---|
| `image_id` | 이미지 stem과 상대경로 해시로 만든 로컬 ID |
| `image_stem` | 원본 파일 stem |
| `file_path` | 로컬 절대경로. 절대 공개 금지 |
| `rel_path` | 데이터셋 루트 기준 상대경로; 상세묘사 그룹 판별에 사용 |

## 공개 OOF

### `results/oof_predictions_minimal.csv`

| 열 | 설명 |
|---|---|
| `record_id` | 공개 표 내 순차 행 ID |
| `artwork_id_hash` | `SHA256("aihub-minhwa-v1.1\\0" + NFC(group_id))`의 앞 24자 |
| `fold` | 고정된 작품 그룹 test fold(1–5) |
| `y_true` | 규칙 기반 상위 레이블 |
| `pred_A`–`pred_D` | 각 조건의 OOF 예측 |

작품 원본 ID, 소재명, 확률, 그래프 노드 수는 최소 공개 표에서 제외했습니다.
