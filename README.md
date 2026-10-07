# Minhwa Symbolic KG Context Classification

한국 전통 민화의 상징 레이블을 소재만으로 분류하는 조건(A)부터 메타데이터 맥락(B), 학습 fold 내 지식그래프 맥락(C), 그래프 임베딩(D)까지를 작품 단위 교차검증으로 비교한 연구의 재현성 패키지입니다.

> 중요: 이 저장소는 AI Hub 원본 이미지나 JSON/원천 메타데이터를 포함하지 않습니다. 이용자는 [AI Hub ‘한국 전통 민화 제작 데이터’](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=realm&dataSetSn=71920)를 직접 신청·다운로드해야 합니다.

## 공개 범위

이 패키지는 “원천 데이터 재배포”가 아닌 “분석 재현”을 목표로 합니다.

- 공개: 분석 코드, 고정 설정, 레이블 규칙, 가명화된 fold 배정, 최소 OOF 예측, 집계 지표, 부트스트랩·민감도 결과
- 비공개: 원본 이미지, 원본 JSON, 작품명·소재명을 포함한 평탄화 원천표, 개인 로컬 경로

세부 기준은 [DATA_USE_NOTICE.md](DATA_USE_NOTICE.md)와 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)를 보세요.

## 실험 설계

| 조건 | 입력 | pooled OOF macro-F1 | micro-F1 |
|---|---|---:|---:|
| A | 대상 소재 | 0.2278 | 0.2533 |
| B | A + 화목·주변 소재 | 0.2680 | 0.2759 |
| C | B + 학습 fold에서만 구축한 KG 관계어 | 0.2701 | 0.2775 |
| D | C + 64차원 소재·화목 그래프 임베딩 | 0.2705 | 0.2781 |

단위는 34,522개 관계 행, 3,779개 작품 그룹, 8개 배타적 상징 클래스입니다. 모든 행은 작품 그룹 단위로 5개 fold 중 하나에만 포함됩니다. D−C 작품군 대응 부트스트랩 95% 구간은 macro-F1 `[-0.0016, +0.0025]`, micro-F1 `[-0.0014, +0.0025]`입니다. 따라서 구조 임베딩의 추가 이득은 작으며 0을 포함합니다.

## 빠른 검증

원천 데이터 없이도 공개 패키지의 행 수, fold, 클래스, 핵심 지표를 검증할 수 있습니다.

```bash
python3 src/verify_results.py
```

기대 출력:

```text
PASS: 34,522 rows; 3,779 artworks; 8 classes; 5 folds
PASS: pooled A/B/C/D metrics and D-C bootstrap intervals match
PASS: public result-file SHA-256 hashes match
```

원천 데이터로 재실행하는 순서는 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)에 있습니다.

## 저장소 구조

```text
config/                 논문 버전의 고정 설정
data/private/           이용자가 로컬에만 두는 입력(.gitignore)
metadata/               스키마, 레이블 규칙, fold, 해시, 계보
results/reference/      논문에서 보고한 공개 가능 집계값
results/generated/      재실행 산출물(.gitignore)
src/                    입력 검증, A–D, 민감도, 에이전트, 패키지 검증
tests/                  공개 패키지 스모크 테스트
```

## 해석 주의

- 8개 상위 레이블은 독립 전문가 정답이 아니라 원천 키워드에 사전 고정한 규칙을 적용한 준정답입니다.
- 같은 작품의 소재와 레이블을 조합한 관계 행이므로 인과관계를 뜻하지 않습니다.
- C의 B 대비, D의 C 대비 증분은 작습니다. 주요 결론은 “메타데이터 맥락의 향상은 확인되지만 KG·임베딩의 추가 이득은 제한적”이라는 것입니다.
- 에이전트의 `supported/plausible/needs_review` 임계값은 외부 검증으로 최적화한 값이 아니며 검토 우선순위용 휴리스틱입니다.

## 인용과 라이선스

인용 정보는 [CITATION.cff](CITATION.cff)에 있습니다. 코드는 MIT License로 제공됩니다. AI Hub 원천 데이터에 대한 권리나 재배포 권한은 부여하지 않습니다.
