# Provenance

```text
AI Hub dataset 71920 v1.1
  └─ local normalized metadata (9,119 rows; not distributed)
      ├─ relation reconstruction + rule labels (34,522 rows)
      │   └─ artwork-grouped fixed 5-fold split
      │       ├─ A: motif only
      │       ├─ B: motif + metadata context
      │       ├─ C: B + training-fold KG terms
      │       └─ D: C + training-fold graph embedding
      ├─ Other-label sensitivity
      └─ Evidence-first agent threshold sensitivity
```

## 핵심 통제

- 작품 그룹 단위로 분할하여 같은 작품이 train/test에 동시에 들어가지 않습니다.
- C의 KG 관계와 D의 그래프 임베딩은 각 outer-training fold에서만 적합합니다.
- C에서 대상 소재는 KG 맥락 조회에서 제외합니다.
- D 그래프는 소재·화목 노드와 학습 작품 내 공출현 간선만 사용하며 정답 레이블 간선을 넣지 않습니다.
- OOF 성능을 보고 임베딩 차원·분류기·임계값을 재선택하지 않았습니다.

## 실행 기록

- A–C 검증: 2026-10-01, Python 3.9.6, scikit-learn 1.6.1
- D 확장: 2026-10-04, 저장된 A–C fold 사용
- 기타 레이블 민감도: 2026-10-05, 저장된 A–C OOF 사용
- 공개 패키지 구성: 2026-10-07
