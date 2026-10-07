# Sources and Method References

## Data source

- AI Hub, [한국 전통 민화 제작 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=realm&dataSetSn=71920), dataset 71920, version 1.1.

## Software and evaluation design

- scikit-learn 1.6.1, [`StratifiedGroupKFold`](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html). 작품 그룹을 분리하면서 클래스 분포를 가능한 보존하는 5-fold 분할에 사용했습니다.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC. 작품 그룹 단위 대응 재표집의 배경 문헌입니다.

## Knowledge graph and cultural heritage context

- Doerr, M. (2003). The CIDOC Conceptual Reference Module: An Ontological Approach to Semantic Interoperability of Metadata. *AI Magazine*, 24(3), 75–92. <https://doi.org/10.1609/aimag.v24i3.1720>
- Hyvönen, E. (2012). *Publishing and Using Cultural Heritage Linked Data on the Semantic Web*. Morgan & Claypool. <https://doi.org/10.2200/S00452ED1V01Y201210WBE003>
- Shen, T. et al. (2022). A Comprehensive Survey on Knowledge Graph Completion. *ACM Transactions on Intelligent Systems and Technology*, 13(5). <https://doi.org/10.1145/3491238>

## Repository citation

GitHub의 인용 메타데이터는 루트 [CITATION.cff](../CITATION.cff)에 정의했습니다. 논문 DOI와 Zenodo DOI는 게재 확정 후 추가하세요.
