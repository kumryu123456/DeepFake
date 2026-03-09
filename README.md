# Deepfake Detection System

> AI보안연구센터 인턴십 (숭실대학교, 2025.10 – 2025.12) — 딥페이크 탐지 데이터 라벨링 및 모델 학습 기록

이 레포는 AI보안연구센터 인턴십 기간 중 직접 수행한 **데이터 수집·라벨링·모델 학습 작업의 실무 기록**입니다.
인턴십에서 쌓은 도메인 지식과 데이터 품질 관리 경험은 [deepfake-detector-aifactory2025](https://github.com/kumryu123456/deepfake-detector-aifactory2025) 대회 모델 설계에 직접 활용됐습니다.

---

## Key Results

| Metric | Before | After |
|---------|--------|-------|
| 라벨 품질 (합의율) | 78% | **95%** |
| 테스트 정확도 | baseline | **92%+** |
| 데이터셋 규모 | — | **6,170+ samples** |
| False-positive rate | 22% | **5%** |

---

## 주요 작업 내용

- 딥페이크/실제 영상 6,170건+ 직접 수집 및 라벨링
- 합의 기반 품질 검수 프로세스 설계 → 합의율 78% → 95% 향상
- PyTorch CNN 모델 파인튜닝 및 이진 분류 학습 파이프라인 구현
- False-positive rate 22% → 5% 감소

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

## 인턴십과의 연결

이 레포의 라벨링 작업에서 얻은 핵심 인사이트 — "GAN 기반 생성 이미지는 주파수 도메인에 아티팩트를 남긴다" — 가 국가수사본부 AI Factory 경진대회 제출 모델의 이중 브랜치 아키텍처 설계 근거가 됐습니다.
대회 제출작: [deepfake-detector-aifactory2025](https://github.com/kumryu123456/deepfake-detector-aifactory2025)

---

Gyeongmin Kim | [GitHub](https://github.com/kumryu123456)
