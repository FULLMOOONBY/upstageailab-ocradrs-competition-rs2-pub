[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/8EfqyhZ8)
# Commerce Purchase Behavior Prediction

## Team

| ![조성지](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김기정](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김묘정](https://avatars.githubusercontent.com/u/156163982?v=4) | ![박지은](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [조성지]             |            [김기정]       |            [김묘정]             |            [박지은]        |
|                            팀장                             |                            팀원                            |                           팀원                             |                            팀원                             |

## 0. Overview
이 프로젝트는 eCommerce behavior data from multi-category store를 활용하여 사용자의 쇼핑 패턴을 분석하고, 향후 일주일 동안 구매한 상품을 예측하는 것이 목표입니다.

### Environment
- Python (>= 3.8)
- Jupyter Notebook
- PyTorch
- TensorFlow/Keras (선택)
- LightGBM / XGBoost
- Pandas / NumPy / SciPy
- Matplotlib / Seaborn
- Scikit-learn
- ALS / SASREC
  
### Requirements

pip install -r requirements.txt

## 1. Competiton Info

### Overview

- eCommerce 데이터에서 사용자 구매 패턴을 예측하는 추천 시스템 대회입니다.
- 사용자의 구매 행동을 분석하여 미래에 어떤 상품을 구매할지 예측합니다.
- 추천 시스템 성능을 향상시키기 위해 EDA, Feature Engineering, 다양한 모델링 기법을 활용합니다.

### Timeline

- 📅 2025년 3월 4일 - 프로젝트 시작
- 📅 2025년 3월 13일 - 최종 제출 마감

## 2. Components

### Directory

```
├── code
│   ├── train_als.py               # ALS 모델 학습 및 추론 코드
│   ├── train_sasrec.py            # SASRec 모델 학습 코드
