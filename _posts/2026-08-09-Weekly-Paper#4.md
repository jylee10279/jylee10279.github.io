---
title: Weekly Paper #4
date: 2026-08-09 16:00:00 +0900
categories: [ML, DL]
tags: [AI, cs]
---


# Tensor와 Array

`PyTorch`의 텐서(Tensor)와 `NumPy`의 어레이(Array, ndarray)는 다차원 배열 데이터를 다룬다는 점에서 외형과 사용법(인덱싱, 연산 등)이 매우 비슷합니다. 실제로 PyTorch는 `Numpy`의 편의성을 거의 그대로 가져와 설계되었죠.  

*하지만 딥러닝과 인공지능 분야에서는 텐서가 필수적입니다.*  
### 차이점
1. **GPU 가속 지원 여부**
  **`Array`**: `CPU`에서만 동작합니다. 데이터가 아주 커지거나 복잡한 연산을 할 때 속도에 한계가 있습니다.

  **`Tensor`**: `CPU`뿐만 아니라 `GPU(CUDA 등)`로 옮겨서 연산할 수 있습니다. `GPU`의 수천 개 코어를 활용해 행렬 연산을 수십~수백 배 빠르게 처리할 수 있습니다.

2. **자동 미분(Autograd) 기능**
딥러닝 모델을 학습시키려면 `역전파(Backpropagation)` 과정에서 미분값(Gradient)을 계산해야 합니다.

  `Array`: 미분 계산 기능이 없습니다.

  `Tensor`: requires_grad=True 옵션으로 텐서로 수행한 모든 연산 과정을 기억해 두었다가 `loss.backward()`로 미분값을 자동으로 계산합니다.

3. **용도와 생태계**
  `Array`: 전통적인 데이터 분석, 통계, 데이터 전처리(pandas 등과 연동), 머신러닝 라이브러리(Scikit-Learn)의 표준 데이터 형태입니다.

  `Tensor`: 인공신경망(Neural Network) 구축, 딥러닝 모델 학습 및 추론에 최적화된 데이터 형태입니다.

### 한눈에 보는 비교표

| 구분 | NumPy `Array` | PyTorch `Tensor` |
| :--- | :--- | :--- |
| **주요 사용 목적** | 일반 과학 연산, 데이터 전처리 | 딥러닝 모델 구현 및 학습 |
| **연산 장치** | CPU 전용 | CPU & GPU 모두 지원 |
| **자동 미분** | 지원 안 함 | 지원(`autograd`) |
| **주 목적** | 데이터 전처리 및 수치 연산 | 딥러닝 모델 구현 및 학습 |
---
# `CNN(Convolutional Neural Network)`
`CNN`은 이미지와 같은 컴퓨터 비전 분야의 데이터를 분석하기 위해 사용되는 인공 신경망의 한 종류입니다.  
파라미터 수의 증가, 지역 정보 손실 등 기존 `FCN(Fully Connected Network)`의 한계로 등장했습니다.  
데이터의 지역적인 특성을 추출하는 데 특화되어 있습니다. 

### `CNN`의 전체적인 흐름

CNN은 크게 [특징 추출 파트]와 [분류 파트] 2가지 단계로 나누어집니다.
1. 특징 추출 (Feature Extraction): 이미지에서 선, 색상, 모양 등의 특징을 찾아냅니다. (`Convolution Layer` + `Activation Layer` + `Pooling Layer`)
2. 분류 (Classification): 추출된 특징을 모아서 "이 이미지는 강아지다/고양이다"를 판단합니다. (`Flatten Layer` + `Fully Connected Layer`)

### 레이어별 상세 특징

1. 합성곱 레이어 (Convolutional Layer)
* 역할: 이미지의 지역적인 특징(Feature Map)을 추출하는 핵심 레이어입니다.
* 특징
  * 작은 도장 같은 필터(Filter 또는 Kernel)가 이미지를 위에서부터 지나가며 연산을 수행합니다.

  * 초반 Conv 레이어는 선, 점, 경계선(Edge) 같은 단순한 특징을 찾고, 뒤쪽 Conv 레이어일수록 모양, 물체의 일부 등 복잡한 특징을 찾아냅니다.

2. 활성화 함수 레이어 (Activation Function)
* 역할: 신경망에 비선형성(Non-linearity)을 부여합니다.

* 특징:

  * Conv 레이어만 계속 쌓을 경우 비선형 연산이 불가능합니다.
  * CNN에서는 보통 ReLU(Rectified Linear Unit)0보다 작은 값은 0으로 바꾸고, 0보다 큰 값은 그대로 유지

3. 풀링 레이어 (Pooling Layer)

* 역할: 특징 맵의 가로·세로 크기를 줄여주는 레이어입니다.

* 종류:

  * Max Pooling: 특정 영역에서 가장 큰 값만 남깁니다.(가장 흔하게 사용됨)

  * Average Pooling(평균값 풀링): 특정 영역의 평균값을 남깁니다.

* 장점: 이미지 크기가 줄어들므로 계산량이 대폭 감소하고, 이미지의 위치가 살짝 바뀌더라도 특징을 잘 알아채는 위치 불변성(Translation Invariance)이 생깁니다.

4. 플래튼 레이어 (Flatten Layer)
* 역할: 추출된 2차원 특징맵(다차원 행렬)을 1차원 벡터로 펼쳐주는 역할입니다.

5. 완전 연결 레이어(Fully Connected Layer)
* 역할: 추출된 특징을 종합하여 최종 결과를 예측합니다.

Output 레이어에서 보통 Softmax 함수(다중 클래스 분류)나 Sigmoid 함수(이진 분류)를 붙여 최종 정답을 출력합니다.

### 한눈에 보는 요약표

| 레이어 (Layer) | 주요 역할 | 핵심 특징 및 키워드 |
| :--- | :--- | :--- |
| **Convolution Layer** | 이미지의 주요 특징(Feature) 추출 | • 필터(Filter/Kernel)가 이미지를 훑으며 연산<br>• 초반: 선/점 추출, 후반: 복잡한 형태 추출 |
| **Activation Layer (ReLU)** | 신경망에 비선형성(Non-linearity) 부여 | • $0$ 이하의 값은 $0$으로, 양수는 그대로 전달<br>• 복잡한 이미지 패턴 학습에 필수 |
| **Pooling Layer** | 특징 맵의 크기 축소 (Down-sampling) | • **Max Pooling**: 가장 도드라지는 특징만 선택<br>• 연산량 감소 및 위치 변형에 강해짐 |
| **Flatten Layer** | 다차원 데이터를 1차원 데이터로 변환 | • 2D/3D 특징 맵을 한 줄로 쫙 펼침<br>• Fully Connected Layer 연결 전 필수 단계 |
| **Fully Connected Layer** | 추출된 특징을 종합하여 최종 클래스 분류 | • 일반적인 인공신경망(ANN) 구조<br>• Softmax/Sigmoid를 통해 최종 확률 출력 |

