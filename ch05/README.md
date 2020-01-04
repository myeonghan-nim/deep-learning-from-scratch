# chapter05. Backpropagation

- 오차역전파법은 가중치의 기울기를 효율적으로 계산하는 방법입니다.

- 오차역전파법은 수식을 통한 계산과 계산 그래프를 통한 계산 두 가지 방법이 있습니다.

## 5.1 계산 그래프

- 계산 그래프란 계산 과정을 그래프로 나타낸 것을 의미합니다.

    - 그래프는 자료구조 중 하나로 **node**와 **edge**로 구성되어 있습니다.

### 5.1.1 계산 그래프로 풀다

- 예시

    - 1개에 100원인 사과 2개를 살 때 소비세 10%가 가산된다면 가격은 얼마인가요?

<img src="README.assets/fig 5-1.png" alt="fig 5-1" style="zoom:50%;" />

> 위의 문제를 해결하는 과정을 나타낸 그림입니다.

- 이 문제를 더 상세히 풀어쓰면 사과의 개수와 소비세에 대한 항목도 추가할 수 있습니다.

<img src="README.assets/fig 5-2.png" alt="fig 5-2" style="zoom:50%;" />

- 이 문제를 더 확장해 귤을 사는 경우도 추가하면 다음과 같습니다.

<img src="README.assets/fig 5-3.png" alt="fig 5-3" style="zoom:50%;" />

- 계산 그래프는 위와 같이 계산 과정을 노드와 화살표로 표현합니다.

    - 노드에는 연산 내용을 화살표에는 연산 결과를 적습니다.

- 계산 과정이 복잡해지면 노드와 화살표가 늘어나며 그래프가 복잡해지지만 순서대로 계산하면 원하는 값을 얻을 수 있습니다.

#### Tip

- 계산 그래프의 문제풀이는 다음과 같이 진행됩니다.

    1. 계산 그래프를 구성합니다.

    2. 그래프를 왼쪽부터 오른쪽으로 진행하며 계산합니다.

        - 이렇게 순서대로 계산하는 과정을 **순전파**라고 합니다.

        - 반대로 역으로 계산하는 과정을 **역전파**라 부르며 이 과정을 딥러닝 계산에 사용합니다.

### 5.1.2 국소적 계산

- 계산 그래프는 국소적으로 작은 범위를 계산하고 그 결과를 전파하여 최종 결과를 얻는 과정입니다.

<img src="README.assets/fig 5-4.png" alt="fig 5-4" style="zoom:50%;" />

- 즉, 최종 결과는 이전에 복잡한 과정을 전부 생각할 필요 없이 바로 직전의 결과들만 연산하면 됩니다.

### 5.1.3 왜 계산 그래프로 푸는가?

- 계산 그래프로 문제를 해결하는 이유는 국소적 계산 덕분에 문제를 단순화할 수 있다는 점입니다.
- 또한 중간 계산 결과를 모두 보관하여 필요한 경우 활용할 수 있습니다.
- 가장 중요한 점으로 역전파를 통해 미분을 효율적으로 계산할 수 있다는 점입니다.

<img src="README.assets/fig 5-5.png" alt="fig 5-5" style="zoom:50%;" />

> 역전파는 순전파와 반대로 굵은 선을 그려 나타냅니다.

## 5.2 연쇄법칙

### 5.2.1 계산 그래프의 역전파

<img src="README.assets/fig 5-6.png" alt="fig 5-6" style="zoom:50%;" />

- 계산 그래프의 역자파 계산 절차는 신호 E에 노드의 국소 미분을 곱한 값을 다음 노드로 전달하는 것입니다.

    - 여기서 국소 미분은 순전파 함수릐 미분을 의미합니다.(예를 들어 `x ** 2`의 미분은 `2 * x`)

### 5.2.2 연쇄법칙이란?

- 연쇄법칙을 설명하기 전에 **합성 함수**를 알아야합니다.

    - 합성 함수란 여러 함수로 구성된 함수를 의미합니다.

    <img src="README.assets/e 5.1.png" alt="e 5.1" style="zoom:50%;" />

    > 예시로 이는 `z = (x + y) ** 2`와 동일합니다.

- **연쇄법칙**은 합성 함수의 미분에 대한 성질로 다음과 같습니다.

> 합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 표현할 수 있다.

- 이를 수식으로 표현하면 다음과 같은 과정을 거칩니다.

<img src="README.assets/e 5.2.png" alt="e 5.2" style="zoom:50%;" />

> x에 대해 편미분한 수식은 위와 같습니다.

<img src="README.assets/e 5.3.png" alt="e 5.3" style="zoom:50%;" />

> 각각 변수에 대해 편미분한 결과는 위와 같습니다.

<img src="README.assets/e 5.4.png" alt="e 5.4" style="zoom:50%;" />

> 이를 x, y 모두에 대해 처리한 결과는 위와 같습니다.

### 5.2.3 연쇄법칙과 계산 그래프

- 위의 계산을 그래프로 표현하면 다음과 같습니다.

<img src="README.assets/fig 5-7.png" alt="fig 5-7" style="zoom:50%;" />

> 편의상 y에 대해서는 생력했습니다.

- 즉, 역전파는 연쇄법칙과 동일한 과정을 통해 진행됨을 알 수 있습니다.

    <img src="README.assets/fig 5-8.png" alt="fig 5-8" style="zoom:50%;" />
    
    - 위의 그래프를 예시로 들면 `dz / dt = 2 * t`이므로 `dz / dx = 2(x + y)`임을 알 수 있습니다.

## 5.3 역전파

### 5.3.1 덧셈 노드의 역전파

- 덧셈 노드의 미분은 모든 상황에서 1이 됩니다.

  <img src="README.assets/e 5.5.png" alt="e 5.5" style="zoom:50%;" />

- 즉, 덧셈 노드의 역전파는 입력 신호를 그대로 전달하는 역할을 합니다.

  <img src="README.assets/fig 5-9.png" alt="fig 5-9" style="zoom:50%;" />

  > 위의 그래프를 추상화하면 다음과 같습니다.

  <img src="README.assets/fig 5-10.png" alt="fig 5-10" style="zoom:50%;" />

  > 위의 그래프를 상세한 예시를 들면 다름과 같습니다.

  <img src="README.assets/fig 5-11.png" alt="fig 5-11" style="zoom:50%;" />

### 5.3.2 곱셈 노드의 역전파

- 곱셈 노드의 미분은 서로의 순전파 입력값을 바꾼 결과가 됩니다.

  <img src="README.assets/e 5.6.png" alt="e 5.6" style="zoom:50%;" />

- 따라서, 곱셈 노드의 역전파는 입력 신호에 순전파 신호를 곱하고 그를 서로 바꾸어 전달합니다.

  <img src="README.assets/fig 5-12.png" alt="fig 5-12" style="zoom:50%;" />

  > 위의 그래프를 상세한 예시를 들면 다음과 같습니다.

  <img src="README.assets/fig 5-13.png" alt="fig 5-13" style="zoom:50%;" />

### 5.3.3 사과 쇼핑의 예

<img src="README.assets/fig 5-14.png" alt="fig 5-14" style="zoom:50%;" />

> 처음 나온 사과 쇼핑의 예를 역전파까지 표현하면 위와 같습니다.
>
> 이를 토대로 사과와 귤을 쇼핑하는 예를 완성하면 역전파에 대해 완벽히 이해할 수 있습니다.

<img src="README.assets/fig 5-15.png" alt="fig 5-15" style="zoom:50%;" />

## 5.4 단순한 계층 구현하기

### 5.4.1 곱셈 계층

> 5.4.1_multiple_layer.py를 참고하면 됩니다.

- 해당 파일을 시각화하면 다음과 같습니다.

<img src="README.assets/fig 5-16.png" alt="fig 5-16" style="zoom:50%;" />

### 5.4.2 덧셈 계층

> 5.4.2_plus_layer.py를 참고하면 됩니다.

- 해당 파일을 시각화하면 다음과 같습니다.

<img src="README.assets/fig 5-17.png" alt="fig 5-17" style="zoom:50%;" />

## 5.5 활성화 함수 계층 구현하기

### 5.5.1 ReLU 계층

- 해당 계층에 사용되는 수식은 다음과 같습니다.

<img src="README.assets/e 5.7.png" alt="e 5.7" style="zoom:50%;" />

> 수식

<img src="README.assets/e 5.8.png" alt="e 5.8" style="zoom:50%;" />

> 미분 수식

- 이를 계산 그래프로 나타내면 다음과 같습니다.

<img src="README.assets/fig 5-18.png" alt="fig 5-18" style="zoom:50%;" />

### 5.5.2 Sigmoid 계층

- Sigmoid 함수를 다시 상기시키면 다음과 같습니다.

<img src="README.assets/e 5.9.png" alt="e 5.9" style="zoom:50%;" />

- 이를 계산 그래프로 나타내면 다음과 같습니다.

<img src="README.assets/fig 5-19.png" alt="fig 5-19" style="zoom:50%;" />

- sigmoid 함수의 역전파를 계산하는 과정은 다음과 같습니다.

1. `/` 노드의 역전파

<img src="README.assets/e 5.10.png" alt="e 5.10" style="zoom:50%;" />

> 해당 노드의 미분 수식

<img src="README.assets/fig 5-19(1)-1578130386731.png" alt="fig 5-19(README.assets/fig 5-19(1)-1578130370246.png)" style="zoom:50%;" />

> 해당 노드의 역전파

2. `+` 노드의 역전파

<img src="README.assets/fig 5-19(2)-1578130428733.png" alt="fig 5-19(README.assets/fig 5-19(2)-1578130424435.png)" style="zoom:50%;" />

> 해당 노드의 역전파

3. `exp()` 노드의 역전파

<img src="README.assets/e 5.11.png" alt="e 5.11" style="zoom:50%;" />

> 해당 노드의 미분 수식

<img src="README.assets/fig 5-19(3)-1578130474101.png" alt="fig 5-19(README.assets/fig 5-19(3).png)" style="zoom:50%;" />

> 해당 노드의 역전파

4. `*` 노드의 역전파

<img src="README.assets/fig 5-20.png" alt="fig 5-20" style="zoom:50%;" />

> 해당 노드의 역전파

- 위의 단계를 전부 간소화하면 다음과 같이 나타낼 수 있습니다.

<img src="README.assets/fig 5-21.png" alt="fig 5-21" style="zoom:50%;" />

- 또한 최종 역전파 수식을 다음과 같이 간략하게 쓸 수 있습니다.

<img src="README.assets/e 5.12.png" alt="e 5.12" style="zoom:50%;" />

> 위의 수식을 계산 그래프로 나타내면 다음과 같습니다.

<img src="README.assets/fig 5-22.png" alt="fig 5-22" style="zoom:50%;" />

## 5.6 Affine/Softmax 계층 구현하기

### 5.6.1 Affine 계층

- 신경망의 순전파에서 가중치 신호의 총합을 행렬의 곱으로 나타내는 데 이를 **Affine 변환**이라고 부릅니다.

<img src="README.assets/fig 5-23.png" alt="fig 5-23" style="zoom:50%;" />

- 이를 계산 그래프로 나타내고 역전파를 계산하는 과정은 다음과 같습니다.

<img src="README.assets/fig 5-24.png" alt="fig 5-24" style="zoom:50%;" />

> Affine 계층의 계산 그래프

0. 행렬의 곱 역전파

<img src="README.assets/e 5.13.png" alt="e 5.13" style="zoom:50%;" />

> 여기서 `T`는 전치행렬로 원 행렬의 행과 열을 뒤집은 형태라고 생각하면 됩니다.

<img src="README.assets/e 5.14.png" alt="e 5.14" style="zoom:50%;" />

1. Affine 계층의 역전파

<img src="README.assets/fig 5-25.png" alt="fig 5-25" style="zoom:50%;" />

> 위의 계산 그래프를 통해 행렬 X와 미분값인 `dL / dX`가 형상이 동일함을 알 수 있습니다.

<img src="README.assets/e 5.15.png" alt="e 5.15" style="zoom:50%;" />

> 즉, 행렬의 곱 노드는 순전파와 역전파의 형상이 동일해야 합니다.

<img src="README.assets/fig 5-26.png" alt="fig 5-26" style="zoom:50%;" />

### 5.6.2 배치용 Affine 계층

- 위의 Affine 계층을 확장하여 배치에 사용가능 하도록 설계하면 다음과 같은 계산 그래프를 그립니다.

<img src="README.assets/fig 5-27.png" alt="fig 5-27" style="zoom:50%;" />

### 5.6.3 Softmax-with-Loss 계층

- 출력에 사용되는 Softmax 계층은 입력 값을 정규화 하는 과정을 거칩니다.

<img src="README.assets/fig 5-28.png" alt="fig 5-28" style="zoom:50%;" />

- 이 계층의 계산 과정은 매우 복잡하므로 계산 그래프로 표현하자면 다음과 같습니다.

<img src="README.assets/fig 5-29.png" alt="fig 5-29" style="zoom:50%;" />

> Softmax-with-Loss 계층의 계산 그래프

<img src="README.assets/fig 5-30.png" alt="fig 5-30" style="zoom:50%;" />

> Softmax-with-Loss 계층의 간소화 버전

## 5.7 오차역전파법 구현하기

### 5.7.1 신경망 학습의 전체 그림

- 신경망 학습 절차를 복습하면 다음과 같습니다.

0. 전제

    - 신경망에는 적용 가능한 가중치와 편향이 있고 이들을 훈련 데이터에 적응하도록 적응하는 과정(학습)이 필요합니다.

1. 미니배치

    - 훈련 데이터 중 일부를 무작위로 가져온 미니배치의 손실 함수를 줄이는 작업을 합니다.

2. 기울기 산출

    - 미니배치의 손실 함수 값을 줄이기 위해 가중치 매개변수의 기울기를 구합니다.

3. 매개변수 갱신

    - 매개변수를 기울기 방향으로 아주 조금 갱신합니다.

4. 반복

    - 1 ~ 3단계를 반복합니다.

### 5.7.2 오차역전파법을 적용한 신경망 구현하기

> 5.7.2_backpropagation_neural_network.py를 참조하면 됩니다.

- 해당 파일에서 사용되는 변수와 클래스는 다음과 같습니다.

1. 변수

| 변수 | 설명 |
| :---: | ----- |
| params | 신경망의 매개 변수를 보관하는 dict |
|       | W는 각 층의 가중치, b는 각 층의 편향을 의미 |
| layers | 신경망 계층을 보관하는 순서가 있는 dict |
|       | 1층부터 순서대로 각 계층을 순서대로 유지합니다. |
| lastLayer | 신경망의 마지막 계층 |

2. method

| method | 설명 |
| :---: | ----- |
| __init__ | 초기화를 수행하며 순서대로 입력층, 은닉층, 출력층 뉴런의 수, 가중치, 정규분포 스케일 |
| predict | 예측, x는 이미지 데이터 |
| loss | 손실 함수 값 출력, x는 이미지 데이터, t는 정답 |
| accuracy | 정확도 |
| numerical_gradient | 가중치 매개변수의 기울기 |
| gradient | numerical_gradient의 개선판으로 오차역전파법으로 구합니다. |

### 5.7.3 오차역전파법으로 구한 기울기 검증하기

> 5.7.3_backpropagation_gradient_check.py를 참조하면 됩니다.

- 느린 대신에 정확한 수치 미분을 통해 오차역전파법으로 계산한 기울기가 올바른지 검증할 수 있습니다.

    - 이 과정을 **기울기 확인**이라고 합니다.

> 다만 오차가 0이 되는 일은 드물기 때문에 매우 작은 수가 나오면 올바르게 구현했다고 할 수 있습니다.

### 5.7.4 오차역전파법을 사용한 학습 구현하기

> 5.7.4_backpropagation_learning.py를 참조하면 됩니다.

## 5.8 정리

- 계산 그래프를 이용하면 계산 과정을 시각적으로 파악할 수 있습니다.

- 계산 그래프의 노드는 국소적 계산으로 구성되면 이들을 조합해 전체 계산을 구성합니다.

- 계산 그래프의 순전파는 통상 계산을 수행하며 역전파는 각 노드의 미분을 구할 수 있습니다.

- 신경망의 구성 요소를 계층으로 구형해 기울기를 효율적으로 계산할 수 있습니다.(오차역전파법)

- 수치 미분과 오차역전파법의 결과를 비교하면 오차역전파법의 구현에 실수가 있었는지 알 수 있습니다.
