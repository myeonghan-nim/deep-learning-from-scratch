# chapter03. Neural Network

- Perceptron은 복잡한 함수를 표현할 수 있는 장점이 있지만 가중치를 사람이 일일히 설정해야 합니다.

- 하지만 이 과정을 **nueral network**, 즉 신경망을 사용해 해결할 수 있습니다.

## 3.1 Perceptron에서 neural network으로

### 3.1.1 Neural network의 예

<img src="README.assets/fig 3-1.png" alt="fig 3-1" style="zoom:50%;" />

- 위의 그림에서 좌측 줄을 **입력층**, 우측 줄을 **출력층**, 가운데를 **은닉층**이라고 부릅니다.

> 위의 그림은 신경망의 하나의 예시로 실제로 더 많은 층이 복잡하게 얽힌 경우가 많습니다.

### 3.1.2 Perceptron 복습

- Perceptron은 직전 노드에서 입력 받은 신호에 가중치를 곱한 값이 한계치를 넘을 경우 1을 출력하는 구조를 가지고 있습니다.

<img src="README.assets/fig 3-2.png" alt="fig 3-2" style="zoom:50%;" />

<img src="README.assets/e 3.1.png" alt="e 3.1" style="zoom:50%;" />

- 여기서 **b**, 즉 **편향 매개변수**는 해당 뉴런이 얼마나 쉽게 활성화 되는지를 제어합니다.

<img src="README.assets/fig 3-3.png" alt="fig 3-3" style="zoom:50%;" />

- 만일, 가중치가 `b`이고 입력이 `1`인 뉴런이 입력에 추가되면 **편향**이 추가된 값이 뉴런의 한계를 넘는지 확인해야 합니다.

- 이를 일반화하여 이전 뉴런에서 온 입력들의 합이 현재 뉴런에서 **활성화 함수**를 만나면 입력이 변환되어 그 값이 출력됩니다.

<img src="README.assets/e 3.2.png" alt="e 3.2" style="zoom:50%;" />

<img src="README.assets/e 3.3.png" alt="e 3.3" style="zoom:50%;" />

### 3.1.3 활성화 함수의 등장

- 앞서 등장한 활성화 함수는 **입력 신호의 총합을 출력 신호로 변환하는 함수**를 의미합니다.

<img src="README.assets/e 3.4.png" alt="e 3.4" style="zoom:50%;" />

<img src="README.assets/e 3.5.png" alt="e 3.5" style="zoom:50%;" />

- 즉, 활성화 함수는 **입력 신호의 총합이 뉴런의 활성화를 일으키는가**를 결정합니다.

<img src="README.assets/fig 3-4.png" alt="fig 3-4" style="zoom: 50%;" />

<img src="README.assets/fig 3-5.png" alt="fig 3-5" style="zoom:50%;" />

- 위의 그림처럼 활성화 함수는 각 노드 안에서 직전 노드에서 들어온 신호의 합을 처리하고 출력을 제어합니다.

#### 주의사항

- 모든 딥러닝 알고리즘에 사용되는 노드를 모두 perceptron이라고 칭하면 엄밀히 틀립니다.

- **단순 perceptron**의 경우 단층 네트워크에서 계단함수(임계값을 경계로 출력이 바뀌는 함수)를 활성화 함수로 사용합니다.

- 반면에 **다층 perceptron**의 경우 여러 층으로 구성되고 시그모이드 함수 등을 활성화 함수로 사용하는 네트워크를 의미합니다.

## 3.2 활성화 함수

> 어떤 활성화 함수를 사용하는가에 따라 신경망 구성이 달라질 수 있습니다.

### 3.2.1 시그모이드 함수

- **시그모이드 함수**란 다음 함수를 의미합니다.

<img src="README.assets/e 3.6.png" alt="e 3.6" style="zoom:50%;" />

> 3.2.x_~.py 중 선형 함수의 그래프는 다음을 참고하면 됩니다.

<img src="README.assets/fig 3-6-1578120439908.png" alt="fig 3-6" style="zoom:50%;" />

> 3.2.3_step_function.py

<img src="README.assets/fig 3-7.png" alt="fig 3-7" style="zoom:50%;" />

> 3.2.4_sigmoid_function.py

<img src="README.assets/fig 3-8.png" alt="fig 3-8" style="zoom:50%;" />

> 3.2.5_sigmoid_vs_step.py

### 3.2.6 비선형 함수

- 특정 입력에 대한 출력 값이 입력의 배수만큼 변하는 함수를 **선형 함수**라고 합니다.

- 하지만 계단 함수, 시그모이드 함수 같이 선형이 아닌 모든 함수를 **비선형 함수**라고 합니다.

- 신경망은 비선형 함수를 활성 함수로 사용합니다.

    - 선형 함수는 아무리 함수를 중첩해도 결국 선형적인 결과 밖에 얻지 못합니다.

    ```
    y = h(h(h(x))) = c * c * c * x = c ** 3 * x = a * x
    ```

    - 반면에 비선형 함수는 다양한 값을 통해 신경망을 발달시킬 수 있습니다.

### 3.2.7 ReLU 함수

> 3.2.7_relu.py를 참고하면 됩니다.

- ReLU 함수의 그래프와 그 결과는 다음과 같습니다.

<img src="README.assets/fig 3-9.png" alt="fig 3-9" style="zoom:50%;" />

> ReLU 함수의 그래프

<img src="README.assets/e 3.7.png" alt="e 3.7" style="zoom:50%;" />

> ReLU 함수의 수식

## 3.3 다차원 배열의 계산

> 3.3_calculation_of_nth_mat.py를 참고하면 됩니다.

- 2차원 배열은 **행렬**이라는 이름으로도 불리며 다음과 같이 **행**과 **열**로 정의합니다.

<img src="README.assets/fig 3-10.png" alt="fig 3-10" style="zoom:50%;" />

- 행렬의 곱은 다음과 같이 계산합니다.

<img src="README.assets/fig 3-11.png" alt="fig 3-11" style="zoom:50%;" />

- 다만, 행렬의 곱은 대응하는 차원의 원소 수를 일치시켜야 합니다. 이는 어떤 차원에 대해서도 적용됩니다.

<img src="README.assets/fig 3-12.png" alt="fig 3-12" style="zoom:50%;" />

<img src="README.assets/fig 3-13.png" alt="fig 3-13" style="zoom:50%;" />

- 이를 신경망 단위로 넘기면 다음과 같습니다.

<img src="README.assets/fig 3-14.png" alt="fig 3-14" style="zoom:50%;" />

## 3.4 3층 신경망 구현하기

> 3.4_multi_layer_neural_network.py를 참고하면 됩니다.

- 해당 신경망을 그림으로 나타내면 아래와 같습니다.

<img src="README.assets/fig 3-15.png" alt="fig 3-15" style="zoom:50%;" />

#### Note

- 여기서 중요한 표기법이 나오는 데 본 장에서만 사용할 것이므로 보고 잊어도 되는 형태입니다.

<img src="README.assets/fig 3-16.png" alt="fig 3-16" style="zoom:50%;" />

> 앞 계층에서 뒷 계층으로 신호가 전달될 때 전달을 나타내는 방식

### 3.4.2 각 층의 신호 전달 구현하기

- 각 층의 전달 과정을 나타내면 다음과 같습니다.

1. 입력층에서 1층으로 신호를 전달할 때

<img src="README.assets/fig 3-17.png" alt="fig 3-17" style="zoom:50%;" />

> 관련 수식은 다음과 같습니다.

<img src="README.assets/e 3.8.png" alt="e 3.8" style="zoom:50%;" />

<img src="README.assets/e 3.9.png" alt="e 3.9" style="zoom:50%;" />

2. 1층의 활성화 함수가 처리할 때

<img src="README.assets/fig 3-18.png" alt="fig 3-18" style="zoom:50%;" />

3. 1층에서 2층으로 신호를 전달할 때

<img src="README.assets/fig 3-19.png" alt="fig 3-19" style="zoom:50%;" />

4. 2층에서 출력층으로 신호를 전달할 때

<img src="README.assets/fig 3-20.png" alt="fig 3-20" style="zoom:50%;" />

## 3.5 출력층 설계하기

> 3.5_simulate_output_layer.py를 참고하면 됩니다.

- 신경망은 분류와 회귀 모두에 사용 가능합니다.

    - 그 중 분류의 경우 출력 함수로 소프트맥스 함수를 사용합니다.

    - 반면, 회귀의 경우 출력 함수로 항등 함수를 사용합니다.

### 3.5.1 항등 함수와 소프트맥스 함수

1. 항등 함수

- **항등 함수**는 입력을 그대로 출력하는 함수입니다.

<img src="README.assets/fig 3-21.png" alt="fig 3-21" style="zoom:50%;" />

2. 소프트맥스 함수

- **소프트맥스 함수**는 모든 입력에 영향을 받는 함수로 다음과 같은 식을 가집니다.

<img src="README.assets/e 3.10.png" alt="e 3.10" style="zoom:50%;" />

- 이를 그림으로 나타내면 아래와 같습니다.

<img src="README.assets/fig 3-22.png" alt="fig 3-22" style="zoom:50%;" />

### 3.5.2 소프트맥스 함수 구현 시 주의점

- 다만 소프트맥스는 지수 함수의 형태이므로 숫자가 매우 커져 overflow 현상이 벌어질 가능성이 높습니다.

- 따라서 이를 해결하기 위해 지수 함수 내부에 log 형태의 임의의 최대값을 추가함으로서 overflow를 막습니다.

<img src="README.assets/e 3.11.png" alt="e 3.11" style="zoom:50%;" />

### 3.5.3 소프트맥스 함수의 특징

- 최대값을 사용해 overflow를 막은 소프트맥스 함수는 각 출력이 선택될 확률과 동일한 의미를 가집니다.

    - 따라서 분류를 실행할 경우 소프트맥스 함수는 가장 높은 확률을 가지는 클래스를 정답으로 추론합니다.

    - 다만, 소프트맥스 함수는 입력의 대소 관계가 유지되므로 자원 소모를 줄이기 위해 생략할 수 있습니다.

### 3.5.4 출력층의 뉴런 수 정하기

- 출력층의 뉴런 수는 풀려는 문제에 맞게 적절히 설정하면 됩니다.

<img src="README.assets/fig 3.23.png" alt="fig 3.23" style="zoom:50%;" />

## 3.6 손글씨 숫자 인식

> 주의사항
>
> 본 책과는 다르게 autopep8을 설치한 경우 해당하는 chapter에 dataset에 있는 mnist.py를 복사하여 사용해야 합니다.
>
> 그 외에도 자잘한 변화가 필요하므로 사용자의 주의가 필요합니다.

### 3.6.1 MNIST 데이터셋

> 3.6.1_mnist_dataset.py를 참고하면 됩니다.

- MNIST는 손글씨 숫자 이미지 집합으로 다음과 같이 구성되어 있습니다.

<img src="README.assets/fig 3-24.png" alt="fig 3-24" style="zoom:50%;" />

<img src="README.assets/fig 3-25.png" alt="fig 3-25" style="zoom:50%;" />

> 좀 더 자세하게 임의의 하나를 출력한 예제입니다.

### 3.6.2 신경망의 추론 처리

> 3.6.2_neural_network_with_mnist.py를 참고하면 됩니다.

### 3.6.3 배치 처리

> 3.6.3_neural_network_with_mnist_batch.py를 참고하면 됩니다.

- 신경망 각 층이 배열 형상을 바꾸는 과정은 일반적으로 다음과 같습니다.

<img src="README.assets/fig 3-26.png" alt="fig 3-26" style="zoom:50%;" />

- 이를 배치로 처리하면 다음과 같이 진행됩니다.

<img src="README.assets/fig 3-27.png" alt="fig 3-27" style="zoom:50%;" />

### Tip

- `axis`는 다차원의 배열이 존재하는 경우 그 중 몇 번째 차원을 구성하는 원소들을 대상으로 할 지 결정하는 parameter입니다.

```python
import numpy as np

x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
y = np.argmax(x, axis=1)
print(y)  # [1, 2, 1]
```

## 3.7 정리

- 신경망은 활성화 함수로 시그모이드, Relu 함수와 같이 매끄럽게 변화하는 함수를 사용합니다.

- NumPy의 다차원 배열을 잘 구사하면 효율적인 신경망을 짤 수 있습니다.

- 머신러닝은 크게 회귀문제와 분류문제로 구분할 수 있습니다.

- 출력층의 활성화 함수는 항등 함수(회귀), 소프트맥스 함수(분류)를 사용합니다.

- 분류문제에서 출력층의 뉴런 수는 분류하려는 클래스의 수에 의해 결정됩니다.

- 입력 데이터의 묶음을 배치라고 부르며 추론을 배치 단위로 진행하면 더 빠른 결과를 얻을 수 있습니다.
