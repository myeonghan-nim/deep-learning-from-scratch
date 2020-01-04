# chapter04. Learning Neural Network

- 학습이란 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는 것을 말합니다.

- 이를 위해서 **손실 함수**를 잘 사용하는 것이 중요합니다.

    - 손실 함수의 결과를 가장 적게 내는 것이 목표입니다.

## 4.1 데이터에서 학습한다!

### 4.1.1 데이터 주도 학습

- 머신러닝은 **데이터**를 통해 모든 문제를 해결합니다.

<img src="README.assets/fig 4-1.png" alt="fig 4-1" style="zoom:50%;" />

> 제일 대표적인 예가 바로 MNIST를 구별하는 알고리즘입니다.

- 즉, 사람의 **직관**과 다르게 수집된 데이터에서 패턴을 분석하고 그를 통해 정답을 유추해 나갑니다.

- 데이터에서 패턴, 즉 **특징**을 추출하고 이 특징의 패턴을 학습하는 방식이 주로 사용됩니다.

    - 지도 학습에서 주로 사용되는 기법은 SVM, KNN 등이 존재합니다.

    <img src="README.assets/fig 4-2.png" alt="fig 4-2" style="zoom: 50%;" />

- 다만, 데이터의 특징은 사람이 설계해야하므로 문제에 적합한 특징을 찾아내어 문제를 해결하는 것이 중요합니다.


### 4.1.2 훈련 데이터와 시험 데이터

- 머신러닝은 **훈련 데이터**를 통해 학습하고 **시험 데이터**를 통해 실험을 수행합니다.

    - 훈련 데이터는 최적의 매개변수를 찾는 목적으로 사용됩니다.

    - 시험 데이터는 훈련된 모델의 실력을 평가하는 데 사용됩니다.

- 이렇게 데이터를 분리해서 사용하는 이유는 모델의 **범용 능력**을 평가하기 위해서 입니다.

    - 학습된 모델이 거의 모든 비슷한 문제에 사용되기 위해서는 범용 능력이 중요합니다.

    - 특히, 특정 데이터에만 적응된 모델은 다른 비슷한 데이터에서 에러를 발생할 수 있습니다.

    - 따라서, 데이터를 학습할 때는 **underfitting**과 **overfiting**이 일어나지 않도록 주의해야 합니다.

## 4.2 손실 함수

- 신경망은 하나의 지표를 기준으로 최적의 매개변수 값을 결정합니다.

- 이 때 사용되는 지표가 **손실 함수**입니다.

    - 손실 함수는 임의의 함수를 사용할 수 있으나 일반적으로 평균 제곱 오차와 교차 엔트로피 오차를 사용합니다.

### 4.2.1 평균 제곱 오차

- 가장 많이 쓰이는 함수로 신경망의 출력과 정답 레이블의 차이의 제곱의 합의 평균을 기준으로 합니다.

<img src="README.assets/e 4.1.png" alt="e 4.1" style="zoom:50%;" />

```python
import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


t = np.array([0, 0, 1, 0, 0])  # 2가 정답인 경우

y1 = np.array([0.1, 0.05, 0.8, 0.0, 0.05])  # 2로 측정
print(mean_squared_error(y1, t))  # 0.0274, 더 작은 평균 제곱 오차

y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.8])  # 5로 측정
print(mean_squared_error(y2, t))  # 0.7312
```

### 4.2.2 교차 엔트로피 오차

- 교차 엔트로피 오차는 정답 레이블의 값에 자연로그를 취한 출력값을 곱하고 이를 합친 것의 음수를 기준으로 합니다.

<img src="README.assets/e 4.2.png" alt="e 4.2" style="zoom:50%;" />

- 이 방식을 사용하면 실질적으로 정답일 때 모델이 출력한 값을 평가합니다.

<img src="README.assets/fig 4-3.png" alt="fig 4-3" style="zoom:50%;" />

```python
import numpy as np


def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-7))  # 자연로그가 무한대가 나오는 걸 방지


t = np.array([0, 0, 1, 0, 0])  # 2가 정답인 경우

y1 = np.array([0.1, 0.05, 0.8, 0.0, 0.05])  # 2로 측정
print(cross_entropy_error(y1, t))  # 0.2231, 더 작은 교차 엔트로피 오차

y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.8])  # 5로 측정
print(cross_entropy_error(y2, t))  # 2.3025
```

### 4.2.3 미니배치 학습

- 훈련 데이터가 많을 경우 손실 함수는 각 오차를 합친 뒤 정규화해서 오차를 구할 수 있습니다.

<img src="README.assets/e 4.3.png" alt="e 4.3" style="zoom:50%;" />

- 다만 데이터가 너무 많은 경우 일일히 계산하기에 너무 많은 시간이 걸리므로 일부의 데이터만 학습하는 **미니배치 학습**을 진행합니다.

```python
import numpy as np

batch_size = 10
batch_mask = np.random.choice(60000, batch_size)
```

### 4.2.4 배치용 교차 엔트로피 오차 구현하기

- 미니배치와 같은 데이터의 손실 함수는 다음과 같이 구현할 수 있습니다.

```python
import numpy as np


def cross_entropy_error(y, t):  # one hot encoding ver
    if y.dim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error(y, t):  # one hot encoding ver
    if y.dim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```

### 4.2.5 왜 손실 함수를 설정하는가?

- 손실 함수를 설정하는 이유는 정확도를 높이기 위해서 입니다.

- 다만, 신경망 학습은 최적 매개변수를 탐색할 때 손실 함수의 값을 적게하기 위해 미분을 사용해 값을 갱신합니다.

- 하지만 미분 값이 0이되어 더 이상 매개변수를 갱신하지 않는 지점은 여러 곳이 있을 수 있으므로 단순히 정확도를 기준으로 삼지 않습니다.

    - 즉, 매개변수가 조금 바뀌어도 거의 변하지 않는 정확도보다 손실 함수가 모델 성능을 높이는 데 유리합니다.

    <img src="README.assets/fig 4-4.png" alt="fig 4-4" style="zoom:50%;" />
    
    - 이는 활성화 함수로 계단 함수를 사용하지 않는 이유와 동일합니다.

## 4.3 수치 미분

- **경사법**은 기울기 값을 기준으로 매개변수를 정합니다.

### 4.3.1 미분

- **미분**은 특정 시간의 변화량을 의미합니다.

<img src="README.assets/e 4.4.png" alt="e 4.4" style="zoom:50%;" />

```python
def numerical_diff(f, x):
    return (f(x + 10e-50) - f(x)) / 10e-50
```

- 위 방식대로 미분을 구현하면 오차가 발생합니다.

    1. 컴퓨터는 매우 작은 값을 반올림 오차로 인해 0으로 간주합니다.

    2. 컴퓨터는 무한히 0으로 좁히는 방식이 불가능합니다.

- 이를 해결하는 방식이 **중심 차분**을 쓰는 방법입니다.

<img src="README.assets/fig 4-5.png" alt="fig 4-5" style="zoom:50%;" />

```python
def numerical_diff(f, x):
    return (f(x + 1e-4) - f(x - 1e-4)) / 2e-4
```

> 하지만 위와 같은 중심차분을 통해 계산하면 오차를 줄일 수 있습니다.

### 4.3.2 수치 미분의 예

> 4.3.2_numerical_diff.py를 참고하면 됩니다.

<img src="README.assets/e 4.5.png" alt="e 4.5" style="zoom:50%;" />

> 예시로 사용하는 함수입니다.

```python
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    return (f(x + 1e-4) - f(x - 1e-4)) / 2e-4


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

x = np.arange(0, 20, 0.1)
y = function_1(x)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()

numerical_diff(function_1, 5)
numerical_diff(function_1, 10)
```

<img src="README.assets/fig 4-6.png" alt="fig 4-6" style="zoom:50%;" />

> 위는 해당 함수의 그래프입니다.

<img src="README.assets/fig 4-7.png" alt="fig 4-7" style="zoom:50%;" />

> 해당 함수를 `x = 5`, `x = 10`에서 미분한 결과입니다.

### 4.3.3 편미분

- 편미분은 두 개 이상의 변수가 존재하는 경우 하나를 무시하고 다른 하나에 대해 미분하는 방식을 의미합니다.

<img src="README.assets/e 4.6.png" alt="e 4.6" style="zoom:50%;" />

> 예시로 사용하는 함수입니다.

```python
import numpy as np


def function_2(x):
    return np.sum(x ** 2)


def function_tmp1(x0):  # diff of x0
    return x0 ** 2 + 4.0 ** 2

def function_tmp2(x1):  # diff of x1
    return 3.0 ** 2 + x1 ** 2


```

<img src="README.assets/fig 4-8.png" alt="fig 4-8" style="zoom:50%;" />

> 해당 함수를 그리면 다음과 같은 그래프를 나타냅니다.

## 4.4 기울기

> 4.4_slope.py를 참고하면 됩니다.

- 앞서서 한 편미분을 동시에 진행하려면 편미분 값을 벡터로 모으면 됩니다. 이를 기울기라고 부릅니다.

```python
import numpy as np


def numerical_gradient(f, x):
    h, g = 1e-4, np.zeros_like(x)

    for idx in range(x.size):
        tval = x[idx]

        x[idx] = tval + h
        fxh1 = f(x)

        x[idx] = tval - h
        fxh2 = f(x)

        g[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tval

    return g


```

<img src="README.assets/fig 4-9.png" alt="fig 4-9" style="zoom:50%;" />

> 위의 파일을 실행한 결과입니다.

- 여기서 기울기가 가르키는 방향은 **각 장소에서 함수의 출력을 가장 크게 줄이는 방향**입니다.

### 4.4.1 경사법(경사 하강법)

> 4.4.1_gradient_method.py를 참고하면 됩니다.

- 손실 함수를 최소로 줄이는 방법 중 하나가 바로 기울기를 활용한 경사법입니다.

    - 다만, 함수가 복잡할 경우 **안장점**과 같은 고원이 있어 기울기를 0으로 만드는 장소가 될 수도 있습니다.

    - 따라서 학습을 진행할 때 정말 최소값을 찾은 것인지 검증이 필요합니다.
- **경사법**은 기울어진 방향으로 나아가며 함수 값을 줄여나가는 것을 목표로 합니다.

<img src="README.assets/e 4.7.png" alt="e 4.7" style="zoom:50%;" />

> 수식으로 위와 같이 표현합니다.

- 신경망 학습의 경우 **학습률**, 즉 매개변수 값을 얼마나 갱신할 지 결정하며 학습합니다.

    - 다만 학습률은 너무 크거나 작을 경우 잘못된 값을 도출할 수 있으니 주의해야 합니다.

<img src="README.assets/fig 4-10.png" alt="fig 4-10" style="zoom:50%;" />

> 해당 파일을 실행한 결과입니다.

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        g = numerical_gradient(f, x)
        x -= lr * g
    
    return x
```

- 여기서 학습률 같은 매개변수를 **하이퍼 파라미터**라고 부릅니다.

    - 이러한 하이퍼 파라미터는 컴퓨터가 설정하지 않고 사람이 직접 지정해주어야 합니다.

### 4.4.2 신경망에서의 기울기

> 4.4.2_gradient_in_neural.py를 참고하면 됩니다.

- 신경망 학습에서도 기울기를 구해야 합니다. 이는 다음과 같이 표현할 수 있습니다.

<img src="README.assets/e 4.8.png" alt="e 4.8" style="zoom:50%;" />

## 4.5 학습 알고리즘 구현하기

- 신경망 학습 절차는 다음과 같습니다.

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

> 위와 같이 미니배치로 갱신하는 방식을 **확률적 경사 하강법**이라고 합니다.

### 4.5.1 2층 신경망 클래스 구현하기

> 4.5.1_2nd_layer_neural_class.py를 참고하면 됩니다.

- 해당 파일에서 사용되는 변수와 클래스는 다음과 같습니다.

1. 변수

| 변수 | 설명 |
| :---: | ----- |
| params | 신경망의 매개 변수를 보관하는 dict |
|       | W는 각 층의 가중치, b는 각 층의 편향을 의미 |
| grads | 기울기를 보관하는 dict |
|       | W는 가중치의 기울기, b는 편향의 기울기를 의미 |

2. method

| method | 설명 |
| :---: | ----- |
| __init__ | 초기화를 수행하며 순서대로 입력층, 은닉층, 출력층 뉴런의 수 |
| predict | 예측, x는 이미지 데이터 |
| loss | 손실 함수 값 출력, x는 이미지 데이터, t는 정답 |
| accuracy | 정확도 |
| numerical_gradient | 가중치 매개변수의 기울기 |
| gradient | numerical_gradient의 개선판 |

### 4.5.2 미니배치 학습 구현하기

> 4.5.2_mini_batch_learning.py를 참고하면 됩니다.

- 시험 데이터로 평가하기 전 학습 그래프는 다음과 같습니다.

<img src="README.assets/fig 4-11.png" alt="fig 4-11" style="zoom:50%;" />

### 4.5.3 시험 데이터로 평가하기

> 4.5.2_mini_batch_learning.py를 참고하면 됩니다.

- 시험 데이터로 평가한 후 평가 그래프는 다음과 같습니다.

<img src="README.assets/fig 4-12.png" alt="fig 4-12" style="zoom:50%;" />

## 4.6 정리

- 머신러닝에 사용하는 데이터들은 훈련 데이터와 시험 데이터로 나누어 사용합니다.

- 훈련 데이터로 학습한 모델의 범용 능력을 시험 데이터로 평가합니다.

- 신경망 학습은 손실 함수를 지표로 손실 함수의 값이 작아지는 방향으로 가중치 매개변수를 갱신합니다.

- 가중치를 갱신할 때는 가중치의 기울기를 이용해 그 방향으로 가중치를 갱신해 나갑니다.

- 아주 작은 값을 주었을 때 차분으로 미분하는 걸 수치 미분이라고 합니다.

- 수치 미분을 통해 가중치의 기울기를 구할 수 있습니다.

- 수치 미분을 통해 계산하면 시간이 걸리지만 구현이 간단합니다.

    - 다음 장에서 배울 오차역전파법은 기울기를 고속으로 구할 수 있습니다.
