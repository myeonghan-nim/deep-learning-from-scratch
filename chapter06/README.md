# chapter06. Technics about learning

## 6.1 매개변수 갱신

- 신경망 학습의 목적은 손실 함수의 값을 가능한 낮추는 매개변수를 찾는 것입니다.

- 이 과정을 **최적화**라고 하며 매우 어려운 과정입니다.

- 지금까지 최적 매개변수를 찾는 방법은 매개변수의 기울기를 사용한 방법으로 이를 **확률적 경사 하강법(SGD)**이라고 합니다.

- 하지만 문제에 따라 SGD보다 더 나은 방법도 존재합니다. 지금부터 이 방법들에 대해 알아봅시다.

### 6.1.1 모험가 이야기

- 이 이후에 나오는 방법들은 전부 이 이야기에 기반합니다.

> 색다른 모험가가 전설에 나오는 세상에서 가장 깊고 낮은 골짜기를 찾으려 하고 있습니다.
>
> 다만 그는 지도를 보지 않고 눈가리개를 쓴 채로 이 곳을 찾으려고 합니다.
>
> 그럼 이 모험가가 어떻게 깊은 곳을 찾아낼 수 있을까요?

- 지금까지 사용한 방법은 땅의 기울기를 통해 가장 깊은 곳을 찾는 것이었습니다.

### 6.1.2 확률적 경사 하강법(SGD)

- SGD는 가중치 매개변수에 그 매개변수의 손실 함수 기울기와 학습률을 곱한 값을 통해 갱신하는 과정입니다.

  - 이를 수식으로 나타내면 다음과 같습니다.

<img src="README.assets/e 6.1.png" alt="e 6.1" style="zoom:50%;" />

- 이를 python으로 구현하면 다음과 같습니다.

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr  # 학습률

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```

- 위의 SGD 클래스를 사용하면 다음과 같이 매개변수를 갱신할 수 있습니다.

```python
network = TwoLayerNet(...)
optimizer = SGD()

for i in range(10000):
    ...
    x_batch, t_batch = get_mini_batch(...)  # 미니배치
    grads = network.gradient(x_batch, t_batch)
    params = network.params
    optimizer.update(params, grads)
    ...
```

### 6.1.3 SGD의 단점

- SGD는 단순하고 구현도 쉽지만 문제에 따라서 비효율적일 수 있습니다.

- 예를 들어 다음과 같은 함수가 있을 경우 기울기는 `y = 0`인 모든 `x`에 대해 기울기가 최소라는 점입니다.

<img src="README.assets/e 6.2.png" alt="e 6.2" style="zoom:50%;" />

- 하지만 실제 최소가 되는 점은 `x, y = 0, 0`인 지점입니다.

<img src="README.assets/fig 6-1.png" alt="fig 6-1" style="zoom:50%;" />

> 예시 함수의 그래프 모델링

- 이 함수의 기울기를 구현하면 다음과 같은 그림이 나옵니다.

<img src="README.assets/fig 6-2.png" alt="fig 6-2" style="zoom:50%;" />

- 만일 이 함수를 SGD로 구현하면 (0, 0)까지 도달할 수 있을지라도 진동하며 움직이는 비효율적인 움직임을 보여줍니다.

<img src="README.assets/fig 6-3.png" alt="fig 6-3" style="zoom:50%;" />

- 따라서 이 문제를 해결하기 위해서 모멘텀, AdaGrad, Adam과 같은 방법 들이 있습니다.

### 6.1.4 모멘텀

- **모멘텀**은 운동량을 의미하는 단어로 갱신할 가중치에 학습률과 손실 함수의 기울기가 반영된 속도 변수를 더해 결정합니다.

<img src="README.assets/e 6.3.png" alt="e 6.3" style="zoom:50%;" />

<img src="README.assets/e 6.4.png" alt="e 6.4" style="zoom:50%;" />

- 즉, 공이 둥그런 그릇의 곡면을 따라 구르듯 움직임을 보여줍니다.

<img src="README.assets/fig 6-4.png" alt="fig 6-4" style="zoom:50%;" />

- 이를 python으로 구현하면 다음과 같습니다.

```python
import numpy as np


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None  # 물체의 속도

    def update(self, params, grads):
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] ++ self.v[key]


```

- 이 방식으로 앞선 문제를 해결하면 x축 진동은 덜하지만 y축 진동은 변화량이 커 y축으로 진동함을 보여줍니다.

<img src="README.assets/fig 6-5.png" alt="fig 6-5" style="zoom:50%;" />

### 6.1.5 AdaGrad

- 신경망 학습은 학습률이 매우 중요한데 이 값이 너무 작으면 변화가 없고 너무 크면 발산합니다.

- 따라서 이를 해결한 방법으로 **학습률 감소**가 있습니다.

  - 이 방식은 처음 학습률을 크게한 뒤 점차 학습률 값을 줄여가는 방법입니다.

- 가장 간단한 방법은 매개변수 전체 학습률 값을 일괄적으로 낮추는 것으로 이 방식의 발전된 방식이 **AdaGrad**입니다.

<img src="README.assets/e 6.5.png" alt="e 6.5" style="zoom:50%;" />

<img src="README.assets/e 6.6.png" alt="e 6.6" style="zoom:50%;" />

- AdaGrad는 가중치를 `h`라는 손실 함수 기울기의 원소별 곱셈을 한 일정 값의 제곱근의 역수를 반영해 점차 줄여나갑니다.

  - 이 경우 원소별로 과거 기울기를 곱하기 때문에 매개변수마다 다르게 적용되고 학습이 진행될수록 갱신 강도가 낮아집니다.

    - 다만 기울기가 0이 되어 더 이상 갱신되지 않는 걸 방지하는 방법으로 **RMSProp**이 있습니다.

    - 이 방법은 과거의 모든 기울기를 반영하지 않고 과거의 기울기 일수록 반영 정도는 낮춥니다.

```python
import numpy as np


class  AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


```

- 이 방식으로 위 문제를 해결하면 처음엔 크게 움직이다 점차 최소값이 될수록 변화량이 작아짐을 알 수 있습니다.

<img src="README.assets/fig 6-6.png" alt="fig 6-6" style="zoom:50%;" />

### 6.1.6 Adam

- **Adam**은 모멘텀과 AdaGrad의 장점을 모두 합친 기법입니다.

- Adam의 경우 하이퍼파라미터의 편향 보정도 같이 진행해주는 장점도 있습니다.

- 이 방식으로 위 문제를 해결하면 둘을 적절히 섞은 듯한 그래프가 그려집니다.

<img src="README.assets/fig 6-7.png" alt="fig 6-7" style="zoom:50%;" />

### 6.1.7 어느 갱신 방법을 이용할 것인가?

> 6.1.7_optimizer_compare.py를 참고하면 됩니다.

- 어느 방법을 사용할 지는 어느 문제를 풀어야할 지에 달려있습니다.

<img src="README.assets/fig 6-8.png" alt="fig 6-8" style="zoom:50%;" />

### 6.1.8 MNIST 데이터셋으로 본 갱신 방법 비교

> 6.1.8_optimizer_compare_with_mnist.py를 참고하면 됩니다.

- 결과 그래프는 다음과 같습니다.

<img src="README.assets/fig 6-9.png" alt="fig 6-9" style="zoom:50%;" />

## 6.2 가중치의 초기값

### 6.2.1 초기값을 0으로 하면?

- 오버피팅을 억제해 범용 성능을 높이는 방법으로 **가중치 감소**가 있습니다.

  - 이 방법은 가중치 매개변수의 값이 작아지도록 학습하는 방법으로 오버피팅이 일어나지 않도록 합니다.

  > 지금껏 모든 가중치는 `0.01 * np.random.randn(10, 100)`으로 설정한 값입니다.

- 하지만 그렇다고 가중치를 0으로 하면(정확히는 모든 가중치를 동일하게 설정하면) 오차역전파법에서 모든 가중치 값이 동일하게 갱신되므로 쓰면 안되는 방법입니다.

  - 즉, 이 방법을 막기 위해서 가중치를 무작위로 설정해야 합니다.

### 6.2.2 은닉층의 활성화값 분포

> 6.2.2_weight_init_activation_histogram.py를 참고하면 됩니다.

- 데이터가 0과 1에 치우쳐 분포하면 역전파의 기울기가 점점 작아지다 사라지는 **기울기 소실**이 발생합니다.

<img src="README.assets/fig 6-10.png" alt="fig 6-10" style="zoom:50%;" />

- 그렇다고 데이터가 어느 한쪽에 집중되는 경우 **표현력을 제한**하는 새로운 문제가 발생합니다.

<img src="README.assets/fig 6-11.png" alt="fig 6-11" style="zoom:50%;" />

- 따라서 이를 적절히 조절해야하는데 가장 권장되는 방법은 **Xavier 초깃값**입니다.

  - Xaiver는 앞 계층의 노드가 n개 인 경우 표준편차가 `1 / sqrt(n)`이면 된다는 이론입니다.

<img src="README.assets/fig 6-12.png" alt="fig 6-12" style="zoom:50%;" />

- 다만 예제를 돌리면 일그러진 종 모양이 나오는 데 이를 해결하려면 `tanh` 함수를 사용하는 편이 좋습니다.

  - 이는 활성화 함수가 원점에서 대칭이 되어야 이상적인 분포가 나오기 때문입니다.

<img src="README.assets/fig 6-13.png" alt="fig 6-13" style="zoom:50%;" />

### 6.2.3 ReLU를 사용할 때의 가중치 초기값

> 6.2.2_weight_init_activation_histogram.py를 참고하면 됩니다.

- 앞 선 Xavier는 활성화 함수가 선형일 때 유용한 초기값 입니다.

- 반면에 ReLu는 특화된 초기값이 있는데 이를 **He 초깃값**이라고 합니다.

  - He는 앞 계층의 노드가 n개일 때 표준편차가 `sqrt(2 / n)`인 정규분포를 사용합니다.

<img src="README.assets/fig 6-14.png" alt="fig 6-14" style="zoom:50%;" />

### 6.2.4 MNIST 데이터셋으로 본 가중치 초깃값 비교

> 6.2.4_weight_init_compare.py를 참고하면 됩니다.

<img src="README.assets/fig 6-15.png" alt="fig 6-15" style="zoom:50%;" />

## 6.3 배치 정규화

- 앞 장과 달리 각층이 활성화를 적당히 퍼뜨리도록 강제하는 방식인 **배치 정규화**에 대해 알아봅시다.

### 6.3.1 배치 정규화 알고리즘

- 배치 정규화는 다음 이유 때문에 주목받는 아이디어 입니다.

  1. 학습 속도 개선을 통해 빨리 진행할 수 있다.

  2. 초깃값에 크게 의존하지 않아도 된다.

  3. 오버피팅을 억제한다.

- 이를 위해 딥러닝 각 층에 배치 정규화 계층을 삽입합니다.

<img src="README.assets/fig 6-16.png" alt="fig 6-16" style="zoom:50%;" />

- 데이터 평균이 0, 분산이 1인 정규화하고 이를 활성화 함수 전이나 후에 처리합니다.

  - 수식으로 나타내면 다음과 같습니다.

    <img src="README.assets/e 6.7.png" alt="e 6.7" style="zoom:50%;" />

- 또한, 배치 정규화 계층마다 이 데이터에 고유한 확대와 이동 변환을 수행합니다.

<img src="README.assets/e 6.8.png" alt="e 6.8" style="zoom:50%;" />

- 이러한 알고리즘을 바탕으로 순전파, 역전파의 계산 그래프를 표현하면 다음과 같습니다.

<img src="README.assets/fig 6-17.png" alt="fig 6-17" style="zoom:50%;" />

### 6.3.2 배치 정규화의 효과

> 6.3.2_batch_normalization_test.py, 6.3.2_batch_normalization.py를 참고하면 됩니다.

<img src="README.assets/fig 6-18.png" alt="fig 6-18" style="zoom:50%;" />

> 배치 정규화의 한 예시입니다. 전체 예시는 다음과 같습니다.

<img src="README.assets/fig 6-19.png" alt="fig 6-19" style="zoom:50%;" />

> 여기서 가중치 초깃값의 표준편자는 각 그래프의 상단에 나타나 있습니다.

## 6.4 바른 학습을 위해

- 이 장에서는 머신러닝의 오버피팅을 억제하는 방법을 배우게 됩니다.

### 6.4.1 오버피팅

- **오버피팅**은 매개변수가 많고 표현력이 높은 경우, 훈련 데이터가 적은 경우 발생합니다.

> 오버피팅이 일어난 예제는 6.4.1_overfit_weight_decay.py를 확인하면 됩니다.

<img src="README.assets/fig 6-20.png" alt="fig 6-20" style="zoom:50%;" />

### 6.4.2 가중치 감소

- 오버피팅을 억제하기 위한 방법 중 **가중치 감소**가 있습니다.

  - 이 방법은 학습 중 가중치가 클수록 큰 페널티를 부과해 오버피팅을 억제하는 방법입니다.

  - 가장 대표적인 방식인 **L2 노름**은 각 원소의 제곱을 더한 뒤 제곱근을 구한 값입니다.

    - 그 외에도 L1 노름(절댓값의 합), Linf 노름(각 원소의 절댓값 중 가장 큰 것)도 있습니다.

> 6.4.1_overfit_wwight_decay.py를 확인하면 됩니다.

<img src="README.assets/fig 6-21.png" alt="fig 6-21" style="zoom:50%;" />

### 6.4.3 드롭아웃

- 가중치 감소 방식은 신경망이 복잡해질수록 대응하기 어려워집니다. 이를 해결하기 위해 **드롭아웃**을 사용합니다.

- 이는 훈련할 때 임의의 뉴런에 신호를 보내지 않는 방식으로 학습하고 실전에만 모든 뉴런에 신호를 전달합니다.

  - 단, 보정을 위해 실전에는 훈련 때 삭제 안 한 비율을 곱해 출력합니다.

<img src="README.assets/fig 6-22.png" alt="fig 6-22" style="zoom:50%;" />

> 6.4.3_overfit_dropout.py를 참고하면 됩니다.

<img src="README.assets/fig 6-23.png" alt="fig 6-23" style="zoom:50%;" />

#### Note

- 머신러닝은 **앙상블 학습**을 애용합니다.

  - 앙상블 학습이란 개별적으로 학습시킨 여러 모델의 출력의 평균을 가지고 추론하는 방식입니다.

- 앙상블 학습과 드롭아웃은 유사한 효과를 내게 되는데 임의의 뉴런을 삭제한 것이 임의의 모델 여러 개를 만든 것과 비슷하기 때문입니다.

## 6.5 적절한 하이퍼파라미터 값 찾기

### 6.5.1 검증 데이터

- 지금껏 훈련 데이터와 시험 데이터를 바탕으로 여러 파라미터를 평가할 수 있었습니다.

- 하지만, 하이퍼파라미터는 이와 달리 시험 데이터를 사용해서 안됩니다.

  - 그 이유는 하이퍼파라미터 값이 좋은지 시험 데이터로 확인할 경우 범용성이 떨어지기 때문입니다.

- 따라서 하이퍼파라미터를 확인하기 위해서 **검증 데이터**가 따로 필요합니다.

  - 일반적으로 검증 데이터는 시험 데이터의 20% 정도를 사용합니다.

### 6.5.2 하이퍼파라미터 최적화

- 하이퍼파라미터 최적화의 핵심은 하이퍼파라미터의 최적 값이 존재하는 범위를 줄이는 것입니다.

- 이를 위해서 일정 범위를 설정하고 샘플링을 통해 하이퍼파라미터 값을 평가합니다. 그리고 이를 반복해 그 범위를 줄여나갑니다.

- 이를 정리하면 다음과 같습니다.

  0. 하이퍼파라미터 값 범위를 설정합니다.

  1. 설정 범위 내의 임의의 값을 추출합니다.

  1. 해당 값으로 학습한 뒤 검증 데이터로 정확도를 평가합니다. 단, 에폭을 작게 설정합니다.

  1. 1 ~ 2단계를 일정 횟수(100회 등) 반복하여 정확도 결과를 보고 하이퍼파라미터의 범위를 좁힙니다.

#### Note

- 위와 같이 직관적이지 않은 방법을 원한다면 **베이즈 최적화** 기법이 유용합니다.

### 6.5.3 하이퍼파라미터 최적화 구현하기

> 6.5.3_hyperparameter_optimization.py를 참고하면 됩니다.

<img src="README.assets/fig 6-24.png" alt="fig 6-24" style="zoom:50%;" />

## 6.6 정리

- 매개변수 갱신 방법에는 SGD외에도 모멘텀, AdaGrad, Adam 등이 있습니다.

- 가중치 초깃값을 정하는 방법은 올바른 학습을 하는 데 매우 중요합니다.

- 가중치 초깃값으로는 Savier와 He가 효과적입니다.

- 배치 정규화를 이용하면 학습을 빠르게 진행할 수 있으며 초깃값에 영향을 덜 받게 됩니다.

- 오버피팅을 억제하는 정규화 기술로 가중치 감소와 드롭아웃이 있습니다.

- 하이퍼파라미터 값 탐색은 최적값이 존재할 법한 범위를 점차 좁히면서 하는 것이 효과적입니다.
