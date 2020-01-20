# chapter07. CNN

- **합성곱 신경망(CNN)**은 주로 이미지 인식 분야에 유용하게 쓰이는 딥러닝 기법입니다.

## 7.1 전체 구조

- CNN은 **합성곱 계층**과 **풀링 계층**을 조합해 만듭니다만 기존의 방식과 약간 차이가 있습니다.

- 우선 기존의 신경망은 입전하는 계층의 모든 뉴런과 결합된 **완전연결** 형태를 가졌습니다.

  - 이 계층을 **Affine 계층**이라고 불렀습니다.

<img src="README.assets/fig 7-1-1578400851611.png" alt="fig 7-1" style="zoom:50%;" />

> 즉, 완전연결 신경망은 Affine-ReLU 조합의 결과를 Softmax를 통해 출력합니다.

- 반면 CNN은 합성곱 계층과 풀링 계층이 새로 추가된 형태를 띄고 있습니다.

<img src="README.assets/fig 7-2.png" alt="fig 7-2" style="zoom:50%;" />

> 즉, CNN은 Conv(합성곱)-ReLU-(Pooling)(풀링)의 흐름으로 연결됩니다.

- 또한 CNN은 출력에 가까운 층에서 지금까지 사용한 Affine-ReLU 구성을 사용할 수 있고 출력 계층도 Affine-Softmax를 그대로 사용할 수 있습니다.

## 7.2 합성곱 계층

- CNN에서는 **패딩**, **스트라이드**와 같은 CNN고유의 용어가 등장합니다.

- 또한, 각 계측 사이에서 3차원 데이터 같은 입체적인 데이터가 흐른다는 점에서 완전연결 신경망과 다릅니다.

### 7.2.1 완전연결 계층의 문제점

- 지금까지 사용한 환전연결 신경망은 완전연결 계층(Affine 계층)을 사용했습니다.

  - 완전연결 계층은 인접하는 계층의 모든 뉴런이 연결되고 출력 수를 임의로 조절할 수 있습니다.

- 다만, 이 형태는 데이터의 형상이 무시된다는 단점을 지니고 있습니다.

  - 예를 들어 이미지의 경우 가로, 세로, 채널의 3차원 데이터지만 완전연결 계층에 입력할 때 1차원 데이터로 평탄화를 해야합니다.

  - 이는 이미지가 가지고 있는 공간적 정보를 무시하며 본질적인 패턴이 있을 수 있는 경우를 무시합니다.

- 반면에 합성곱 계층은 형상을 유지합니다. 따라서 입력된 데이터의 형상과 출력된 데이터의 형상을 유지합니다.

- CNN에서는 합성곱 계층의 입출력 데이터를 **특징 맵**이라고 합니다.(입력은 입력 특징 맵, 출력은 출력 특징 맵)

### 7.2.2 합성곱 연산

- 합성곱 계층에서 처리하는 **합성곱 연산**은 이미지 처리에서 말하는 **필터 연산**입니다.

<img src="README.assets/fig 7-3.png" alt="fig 7-3" style="zoom:50%;" />

- 위와 같이 합성곱 연산은 입력 데이터에 **필터**(혹은 **커널**)를 적용합니다.

  - 합성곱 연산은 필터의 **윈도우**를 일정 간격으로 이동해가며 입력 데이터에 적용합니다.

  - 이 이동의 결과에 따라 입력과 필터에 대응하는 원소의 곱의 합을 출력하는데 이를 **단일 곱셈 누산**이라고 합니다.

  - 이 과정을 나열하면 다음과 같습니다.

<img src="README.assets/fig 7-4.png" alt="fig 7-4" style="zoom:50%;" />

> 완전연결 신경망의 가중치와 편항 중 CNN의 필터의 매개변수가 가중치에 해당합니다.
>
> 여기에 편향이 적용되면 다음과 같습니다.

<img src="README.assets/fig 7-5.png" alt="fig 7-5" style="zoom:50%;" />

### 7.2.3 패딩

- **패딩**은 합성곱 연산을 수행하기 전에 입력 데이터 주변을 특정 값(예를 들어 0)으로 채우는 작업을 말합니다.(다음 그림을 참고하세요.)

<img src="README.assets/fig 7-6.png" alt="fig 7-6" style="zoom:50%;" />

- 이렇게 하면 입력 데이터와 동일한 형상의 출력데이터를 얻을 수 있습니다.

- 패딩의 크기는 원하는 대로 설정할 수 있으며(위 그림같은 경우는 1입니다.) 출력 데이터의 크기를 조절합니다.

#### Note

- 패딩을 주로 출력 크기를 조정할 목적으로 사용됩니다.

- 특히 합성곱 연산을 반복하는 심층 신경망에서 데이터의 크기가 작아지는 것을 방지하는 차원에서 주로 사용합니다.

### 7.2.4 스트라이드

- 필터를 적용하는 위치 간격을 **스트라이드**라고 합니다.

- 앞선 예시는 스트라이드가 1이었지만 다음과 같은 예시는 스트라이드가 2입니다.

<img src="README.assets/fig 7-7.png" alt="fig 7-7" style="zoom:50%;" />

- 위와 같이 연산을 지속하다보면 패팅을 크게하면 출력의 크기는 커지고 스트라이드를 키우면 출력의 크기는 작아집니다.

  - 이들의 상관 관계는 다음의 수식과 같습니다.

  <img src="README.assets/e 7.1.png" alt="e 7.1" style="zoom:50%;" />

  > H, W: 입력의 크기(세로, 가로), FH, FW: 필터의 크기, OH, OW: 출력의 크기, P: 패딩, S, 스트라이드

  - 다만, 출력의 크기가 정수로 나누어 떨어지게 끔 올바른 파라미터 값을 조절해야 합니다.

### 7.2.5 3차원 데이터의 합성곱 연산

- 다음과 같이 3차원 데이터는 3차원 방향으로 특징 맵이 늘어났습니다.

  - 그 결과 입력 데이터와 합성곱 연산을 채널마다 수행하고 그 결과를 더해 하나의 출력을 만듭니다.

<img src="README.assets/fig 7-8.png" alt="fig 7-8" style="zoom:50%;" />

> 조금 더 상세한 연산 과정은 다음과 같습니다.

<img src="README.assets/fig 7-9.png" alt="fig 7-9" style="zoom:50%;" />

- 이 때 입력 데이터의 채널 수와 필터의 채널 수가 같아야 함에 유의하세요.

### 7.2.6 블록으로 생각하기

- 3차원의 합성곱 연산은 데이터와 필터를 직육면체 블록으로 생각하면 더 쉽습니다.

> 즉, C, H, W 형태의 입력 데이터와 C, FH, FW 형태의 필터가 만나 1, OH, OW 형태의 출력 데이터를 얻습니다.

- 만약 연산의 출력으로 다수의 채널을 보내고 싶다면 필터를 다수 이용하는 다음과 같은 과정을 거치면 됩니다.

<img src="README.assets/fig 7-10.png" alt="fig 7-10" style="zoom:50%;" />

> 위와 같이 필터를 FN개 적용하면 출력도 FN개가 됩니다.

- 즉, 합성곱 연산은 필터의 수도 고려해야 합니다.

  - 그래서 필터의 가중치 데이터는 4차원 데이터(FN, C, FH, FW: 출력 채널 수, 입력 채널 수, 높이, 너비)입니다.

- 여기에 편향을 더하면 다음과 같습니다.

<img src="README.assets/fig 7-12.png" alt="fig 7-12" style="zoom:50%;" />

### 7.2.7 배치 처리

- 신경망에서 입력 데이터를 한 덩어리로 묶어 배치로 처리한 것 처럼 합성곱 연산도 배치 처리를 지원합니다.

  - 구체적으로는 각 계층을 흐르는 데이터의 차원을 하나 늘려 4차원 데이터(데이터 수, 채널 수, 높이, 너비)로 합니다.

<img src="README.assets/fig 7-13.png" alt="fig 7-13" style="zoom:50%;" />

> 즉, 위와 같이 4차원 데이터가 하나 흐르면 데이터 N개에 대한 합성곱 연산을 수행합니다.

## 7.3 풀링 계층

- 풀링이란 가로, 세로 방향의 공간을 줄이는 연산을 말합니다.

<img src="README.assets/fig 7-14.png" alt="fig 7-14" style="zoom:50%;" />

- 위는 2 by 2 **최대 풀링**을 스트라이드 2로 처리하는 순서 예시입니다.

  - 최대 풀링은 최댓값을 구하는 연산으로 2 by 2의 크기에서 가장 큰 원소를 꺼냅니다.

  - 여기에 스트라이드가 2이므로 윈도우가 2칸 간격으로 이동하게 됩니다.

  > 일반적으로 풀링의 크기와 스트라이드의 크기는 일치합니다.

> 이 외에도 **평균 풀링**(대상 영역의 평균을 계산) 등 다양한 풀링이 있습니다.

### 7.3.1 풀링 계층의 특징

- 풀링 계층의 특징은 다음과 같습니다.

  1. 학습해야 할 매개변수가 없습니다.

     - 풀링 게층은 합성곱 계층과 달리 학습해야 할 매개변수가 없습니다.

  2. 채널 수가 변하지 않습니다.

     - 풀링 연산은 입력 데이터의 채널 수 그대로 출력 데이터로 내보내는데 이는 채널마다 독립적으로 계산하기 때문입니다.

     <img src="README.assets/fig 7-15.png" alt="fig 7-15" style="zoom:50%;" />

  3. 입력의 변화에 영향을 적게 받습니다.

     - 입력 데이터가 조금 변하더라도 풀링의 결과는 잘 변하지 않습니다.

     <img src="README.assets/fig 7-16.png" alt="fig 7-16" style="zoom:50%;" />

## 7.4 합성곱/풀링 계층 구현하기

- 이 두 계층을 오차역전파법에서 배운 것처럼 `forward`와 `backward` 메서드를 추가해 구현해보세요.

### 7.4.1 4차원 배열

- CNN에 흐르는 데이터는 4차원 입니다. 이는 python으로 다음과 같이 구현할 수 있습니다.

```python
import numpy as np

x = np.random.rand(10, 1, 28, 28)  # 4차원 무작위 데이터를 생성합니다.
print(x.shape)
```

- 이 중 특정 인덱스의 값에 접근하려면 다음과 같습니다.

```python
# 위에 이어집니다.
print(x[0].shape)  # (1, 28, 28)
print(x[5].shape)
```

- 또, 특정 인덱스의 데이터 특정 채널의 공간 데이터에 접근하려면 다음과 같습니다.

```python
# 위에 이어집니다.
print(x[0, 0])  # 혹은 x[0][0]
```

- 하지만 위와 같이 복잡한 구현을 쉽게 해주는 **im2col**을 사용하면 쉽게 해결할 수 있습니다.

### 7.4.2 im2col로 데이터 전개하기

- 합성곱 연산을 곧이곧대로 구현하려면 for문을 여러 번 돌아야하는 문제가 발생합니다.

  - 따라서 이를 해결하기 위해 **im2col**이라는 함수를 사용해 구현해보겠습니다.

- im2col은 데이터를 필터링(가중치 계산)하기 좋게 전개하는 함수입니다. 다음 그림처럼 4차원을 2차원으로 변경합니다.

<img src="README.assets/fig 7-17.png" alt="fig 7-17" style="zoom:50%;" />

- 조금 더 구체적으로 입력 데이터 중 필터를 적용하는 영역인 3차원 블록을 한 줄로 늘어놓고 여기에 필터를 사용하는 게 im2col 방식입니다.

<img src="README.assets/fig 7-18.png" alt="fig 7-18" style="zoom:50%;" />

- 이처럼 보기 좋게 스트라이드를 크게 잡아 필터 적용 영역이 겹치지 않도록 했지만, 실제로 영역이 겹치는 경우가 대다수입니다.

- 이 필터 적용 영역이 겹치게 되면 im2col로 전개한 후 원소 수가 원래 블록의 원소 수보다 많아집니다.

  - 그래서 im2col 방식은 메모리 소비량이 많아지지만 컴퓨터에 이를 계산하는 데 탁월한 능력을 가지고 있습니다.

  - 예를 들어 행렬 계산 라이브러리 등은 고도로 최적화된 큰 행렬의 곱셈을 빠르게 계산할 수 있습니다.

- im2col로 데이터를 전개한 다음에 합성곱 계층의 필터를 1열로 전개하고 두 행렬의 곱을 진행하면 됩니다.

> 이는 완전연결 계층의 Affine 게층에서 한 것과 거의 일치합니다.

<img src="README.assets/fig 7-19.png" alt="fig 7-19" style="zoom:50%;" />

- 마지막으로 출력 데이터가 2차원이므로 이를 4차원으로 변형하면 완성입니다.

### 7.4.3 합성곱 계층 구현하기

> 미리 구현된 함수는 ch07/commons/util.py를 참고하세요.

- 우선 im2col 함수의 인터페이스는 다음과 같습니다.

```python
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    '''
    input_data: 데이터 수, 채널 수, 높이, 너비로 구성된 4차원 배열 입력 데이터
    filter_h: 필터의 높이
    filter_w: 필터의 너비
    stride: 스트라이드
    pad: 패딩
    '''
    ...
```

- im2col은 주어진 인자를 고려해 입력 데이터를 2차원 배열로 전개합니다. 이를 사용한 예제는 다음과 같습니다.

```python
from commons.util import im2col
import numpy as np

x1 = np.random.rand(1, 3, 7, 7)  # 입력 데이터(데이터 수, 채널 수, 높이, 너비)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)  # (9, 75)

x2 = np.random.rand(10, 3, 7, 7)  # 입력 데이터 10개
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)  # (90, 75)
```

> 이처럼 배치 크기가 증가하면 출력 데이터의 크기도 증가합니다.

- 이를 이용해 합성곱 계층을 구현하면 다음과 같습니다.

```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T  # 필터를 전개합니다.
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
```

- 위의 코드를 해석하자면 다음과 같습니다.

  0. 필터(가중치), 편향, 스트라이드, 패딩을 인수로 받아 계층을 초기화 합니다.

     - 필터는 (FN, C, FH, FW)의 4차원 형상으로 각각 필터 개수, 채널, 필터 높이, 필터 너비를 의미합니다.

  1. `col`을 정의하는 부분의 세 줄을 통해 im2col로 전개한 데이터와 필터의 행렬의 곱을 계산합니다.

     - 여기서 `col_W`의 `reshape(FN, -1)`에서 -1은 reshape의 편의 기능으로 다차원 배열의 원소 수가 변환 후에도 동일하게 유지해줍니다.

     - 예를 들어 (10, 3, 5, 5)의 750개 원소를 (10, -1)을 통하면 10개 묶음으로 (10, 75)로 만들어줍니다.

  1. 마지막으로 출력 데이터를 적절한 형상으로 바꿔줍니다.

     - 여기서 `transpose` 함수는 다차원 배열의 축 순서를 다음과 같이 바꿔줍니다.

     <img src="README.assets/fig 7-20.png" alt="fig 7-20" style="zoom:50%;" />

> 이상으로 완전연결 계층의 Affine 계층과 거의 일치하게 구현할 수 이었습니다.

- 한편, 합성곱 계층의 역전파는 Affine 계층의 구현과 공통점이 많습니다.

  - 다만 주의할 사항으로 역전파에서는 `col2im` 함수를 사용해야 합니다.

  > 더 자세한 내용은 ch07/commons/layer.py를 확인하세요.

### 7.4.4 풀링 게층 구현하기

- 풀링 계층로 im2col을 통해 입력 데이터를 전개합니다.

  - 다만, 채널이 독립적이라는 점이 합성곱 계층의 구현과 다른 점입니다.

  <img src="README.assets/fig 7-21.png" alt="fig 7-21" style="zoom:50%;" />

  - 이 전개에서 행별 최댓값을 구하고 적절한 형상으로 성형하면 다음과 같습니다.

  <img src="README.assets/fig 7-22.png" alt="fig 7-22" style="zoom:50%;" />

- 이를 python으로 구현하면 다음과 같습니다.

```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape

        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 전개
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 최대값
        out = np.max(col, axis=1)

        # reshape
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out
```

- 풀링 계층은 다음의 단계를 거칩니다.

  1. 입력 데이터를 전개합니다.

  2. 각 행별 최댓값을 구합니다

  3. 적절한 모양으로 성형합니다.

- 이상이 풀링 계층의 forward 처리이며 반면에 backward는 `ReLU` 게층에서 사용한 max의 역전파를 참고하면 됩니다.

> 더 자세한 구현은 ch07/commons/layer.py를 확인하세요.

## 7.5 CNN 구현하기

- 본격적으로 CNN을 구현해보면 다음과 같은 형태를 구현하게 됩니다.

<img src="README.assets/fig 7-23.png" alt="fig 7-23" style="zoom:50%;" />

- 위의 CNN을 구성할 `SimpleConvNet`은 다음과 같이 인자를 받고 시작합니다.

| 인자            | 설명                                    |
| :-------------- | :-------------------------------------- |
| input_dim       | 입력 데이터(채널 수, 높이, 너비)의 차원 |
| conv_param      | 합성곱 계층의 하이퍼파라미터 dict       |
| - filter_num    | 필터의 수                               |
| - filter_size   | 필터의 크기                             |
| - stride        | 스트라이드                              |
| - pad           | 패딩                                    |
| hidden_size     | 은닉층(완전연결)의 뉴런 수              |
| output_size     | 출력층(완전연결)의 뉴런 수              |
| weight_init_std | 초기화 할 때 가중치 표준편차            |

- 이 `SimpleConvNet`을 3단계로 나눠서 설명하면 다음과 같습니다.

  1. part01

  ```python
  class SimpleConvNet:
      def __init__(self, input_dim=(1, 28, 28),
                   conv_param={'filter_num': 30, 'filter_size': 5,
                               'stride': 1, 'pad': 0}
                   hidden_size=100, output_size=10, weight_init_std=0.01):
          filter_num = conv_param['filter_num']
          filter_size = conv_param['filter_size']
          filter_stride = conv_param['stride']
          filter_pad = conv_param['pad']

          input_size = input_dim[1]

          conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
          pool_output_size = int(filter_num * ((conv_output_size / 2) ** 2))
  ```

  - 여기서 초기화 인수로 주어진 합성곱 계층의 하이퍼파라미터를 짇셔너리에서 꺼냅니다.

  - 그리고 합성곱 계층의 출력 크기를 계산합니다.

  2. part02

  ```python
      self.params = {}
      self.params['W1'] = weight_init_std * np.random.rand(filter_num, input_dim[0],
                                                           filter_size, filter_size)
      self.params['b1'] = np.zeros(filter_num)
      self.params['W2'] = weight_init_std * np.random.rand(pool_output_size, hidden_size)
      self.params['b3'] = np.zeros(hidden_size)
      self.params['W3'] = weight_init_std * np.random.rand(hidden_size, output_size)
      self.params['b3'] = np.zeros(output_size)
  ```

  - 학습에 필요한 매개변수인 합성곱 계층과 나머지 두 완전연결 계층의 가중치와 편향을 params에 저장합니다.

  - 그리고 CNN을 구성하는 계층들을 생성합니다.

  3. part03

  ```python
      self.layers = OrderedDict()
      self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                         self.params['stride'], self.params['pad'])
      self.layers['Relu1'] = Relu()
      self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
      self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
      self.layers['Relu2'] = Relu()
      self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

      self.last_layer = SoftmaxWithLoss()
  ```

  - `OrderedDict`를 사용해 계층을 순서대로 추가하고 마지막 계층인 `SoftmaxWithLoss`는 별도의 변수에 저장합니다.

  - 이렇게 초기화를 마치고 이를 추론하는 `predict`와 손실 함수를 계산하는 `loss` 메서드를 구현합니다.

  4. part04

  ```python
      def predict(self, x):
          for layer in self.layers.values():
              x = layer.forward(x)
          return x

      def loss(self, x, t):
          y = self.predict(x)
          return self.last_layer.forward(y, t)
  ```

  - part04에서 x는 입력 데이터, t는 정답 레이블입니다.

  - predict는 layers의 계층을 앞에서 부터 차례로 호출하며 그 결과를 다음 게층에 전달합니다.

  - 반면, loss는 predict의 결과를 인자로 마지막 층의 forward를 계산합니다.

  - 이어 오차역전파법으로 기울기를 구하는 방법은 다음과 같습니다.

  ```python
      def gradient(self, x, t):
          # forward
          self.loss(x, t)

          # backward
          dout = 1
          dout = self.last_layer.backward(dout)

          layers = list(self.layers.values())
          layers.reverse()
          for layer in layers:
              dout = layer.backward(dout)

          # save result
          grads = {}
          grads['W1'] = self.layers['Conv1'].dW
          grads['b1'] = self.layers['Conv2'].dW
          grads['W2'] = self.layers['Affine1'].dW
          grads['b2'] = self.layers['Affine1'].dW
          grads['W3'] = self.layers['Affine2'].dW
          grads['b3'] = self.layers['Affine2'].dW

          return grads
  ```

  - 기울기를 오차역전파법으로 구하는 과정에서 순전파와 역전파를 반복하여 이를 dict에 담아 저장합니다.

- 이를 기반으로 MNIST 데이터셋을 학습해보는 시간입니다.

> ch07/7.5_simulate_CNN.py를 확인하세요.

- 결과가 매우 정확하게 나온 것을 보면 비교적 작은 네트워크로서 아주 높은 결과를 얻은 것입니다.

  - 이처럼 합성곱 계층과 풀링 계층은 이미지 인식에 필수적인 모듈로 CNN을 통해 더 높은 정확도를 얻을 수 있습니다.

## 7.6 CNN 시각화하기

### 7.6.1 1번째 층의 가중치 시각화하기

- 앞서서 MNIST로 간단한 CNN 학습을 했을 때 1번째 층의 합성곱 계층 가중치는 (30, 1, 5, 5) 형상을 가지고 있었습니다.

  - 채널이 1개라는 건 이 필터를 1채널 회색조 이미지로 시각화 할 수 있다는 것입니다.

- 이를 시각화한 이미지는 다음과 같습니다.

> 추가적으로 ch07/7.6.1_visualize_filter.py를 같이 참고해주세요.

<img src="README.assets/fig 7-24.png" alt="fig 7-24" style="zoom:50%;" />

- 이처럼 학습 전 필터는 흑백의 정도에 규칙성이 없는 것을 확인할 수 있습니다.

- 반면에 학습 후 필터는 흑백의 정도에 규칙성을 가지고 있는 이미지가 되었습니다.

- 이렇게 규칙성 있는 필터는 **에지**(색상이 바뀐 경계선)와 **블롭**(국소적으로 덩어리진 영역) 등을 보고 있습니다.

  - 자세한 내용은 다음 이미지를 통해 설명할 수 있습니다.

<img src="README.assets/fig 7-25.png" alt="fig 7-25" style="zoom:50%;" />

- 이처럼 필터 1은 세로 에지에, 2는 가로 에지에 반응하여 에지나 블롭 등 원시적인 정보를 추출할 수 있습니다.

  - 이 정보는 후단 계층에 전달되는 과정이 앞서 구현한 CNN입니다.

### 7.6.2 층 깊이에 따른 추출 정보 변화

- 앞 절의 결과는 1번째 층의 합성곱 계층을 대상으로 한 것입니다.

- 이를 확장하면 겹겹이 쌓인 CNN 계층에서는 계층이 깊어질수록 추출되는 정보, 즉 강하게 반응하는 뉴런이 더 추상화됨을 알 수 있습니다.

> 다음 그림을 확인하세요.

<img src="README.assets/fig 7-26.png" alt="fig 7-26" style="zoom:50%;" />

- 위 그림은 AlexNet 8층 구조로 일반 사물을 인식한 결과입니다.

  - 해당 CNN은 합성곱 계층과 풀링 계층을 여러 겹 쌓고 마지막으로 완전연결 계층을 거쳐 결과를 출력합니다.

  - 그림 중간중간 존재하는 블록은 중간 데이터이며 이 중간 데이터에 합성곱 연산을 반복합니다.

- 딥러닝의 흥미로운 점은 합성곱 계층을 쌓을수록 층이 깊어지며 더 복잡하고 추상화된 정보가 추출된다는 것입니다.

- 이는 처음에 단순한 에지에 반응하던 것이 텍스처, 복잡한 사물에 반응하며 점점 변화함을 의미합니다.

- 즉, 층이 깊어지면서 뉴런이 반응하는 대상이 단순한 모양에서 **고급 정보**로 변화합니다.

  - 다시 말하면 사물의 **의미**를 이해하도록 변화합니다.

## 7.7 대표적인 CNN

- 지금까지 제안된 CNN 다양한 네트워크 구성 중 특히 중요한 네트워크를 소개합니다.

  - 그 중 하나는 CNN의 원조인 **LeNet**이고 다른 하나는 딥러닝이 주목받게 만든 **AlexNet**입니다.

### 7.7.1 LeNet

- LeNet은 손글씨 숫자를 인식하는 네트워크로 1998년에 제안되었습니다.

- 이 네트워크는 합성곱 계층과 풀링 계층(정확히는 단순히 원소를 줄이기만 하는 서브샘플링 계층)을 반복하고 완전연결 계층을 거쳐 결과를 출력하는 다음과 같습니다.

<img src="README.assets/fig 7-27.png" alt="fig 7-27" style="zoom:50%;" />

- LeNet과 현재의 CNN은 몇몇 차이를 지니고 있습니다.

  1. 활성화 함수: LeNet은 시그모이드를, 현재는 주로 ReLU를 활성화 함수로 사용합니다.

  2. 풀링 계층: LeNet은 서브샘플링을 통해 중간 데이터의 크기를 줄이지만 현재는 최대 풀링을 주로 사용합니다.

### 7.7.2 AlexNet

- AlexNet은 딥러닝 열풍을 일으킨 주역으로 그 과정은 LeNet과 크게 다르지 않습니다.

<img src="README.assets/fig 7-28.png" alt="fig 7-28" style="zoom:50%;" />

- 이 네트워크는 합성곱 계층과 풀링 계층을 거듭하며 마지막으로 완전연결 계층을 거쳐 결과를 출력합니다.

- LeNet과 비교해서 AlexNet은 다음의 변화를 가지고 있습니다.

  1. 활성화 함수로 ReLU를 사용합니다.

  2. LRN이라는 국소적 정규화를 실시하는 계층이 들어갑니다.

  3. 드롭아웃을 사용합니다.

#### Note

- 딥러닝은 수많은 매개변수를 사용합니다. 따라서 학습을 위해서는 엄청냔 양을 계산해야 합니다.

- 이 과정을 **피팅**이라고 하는데 이 피팅에 필요한 데이터도 다량으로 필요합니다.

- 그를 위해서 등장한 GPU 사용과 빅데이터는 이런 문제의 해결책이 될 수 있습니다.

## 7.8 정리

- CNN은 지금까지 배운 완전연결 계층 네트워크에 합성곱 계층과 풀링 계층을 새로 추가합니다.

- 합성곱 계층과 풀링 계층은 im2col(이미지를 행렬로 전개하는 함수)을 이용하면 간단하고 효율적으로 구현할 수 있습니다.

- CNN을 시각화하면 계층이 깊어질수록 고급 정보가 추출되는 모습을 볼 수 있습니다.

- 대표적인 CNN은 LeNet과 AlexNet이 있습니다.

- 딥러닝의 발전에는 GPU와 빅데이터가 크게 기여했습니다.
