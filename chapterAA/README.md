# AppendixA. Softmax-with-Loss Layer`s Calculation Graph

- 이 부록에서는 소프트맥스 함수와 교차 엔트로피 오차의 계산 그래프를 그려보고 그 역전파를 구해보겠습니다.

- 소프트맥스 함수는 Softmax 계층, 교차 엔트로피 오차는 Cross Entropy Error 계층, 이 둘을 조합한 것을 Softmax-with-Loss 계층이라 부릅니다.

> 우선 결과를 미리 보면 다음과 같습니다.

<img src="README.assets/fig a-1.png" alt="fig a-1" style="zoom:50%;" />

- 위 예시는 세 개의 클래스 분류를 수행하는 신경망을 가정하고 있습니다.

  - 이전 계층에서 입력은 (a<sub>1</sub>, a<sub>2</sub>, a<sub>3</sub>)이며 Softmax 계층은 (y<sub>1</sub>, y<sub>2</sub>, y<sub>3</sub>)을 출력합니다.

  - 또한 정답 레이블은 (t<sub>1</sub>, t<sub>2</sub>, t<sub>3</sub>)이며 Cross Enttopy Error 계층은 손실 L을 출력합니다.

  - 이 부록에서 Softmax-with-Loss 계층의 역전파 결과는 (y<sub>1</sub> - t<sub>1</sub>, y<sub>2</sub> - t<sub>2</sub>, y<sub>3</sub> - t<sub>3</sub>)가 될 것입니다.

## A.1 순전파

- 우선 위와 같은 게산 그래프에서는 Softmax 계층과 Cross Entropy Error 계층의 내용은 생략했습니다.

- 따라서 이번 절에서는 이 내용을 생략하지 않고 그리는 것부터 시작하겠습니다.

- 우선 Softmax 계층의 소프트맥스 함수의 수식은 다음과 같습니다.

<img src="README.assets/e a.1.png" alt="e a.1" style="zoom:50%;" />

- 그리고 이에 따른 계산 그래프는 다음과 같습니다.

<img src="README.assets/fig a-2.png" alt="fig a-2" style="zoom:50%;" />

- 위 그림에서 지수의 합, 즉 위 식의 분모 항을 S로 표기했습니다. 또한, 최종 출력은 (y<sub>1</sub>, y<sub>2</sub>, y<sub>3</sub>)입니다.

- 이어서 Cross Entropy Error 계층의 교차 엔트로피 수식은 다음과 같습니다.

<img src="README.assets/e a.2.png" alt="e a.2" style="zoom:50%;" />

- 이를 바탕으로 Cross Entropy Error 계층의 계산 그래프는 다음과 같습니다.

<img src="README.assets/fig a-3.png" alt="fig a-3" style="zoom:50%;" />

- 이처럼 각 계산 그래프는 식을 그대로 계산 그래프로 그린 것이라 크게 어려운 점이 없을 것입니다.

> 이어서 역전파를 살펴보겠습니다.

## A.2 역전파

- 우선 Cross Entropy Error 계층의 역전파는 다음과 같습니다.

<img src="README.assets/fig a-4.png" alt="fig a-4" style="zoom:50%;" />

- 이 계산 그래프의 역전파를 구할 때는 다음을 유념해야 합니다.

  - 역전파의 초깃값, 즉 위 그림의 가장 오른쪽 역전파 값은 1입니다.(자기 자신에 대해 편미분하면 1입니다.)

  - '\*' 노드의 역전파는 순전파의 입력값을 서로 바꿔 상류의 미분에 곱해 하류로 흘립니다.

  - '+' 노드의 역전파는 상류의 미분을 그대로 흘립니다.

  - log 노드의 역전파는 다음 식을 따릅니다.

    - y = log(x) → dy/dx = 1/x

- 위 규칙을 따르면 Cross Entropy Error 계층의 역전파는 쉽게 구할 수 있습니다.

  - 결과는 (-t<sub>1</sub>/y<sub>1</sub>, -t<sub>2</sub>/y<sub>2</sub>, -t<sub>3</sub>/y<sub>3</sub>)로 이 값이 Softmax 계층의 역전파 입력이 됩니다.

- 이어서 Softmax 계층의 역전파입니다. 이 계층의 역전파는 조금 복잡해 하나씩 확인하며 진행할 것입니다.

  1. 앞 계층(Cross Entropy Error)의 역전파 값이 흘러옵니다.

  <img src="README.assets/fig a-4(1).png" alt="fig a-4(1).png" style="zoom:50%;" />

  2. '\*' 노드에서 순전파의 입력들을 서로 바꿔 곱합니다. 여기서 이루어지는 계산은 다음과 같습니다.

  <img src="README.assets/e a.3.png" alt="e a.3" style="zoom:50%;" />

  <img src="README.assets/fig a-4(2).png" alt="fig a-4(2).png" style="zoom:50%;" />

  3. 순전파에서 여러 갈래로 나뉘어 흘렸던 것과 반대로 역전파에서는 흘러온 값들을 더합니다.

     - 여기에서는 3개의 갈라진 역전파 값 (-t<sub>1</sub>S, -t<sub>2</sub>S, -t<sub>3</sub>S)가 더해집니다.

     - 이 값이 '/' 노드의 역전파를 거쳐 (t<sub>1</sub> + t<sub>2</sub> + t<sub>3</sub>)/S가 됩니다.

     > '/' 노드의 역전파는 상류에서 흘러온 값에 순전파 때 출력을 제곱한 후 마이너스를 붙인 값을 곱해 하류로 전달합니다.

     - 그런데 여기에서 (t<sub>1</sub>, t<sub>2</sub>, t<sub>3</sub>)은 원-핫 벡터로 표현된 정답 레이블입니다.

     > 원-핫 벡터란 (t<sub>1</sub>, t<sub>2</sub>, t<sub>3</sub>) 중 단 하나만 1이고 나머지는 0임을 의미합니다.

     - 따라서 t<sub>1</sub> + t<sub>2</sub> + t<sub>3</sub> = 1이 됩니다.

  <img src="README.assets/fig a-4(3).png" alt="fig a-4(3).png" style="zoom:50%;" />

  4. '+' 노드는 입력을 여과 없이 내보냅니다.

  <img src="README.assets/fig a-4(4).png" alt="fig a-4(4).png" style="zoom:50%;" />

  5. '\*' 노드는 입력을 서로 바꾼 곱셈이므로 여기에서는 y<sub>1</sub> = exp(a<sub>1</sub>)/S를 이용해 식을 변형했습니다.

  <img src="README.assets/fig a-4(5).png" alt="fig a-4(5).png" style="zoom:50%;" />

  6. exp 노드는 다음 식을 통해 계산합니다.

  <img src="README.assets/e a.4.png" alt="e a.4" style="zoom:50%;" />

  <img src="README.assets/fig a-4(6).png" alt="fig a-4(6).png" style="zoom:50%;" />

  - 그리고 두 갈래의 입력의 합에 exp(a<sub>1</sub>)를 곱한 수치가 바로 여기에서 구하는 역전파입니다.

  > 식으로 표현하면 (1/S - t<sub>1</sub>/exp(a<sub>1</sub>))exp(a<sub>1</sub>)이고 이를 변형하면 y<sub>1</sub> - t<sub>1</sub>이 됩니다.

- 이상으로 순전파의 입력이 a<sub>1</sub>인 노드에 대해 역전파가 y<sub>1</sub> - t<sub>1</sub>임이 유도되었습니다.

- 나머지 a<sub>2</sub>, a<sub>3</sub>의 역전파도 동일하게 각각 y<sub>2</sub> - t<sub>2</sub>, y<sub>3</sub> - t<sub>3</sub>입니다.

- 또한, n개의 클래스 분류에서도 같은 결과가 유도됨을 쉽게 알 수 있습니다.

## A.3 정리

- 여기에서는 Softmax-with-Loss 계층의 계산 그래프를 생략 없이 그려가며 그 역전파를 구했습니다.

> Softmax-with-Loss 계층의 계산 그래프를 생략하지 않고 그리면 다음과 같습니다.

<img src="README.assets/fig a-5.png" alt="fig a-5" style="zoom:50%;" />

- 복잡해 보이지만, 계산 그래프로 한 단계씩 확인하는 것은 그렇게 힘든 작업이 아닙니다.

> Softmax-with-Loss 계층 외에도 배치 정규화 계층 등 복잡해 보이는 계층을 이와 같이 작업하면 분명 쉽게 이해할 수 있습니다.
