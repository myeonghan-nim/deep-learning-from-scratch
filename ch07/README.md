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
