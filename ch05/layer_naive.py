class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):  # 순전파
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):  # 역전파
        dx = dout * self.y  # 역전파는 순전파를 바꾸어 전달합니다.
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
