class Man:  # OOP, class
    def __init__(self, name):
        self.name = name
        print('Initilized!')

    def hello(self):
        print('Hello ' + self.name + '!')

    def goodbye(self):
        print('Good-bye ' + self.name + '!')


m = Man('Reny')

m.hello()
m.goodbye()
