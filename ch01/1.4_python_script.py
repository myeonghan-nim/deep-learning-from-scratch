# study python with using script file

# 1. save files

'''
for run file with terminal(powershell, bash, etc)
enter python your_file_name.py in terminal
'''

print('I am hungry!')

# 2. class


class Man:
    def __init__(self, name):
        self.name = name
        print('Initilized!')

    def hello(self):
        print('Hello ' + self.name + '!')

    def goodbye(self):
        print('Good-bye ' + self.name + '!')


m = Man('David')
m.hello()
m.goodbye()
