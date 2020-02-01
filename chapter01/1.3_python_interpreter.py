# calculations
print('plus:', 1 + 2, 'minus:', 1 - 2, 'multiple:', 4 * 5, 'division:', 7 / 5)
print('share:', 7 // 5, 'rest:', 7 % 5, 'square:', 3 ** 2)

# data types
print('types:', type(10), type(1.0), type('a'))

# variables
x = 10  # initialization and substisution
print('x:', x)

x = 100  # substisution
print('x:', x)

# list
arr = [1, 2, 3, 4, 5]
print('list:', arr, 'len of list:', len(arr))

arr[4] = 99  # substitution
print('list:', arr)
print('accessing:', arr[0])
print('slicing:', arr[0:2], arr[1:], arr[:3], arr[:-1])

# dictionary
dic = {'a': 1, 'b': 2}
print('accessing:', dic['a'])

dic['c'] = 3  # initialization and substisution
print('dict:', dic)

# booleans
a, b = True, False
print('booleans:', type(a))
print('compares:', a and b, a or b, not b)

if a:  # if
    print('check: a is True.')
    if not b:
        print('check: b is False.')

for i in [1, 2, 3, 4, 5]:  # for
    print(i, end=' ')
print()


def greeting():  # function
    print('Hello, python!')


greeting()
