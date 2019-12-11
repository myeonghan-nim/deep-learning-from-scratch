# study basic python language

# 1. basic calculations

print(1 + 2)  # plus
print(1 - 2)  # minus
print(4 * 5)  # multiple
print(7 / 5)  # divide

print(7 // 5)  # share of divide
print(7 % 5)  # rest of divide

print(3 ** 2)  # power

# 2. data types

print(type(10))  # int
print(type(2.1))  # float
print(type('a'))  # str

# 3. variables

x = 10  # initialization
print(x)

x = 100  # substisution
print(x)

# 4. list

arr = [1, 2, 3, 4, 5]
print(arr)

print(len(arr))  # length of list
print(arr[0])  # index accessing

arr[4] = 99  # substitution
print(arr)

print(arr[0:2])  # slicing
print(arr[1:])
print(arr[:3])
print(arr[:-1])

# 5. dictionary

dic = {'a': 1, 'b': 2}
print(dic['a'])  # accessing atom

dic['c'] = 3  # substitution
print(dic)

# 6. boolean

a = True
b = False

print(type(a))

print(a and b)
print(a or b)
print(not b)

# 7. if

if a:
    print('a is True.')

if b:
    print('Is it right?')
else:
    print('b is False.')

# 8. for

for i in [1, 2, 3, 4, 5]:
    print(i)

# 9. function


def greeting():
    print('Hello, python!')


greeting()
