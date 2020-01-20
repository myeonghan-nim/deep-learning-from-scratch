# 1. calculations in python
print(1 + 2)
print(1 - 2)
print(4 * 5)
print(7 / 5)

print(7 // 5)
print(7 % 5)

print(3 ** 2)


# 2. data types in python
print(type(10))
print(type(2.1))
print(type('a'))


# 3. variables
x = 10  # initialization
print(x)

x = 100  # substisution
print(x)


# 4. array in python, list
arr = [1, 2, 3, 4, 5]
print(arr)
print(len(arr))  # length of list

arr[4] = 99  # substitution of list
print(arr)

print(arr[0])  # index accessing
print(arr[0:2])  # slicing
print(arr[1:])
print(arr[:3])
print(arr[:-1])


# 5. dictionary in python
dic = {'a': 1, 'b': 2}
print(dic['a'])  # accessing

dic['c'] = 3  # add data in dict
print(dic)


# 6. boolean types in python
a, b = True, False

print(type(a))

print(a and b)
print(a or b)
print(not b)


# 7. if in python
if a:
    print('a is True.')

if b:
    print('Is it right?')
else:
    print('b is False.')


# 8. loop in python, for
for i in [1, 2, 3, 4, 5]:
    print(i)


# 9. function in python


def greeting():
    print('Hello, python!')


greeting()
