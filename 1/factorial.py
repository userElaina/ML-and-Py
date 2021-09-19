def f(a):
    if a<0:
        return 'Error: Do you want the factorial of a negative?'
    c=1
    for i in range(1,a+1):
        c*=i
    return c
def _f(a):
    try:
        a=int(a)
        return f(a)
    except ValueError:
        return 'Error: Do you want the factorial of a string?'
while 1:
    q=input()
    if q=='exit':
        break
    print(_f(q))

'''
5
12
0
-1
Hello world
exit
'''