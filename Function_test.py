from Function import Square, Exp
from Variable import Variable
import numpy as np

def Test1():
    x = Variable(11)
    f = Square()
    print(f(x))

def Test2():
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    # <-----| forward
    # C(B(A(x)))
    # |-----> backward
    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)

Test2()
