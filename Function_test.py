from Function import Square, Exp, square
from Variable import Variable
import numpy as np
import unittest

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

def Test3():
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    # C(B(A(x)))
    # |-----> backward
    # y b a
    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x
    print('Pass Assert, try BP')

    y.grad = np.array(1.)
    C = y.creator
    b = C.input
    b.grad = C.backward(y.grad)

    B = b.creator
    a = B.input
    a.grad = B.backward(b.grad)

    A = a.creator
    x = A.input
    x.grad = A.backward(a.grad)
    print(x.grad)

def Test4():
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.)
    y.backward()
    print(x.grad)

Test4()

class SquareTest(unittest.TestCase):
    def test_forward(self):
        X = Variable(np.array(2.))
        y = square(X)
        expected = np.array(4.)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        X = Variable(np.array(3.))
        y = square(X)
        y.backward()

        expected = np.array(6.)
        self.assertEqual(X.grad, expected)
