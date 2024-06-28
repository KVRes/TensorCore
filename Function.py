from Variable import Variable
import Utils as u
import numpy as np

class Function:
    def __call__(self, *inputs):
        Xs = [x.data for x in inputs]
        ys = self.forward(*Xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(u.as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x1, x2):
        return (x1+x2,)

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        X = self.input.data
        grad = gy * 2 * X
        return grad


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        X = self.input.data
        grad = gy * np.exp(X)
        return grad

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x1, x2):
    return Add()(x1, x2)
