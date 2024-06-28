import numpy as np

class Variable:
    def __init__(self, data) -> None:
        if data is None:
            raise ValueError("data is None")
        if not isinstance(data, np.ndarray):
            raise TypeError(f'{type(data)} should be ndarray!')

        self.data = data
        self.grad : np.ndarray | None = None
        self.creator = None

    def __str__(self) -> str:
        return super().__str__() + " with value: " + self.data.__str__()

    def set_creator(self, fx):
        self.creator = fx

    def backward(self):
        if self.grad == None:
            self.grad = np.ones_like(self.data) # default grad = 1

        fxs = [self.creator]
        while fxs:
            fx = fxs.pop()
            X, y = fx.input, fx.output
            X.grad = fx.backward(y.grad)
            if X.creator is not None:
                fxs.append(X.creator)
