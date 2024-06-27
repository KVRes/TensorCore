import numpy as np

class Variable:
    def __init__(self, data) -> None:
        if data is None:
            raise ValueError("data is None")

        self.data = data
        self.grad : np.ndarray | None = None
        self.creator = None

    def __str__(self) -> str:
        return super().__str__() + " with value: " + self.data.__str__()

    def set_creator(self, fx):
        self.creator = fx
