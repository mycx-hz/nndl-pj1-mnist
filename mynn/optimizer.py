from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    """
    SGD with momentum:
        v <- mu * v + grad
        w <- w - lr * v
    """
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        # velocity buffers keyed by id(param array) so we can match across steps
        self.velocity = {}
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocity[id(layer)] = {
                    key: np.zeros_like(val) for key, val in layer.params.items()
                }

    def step(self):
        for layer in self.model.layers:
            if not layer.optimizable:
                continue
            v_layer = self.velocity[id(layer)]
            for key in layer.params.keys():
                grad = layer.grads[key]
                if layer.weight_decay:
                    # decoupled style: shrink the parameter directly
                    layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                v_layer[key] = self.mu * v_layer[key] + grad
                layer.params[key] = layer.params[key] - self.init_lr * v_layer[key]
