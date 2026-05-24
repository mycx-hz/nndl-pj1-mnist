from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN(Layer):
    """
    A simple CNN for MNIST:
        Conv(1->8, 3x3, pad=1) -> ReLU -> MaxPool(2)   # 28x28 -> 14x14
        Conv(8->16, 3x3, pad=1) -> ReLU -> MaxPool(2)  # 14x14 -> 7x7
        Flatten -> Linear(16*7*7 -> 64) -> ReLU -> Linear(64 -> 10)
    Input shape: [batch, 1, 28, 28]
    """
    def __init__(self, in_channels=1, num_classes=10, conv_channels=(8, 16), hidden=64,
                 lambda_list=None, build=True):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_channels = tuple(conv_channels)
        self.hidden = hidden
        self.lambda_list = lambda_list  # list of 4 lambdas for the 4 trainable layers

        if build:
            self._build()

    def _build(self):
        c1, c2 = self.conv_channels
        self.layers = [
            conv2D(self.in_channels, c1, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            conv2D(c1, c2, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Flatten(),
            Linear(c2 * 7 * 7, self.hidden),
            ReLU(),
            Linear(self.hidden, self.num_classes),
        ]
        if self.lambda_list is not None:
            opt_layers = [l for l in self.layers if l.optimizable]
            assert len(self.lambda_list) == len(opt_layers), \
                f'lambda_list length {len(self.lambda_list)} != trainable layers {len(opt_layers)}'
            for lam, layer in zip(self.lambda_list, opt_layers):
                layer.weight_decay = True
                layer.weight_decay_lambda = lam

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # Accept [N, 784] or [N, 1, 28, 28]
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, 28, 28)
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def save_model(self, save_path):
        meta = {
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'conv_channels': self.conv_channels,
            'hidden': self.hidden,
        }
        param_list = [meta]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, save_path):
        with open(save_path, 'rb') as f:
            param_list = pickle.load(f)
        meta = param_list[0]
        self.in_channels = meta['in_channels']
        self.num_classes = meta['num_classes']
        self.conv_channels = meta['conv_channels']
        self.hidden = meta['hidden']
        self._build()
        opt_layers = [l for l in self.layers if l.optimizable]
        assert len(opt_layers) == len(param_list) - 1
        for layer, p in zip(opt_layers, param_list[1:]):
            layer.params['W'] = p['W']
            layer.params['b'] = p['b']
            layer.W = p['W']
            layer.b = p['b']
            layer.weight_decay = p['weight_decay']
            layer.weight_decay_lambda = p['lambda']
