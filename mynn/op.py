from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # Use a small std (Xavier-like) to keep activations in a stable range,
        # otherwise the default np.random.normal(std=1) makes the loss explode.
        if initialize_method is np.random.normal:
            std = np.sqrt(2.0 / in_dim)
            self.W = np.random.normal(loc=0.0, scale=std, size=(in_dim, out_dim))
            self.b = np.zeros((1, out_dim))
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
            self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay


    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        X = self.input
        dW = X.T @ grad
        db = np.sum(grad, axis=0, keepdims=True)
        if self.weight_decay:
            dW = dW + self.weight_decay_lambda * self.params['W']
        self.grads['W'] = dW
        self.grads['b'] = db
        return grad @ self.params['W'].T

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


def _im2col(X, k_h, k_w, stride, padding):
    """
    X: [N, C, H, W]
    return cols: [N, C*k_h*k_w, out_H*out_W], out_H, out_W
    """
    N, C, H, W = X.shape
    if padding > 0:
        X_pad = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    else:
        X_pad = X
    out_H = (H + 2*padding - k_h) // stride + 1
    out_W = (W + 2*padding - k_w) // stride + 1

    cols = np.zeros((N, C, k_h, k_w, out_H, out_W), dtype=X.dtype)
    for i in range(k_h):
        i_max = i + stride * out_H
        for j in range(k_w):
            j_max = j + stride * out_W
            cols[:, :, i, j, :, :] = X_pad[:, :, i:i_max:stride, j:j_max:stride]
    # reshape to [N, C*k_h*k_w, out_H*out_W]
    cols = cols.reshape(N, C * k_h * k_w, out_H * out_W)
    return cols, out_H, out_W


def _col2im(cols, X_shape, k_h, k_w, stride, padding):
    """
    Inverse of _im2col: scatter-add gradients back to image space.
    cols: [N, C*k_h*k_w, out_H*out_W]
    """
    N, C, H, W = X_shape
    out_H = (H + 2*padding - k_h) // stride + 1
    out_W = (W + 2*padding - k_w) // stride + 1
    cols_reshaped = cols.reshape(N, C, k_h, k_w, out_H, out_W)
    H_pad = H + 2*padding
    W_pad = W + 2*padding
    X_pad = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)
    for i in range(k_h):
        i_max = i + stride * out_H
        for j in range(k_w):
            j_max = j + stride * out_W
            X_pad[:, :, i:i_max:stride, j:j_max:stride] += cols_reshaped[:, :, i, j, :, :]
    if padding > 0:
        return X_pad[:, :, padding:padding+H, padding:padding+W]
    return X_pad


class conv2D(Layer):
    """
    The 2D convolutional layer. Implemented with im2col + matrix multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_h = k_h
        self.k_w = k_w
        self.stride = stride
        self.padding = padding

        # Kaiming-style init so the CNN can train stably on CPU.
        fan_in = in_channels * k_h * k_w
        std = np.sqrt(2.0 / fan_in)
        self.W = np.random.normal(loc=0.0, scale=std, size=(out_channels, in_channels, k_h, k_w))
        self.b = np.zeros((out_channels,))

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self.input = None
        self._cache = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, in_channels, H, W]
        weight : [out_channels, in_channels, k, k]
        output : [batch, out_channels, out_H, out_W]
        """
        self.input = X
        N = X.shape[0]
        cols, out_H, out_W = _im2col(X, self.k_h, self.k_w, self.stride, self.padding)
        # cols: [N, C*k*k, L], W flat: [out, C*k*k]
        W_flat = self.params['W'].reshape(self.out_channels, -1)
        # [N, out, L]
        out = np.einsum('oc,ncl->nol', W_flat, cols) + self.params['b'].reshape(1, -1, 1)
        out = out.reshape(N, self.out_channels, out_H, out_W)
        self._cache = (cols, out_H, out_W)
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, out_H, out_W]
        return: gradient w.r.t. input, same shape as self.input
        """
        cols, out_H, out_W = self._cache
        N = grads.shape[0]
        # reshape grads to [N, out, L]
        grad_reshaped = grads.reshape(N, self.out_channels, out_H * out_W)

        # dW: [out, C*k*k] = sum_n grad[n] @ cols[n].T  ==> einsum
        W_flat = self.params['W'].reshape(self.out_channels, -1)
        dW_flat = np.einsum('nol,ncl->oc', grad_reshaped, cols)
        dW = dW_flat.reshape(self.params['W'].shape)
        db = grad_reshaped.sum(axis=(0, 2))

        if self.weight_decay:
            dW = dW + self.weight_decay_lambda * self.params['W']

        self.grads['W'] = dW
        self.grads['b'] = db

        # dX_cols: [N, C*k*k, L] = W_flat.T @ grad_reshaped
        dcols = np.einsum('oc,nol->ncl', W_flat, grad_reshaped)
        dX = _col2im(dcols, self.input.shape, self.k_h, self.k_w, self.stride, self.padding)
        return dX

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class MaxPool2D(Layer):
    """
    Simple max pooling layer.
    """
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.optimizable = False
        if isinstance(kernel_size, int):
            self.k_h = self.k_w = kernel_size
        else:
            self.k_h, self.k_w = kernel_size
        self.stride = stride
        self.input_shape = None
        self.argmax = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        N, C, H, W = X.shape
        k_h, k_w, s = self.k_h, self.k_w, self.stride
        out_H = (H - k_h) // s + 1
        out_W = (W - k_w) // s + 1

        # use im2col-style indexing per channel by reshaping channel into batch dim
        X_r = X.reshape(N * C, 1, H, W)
        cols, _, _ = _im2col(X_r, k_h, k_w, s, padding=0)
        # cols: [N*C, k*k, out_H*out_W]
        argmax = np.argmax(cols, axis=1)  # [N*C, L]
        out_flat = np.take_along_axis(cols, argmax[:, None, :], axis=1).squeeze(1)
        out = out_flat.reshape(N, C, out_H, out_W)

        self.argmax = argmax
        self._k = (k_h, k_w, s, out_H, out_W)
        return out

    def backward(self, grads):
        N, C, H, W = self.input_shape
        k_h, k_w, s, out_H, out_W = self._k
        grads_flat = grads.reshape(N * C, out_H * out_W)

        dcols = np.zeros((N * C, k_h * k_w, out_H * out_W), dtype=grads.dtype)
        np.put_along_axis(dcols, self.argmax[:, None, :], grads_flat[:, None, :], axis=1)
        dX = _col2im(dcols, (N * C, 1, H, W), k_h, k_w, s, padding=0)
        return dX.reshape(N, C, H, W)


class Flatten(Layer):
    """
    Flatten everything but batch dimension.
    """
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output

    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.optimizable = False
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.labels = labels
        batch_size = predicts.shape[0]
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
        # clip to avoid log(0)
        probs_safe = np.clip(probs, 1e-12, 1.0)
        log_probs = np.log(probs_safe[np.arange(batch_size), labels])
        loss = -np.mean(log_probs)

        # cache for backward
        self.predicts = probs  # store probabilities (after softmax) or raw if cancelled
        self._batch_size = batch_size
        return loss

    def backward(self):
        # grad of cross-entropy + softmax w.r.t. logits = (p - y_onehot) / N
        batch_size = self._batch_size
        if self.has_softmax:
            grads = self.predicts.copy()
            grads[np.arange(batch_size), self.labels] -= 1.0
            grads = grads / batch_size
        else:
            # if softmax cancelled, fall back to -1 / p_i for the true class
            grads = np.zeros_like(self.predicts)
            grads[np.arange(batch_size), self.labels] = -1.0 / np.clip(
                self.predicts[np.arange(batch_size), self.labels], 1e-12, None
            )
            grads = grads / batch_size
        self.grads = grads
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self

class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
