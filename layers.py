'''
ECE 579
Machine Learning Final Project
Yuhang Zhou
yz853
5/8/2019

Classes of layers
'''
import numpy as np


class conv:
    def __init__(self, Cin, Cout, kernel=5, learning_rate=0.01):
        self.Cin = Cin
        self.Cout = Cout
        self.K = kernel
        self.W = 0.1 * np.random.randn(Cout, Cin, kernel, kernel)
        self.b = np.zeros((Cout, 1))
        self.lr = learning_rate
        self.x = None

    def forward(self, X):
        self.x = X
        (Cin, H, W) = X.shape  # n, channel, height, width
        
        HH = H - self.K + 1  # height and width of output feature map
        WW = W - self.K + 1
        out = np.zeros((self.Cout, HH, WW))

        for c in range(self.Cout):
            for h in range(HH):
                for w in range(WW):
                    out[c, h, w] = np.sum(
                        X[:, h:h + self.K, w:w + self.K] * self.W[c, :, :, :]) + self.b[c]

        return out

    def backward(self, dout):
        X = self.x
        dx = np.zeros(X.shape)
        dw = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        (Co, H, W) = dout.shape

        for c in range(Co):
            for h in range(H):
                for w in range(W):
                    dw[c, :, :, :] += dout[c, h, w] * \
                        X[:, h:h + self.K, w:w + self.K]
                    dx[:, h:h + self.K, w:w + self.K] += dout[c, h, w] * \
                        self.W[c, :, :, :]

        for c in range(Co):
            db[c] = np.sum(dout[c, :, :])
        # update parameters
        self.W -= self.lr * dw
        self.b -= self.lr * db
        return dx


class fc:
    def __init__(self, D_in, D_out, learning_rate=0.01):
        self.W = 0.01 * np.random.randn(D_in, D_out)
        self.b = np.zeros((D_out,1))
        self.x = None
        self.lr = learning_rate

    def forward(self, X):
        out = np.dot(X, self.W) + self.b.T
        self.x = X
        return out

    def backward(self, dout):
        X = self.x
        if dout.shape[0] == X.shape[0]:
            dout = dout.T
        dW = np.dot(dout,X)
        db = np.sum(dout, axis=1, keepdims=True)
        dx = np.dot(dout.T, self.W.T)
        self.W -= self.lr * dW.T
        self.b -= self.lr * db
        return dx


class max_pool:
    def __init__(self, kernel=2, stride=2):
        self.K = kernel
        self.s = stride
        self.x = None

    def forward(self, X):
        self.x = X
        (Cin, H, W) = X.shape
        HH = (H - self.K) / self.s + 1
        WW = (W - self.K) / self.s + 1
        out = np.zeros((Cin, int(HH), int(WW)))

        for c in range(Cin):
            for h in range(int(H / self.s)):
                for w in range(int(W / self.s)):
                    out[c, h, w] = np.max(
                        X[c, h*self.s:h*self.s + self.K, w*self.s:w*self.s + self.K])

        return out

    def backward(self, dout):
        X = self.x
        (C, H, W) = X.shape
        dx = np.zeros(X.shape)

        for c in range(C):
            for h in range(0, H, self.K):
                for w in range(0, W, self.K):
                    st = np.argmax(X[c, h:h + self.K, w:w + self.K])
                    (idx, idy) = np.unravel_index(st, (self.K, self.K))
                    dx[c, h + idx, w + idy] = dout[c, int(h / self.K), int(w / self.K)]
                    
        return dx



class relu:
    def __init__(self):
        self.x = None
        
    def forward(self, X):
        self.x = X
        out = X.copy()
        out[out < 0] = 0
        return out
    
    def backward(self, dout):
        X=self.x
        dx = dout.copy()
        dx[X < 0] = 0
        return dx


class softmax:
    def __init__(self):
        self.out = None
    
    def forward(self, X):
        exp = np.exp(X, dtype=float)
        self.out = exp / np.sum(exp)
        return self.out

    def backward(self, dout):
        dx = self.out.T - dout.reshape(dout.shape[0], 1)
        return dx
        
        
        

