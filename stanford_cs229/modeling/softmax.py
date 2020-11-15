import numpy as np
from stanford_cs229.utils.util import *
from datetime import datetime

class SoftmaxRegression:
    def __init__(self, lr=0.01, max_iter=10000, eps=1e-15, verbose=True):
        self.theta = None
        self.lr = lr
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def likelihood(self, X, Y):
        pass

    def gradient(self, X, Y):
        n, d = X.shape
        k = self.theta.shape[1]

        result = np.zeros(self.theta.shape)
        for i in range(n):
            x = X[i]
            y = Y[i]

            probs = 0
            for j in range(k):
                probs += np.exp(self.theta[:,j].T.dot(x))
            probs = 1. / probs
            for l in range(k):
                result[:, l] += x * ((y == l) - np.exp(self.theta[:,l].T.dot(x)) * probs)
        return result

    def gradient_vec(self, X, Y):
        n, d = X.shape
        k = self.theta.shape[1]

        Y2 = np.zeros((n, k))
        Y2[np.arange(n, dtype='int32'), Y] = 1

        exp = np.exp(np.matmul(X, self.theta))
        exp = exp / np.expand_dims(np.sum(exp, 1), 1)
        return np.matmul(X.T, Y2 - exp)

    def predict(self, X):
        exp = np.exp(np.matmul(X, self.theta))
        exp = exp / np.expand_dims(np.sum(exp, 1), 1)
        return np.argmax(exp, 1)

    def train(self, X, Y, k, period=1):
        start = datetime.now()
        n, d = X.shape

        # Obtain reshaped data as a result of period > 1
        n, d, X = reshape_data(X, period)

        self.theta = np.zeros((d, k))

        num_iter = 0

        while True:
            prev_theta = np.copy(self.theta)
            self.theta += self.lr * self.gradient_vec(X, Y)

            num_iter += 1
            update_size = np.linalg.norm(self.theta - prev_theta, ord=1)
            if self.verbose:
                print('Iteration {:n}, Update Size {:f}'.format(num_iter, update_size))
            if num_iter > self.max_iter or np.isnan(update_size) or update_size < self.eps:
                break
        stop = datetime.now()
        print('Training Time: {}'.format(stop-start))
