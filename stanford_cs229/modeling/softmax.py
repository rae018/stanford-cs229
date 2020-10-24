import numpy as np
from stanford_cs229.utils.util import *

class SoftmaxRegression:
    def __init__(self, lr=0.001, max_iter=1000000, eps=1e-15, verbose=True):
        self.theta = None
        self.lr = lr
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def likelihood(self, X, Y):
        pass

    def gradient(self, X, Y):
        """ This doesn't work at the moment """
        n, d = X.shape
        k = self.theta.shape[1]

        result = np.zeros(self.theta.shape)
        for i in range(n):
            x = X[i]
            y = Y[i]

            probs = np.zeros(x.shape)
            for j in range(k):
                probs += np.exp(np.matmul(self.theta[:,j].T, x))
            probs = 1/probs
            for l in range(k):
                result[:, l] = x * ((y == l) - np.exp(self.theta[:,l].T, x) * probs)

        return result

    def gradient_vec(self, X, Y):
        n, d = X.shape
        k = self.theta.shape[1]

        Y2 = np.zeros((n, k))
        Y2[np.arange(n), Y] = 1

        exp = np.exp(np.matmul(X, self.theta))
        exp = exp / np.expand_dims(np.sum(exp, 1), 1)
        return np.matmul(X.T, Y2 - exp)

    def predict(self, X):
        exp = np.exp(np.matmul(X, self.theta))
        exp = exp / np.expand_dims(np.sum(exp, 1), 1)
        return np.argmax(exp, 1)

    def train(self, X, Y, k):
        n, d = X.shape
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
                print('DONE TRAINING')
                break

def main():
    X, Y = load_dataset('', add_intercept=True)
    k = 1

    softmax = SoftmaxRegression()
    softmax.train(X, Y, k)

if __name__ == '__main__':
    main()
