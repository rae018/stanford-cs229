import numpy as np
from stanford_cs229.utils.util import *
from datetime import datetime

class KMeans:
    def __init__(self, max_iter=10000, eps=10**(-6), verbose=True):
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def loss(self, X, assignments, centroids):
        n, d = X.shape
        loss = 0
        for i in range(n):
            loss += np.linalg.norm(X[i,:]-centroids[assignments[i],:])**2
        return loss

    def assign(self, X, k, period=1, random=True):
        # k is total number of clusters
        # period is the number of rows of X included in a data point
        # If random is false, we hard-code the initial clusters
        start = datetime.now()
        # The first two columns are the acceleration data
        X = X[:,:2]
        n, d = X.shape
        x_acc = X[:,0]
        y_acc = X[:,1]

        if self.verbose:
            if n % period != 0:
                print('The period did not go in evenly. This may not be promising')

        # Obtain reshaped data as a result of period > 1
        n,d,X = reshape_data(X,period)
        centroids = np.zeros((k,d))
        assignments = np.zeros(n,dtype=int)

        # If the cluster centroids are randomized
        if random:
            for j in range(k):
                mu = np.random.uniform()*(np.max(x_acc)-np.min(x_acc))+np.min(x_acc)
                mu = np.hstack([mu,np.random.uniform()*(np.max(y_acc)-np.min(y_acc))+np.min(y_acc)])
                centroids[j,:] = np.tile(mu,period) # Establish random centroid
        else:
            if k!=5:
                if self.verbose:
                    print('K does not work with deterministic centroid initialization.')
                    print('Nothing I can really do for you at this point :/')
            mid_x = (np.max(x_acc)+np.min(x_acc))/2
            mid_y = (np.max(y_acc)+np.min(y_acc))/2
            mu_1 = np.array([np.min(x_acc),mid_y])
            mu_2 = np.array([np.max(x_acc),mid_y])
            mu_3 = np.array([mid_x,np.min(y_acc)])
            mu_4 = np.array([mid_x,np.max(y_acc)])
            mu_7 = np.array([mid_x,mid_y])
            # [ blue, grey, light green, green, red ]
            centroids = np.tile(np.vstack([mu_7,mu_3,mu_1,mu_2,mu_4]),period) # Tile by period
        num_iter = 0

        loss = 0
        # Initialize to absurdity
        loss_prev = 10**(10)
        assignments_prev = np.ones(n)
        while np.abs(loss-loss_prev)>self.eps and num_iter<self.max_iter:
            loss_prev = loss
            # Assignment step
            for i in range(n):
                assignments[i] = np.argmin([np.linalg.norm(X[i,:]-centroids[j,:])**2 for j in range(k)])
            # Update step
            for j in range(k):
                mu_j = np.sum((assignments[:,None]==j)*X,axis=0) # Only keep relevant assignments
                mu_j /= sum((assignments==j))
                centroids[j,:] = mu_j
            num_iter += 1
            if self.verbose: 
              print('Iteration: {}'.format(num_iter), end='')
              loss = self.loss(X, assignments, centroids)
              print(', Loss: {}'.format(loss))

        stop = datetime.now()
        print('Time: {}'.format(stop-start))
        return assignments
