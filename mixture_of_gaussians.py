import numpy as np
import util

# Below is used to evalue the probability based on Gaussian PDF
def gaus_pdf(x,mu,Sigma):
    d = len(x)
    prob = np.exp(-(1/2)*(x-mu).T@np.linalg.solve(Sigma,(x-mu)))
    prob *= (np.linalg.det(Sigma)**(-1/2))/((2*np.pi)**(d/2))
    return prob

# The loss function for Gaussian Discriminant  Analysis
def loss(X,z,mu,Sigma,phi):
    n = len(X[:,0])
    k = len(phi)
    loss = 0
    for i in range(n):
        loss += np.log(sum([phi[j]*gaus_pdf(X[i,:],mu[:,j],Sigma[:,:,j]) for j in range(k)]))
    return loss

def main(train_path, valid_path, save_path):
    #X,_ = util.load_dataset(train_path, add_intercept=False) # No labels

    mean1 = [0,0]
    mean2 = [8,5]
    cov1 = np.identity(2)
    cov2 = np.identity(2)
    X1 = np.random.multivariate_normal(mean1,cov1,1600)
    X2 = np.random.multivariate_normal(mean2,cov2,800)
    X = np.vstack([X1,X2])

    # k is the number of classes
    k = 2
    n = len(X[:,0])
    d = len(X[0,:])

    loss_prev = 10**(5) # Initialize to absurdity
    loss_cur = 0

    # w is size (total data-points, total classes)
    # mu is size (dim, total classes)
    # Sigma is size (dim, dim, total classes)
    w = np.ones((n,k))/k
    #mu = np.random.randn(d,k) # Random Gaussian centroids
    mu = np.array([[8,-5],[8,-5]])
    Sigma = np.zeros((d,d,k))
    phi = np.ones(k)/k # We start with no prior distribution
    for j in range(k):
        Sigma[:,:,j] = np.identity(d) # Initialize Sigma to identity for invertibility

    while np.abs(loss_prev-loss_cur)>10**(-5):
        loss_prev = loss_cur
        # E step
        for i in range(n):
            #print('gaus',np.array([gaus_pdf(X[i,:],mu[:,j],Sigma[:,:,j]) for j in range(k)]))
            w[i,:] = phi*np.array([gaus_pdf(X[i,:],mu[:,j],Sigma[:,:,j]) for j in range(k)])
            w[i,:] = w[i,:]/sum(w[i,:]) # Now normalize

        # M step
        #print('w',w)
        sum_w_vec = np.sum(w, axis=0) # Sum along the rows
        #print('sum_w_vec',sum_w_vec)
        phi = (1./n)*sum_w_vec
        for j in range(k):
            mu[:,j] = sum([w[i,j]*X[i,:] for i in range(n)]) # MLE for mu
            mu[:,j] = (1./sum_w_vec[j])*mu[:,j]
            #print(mu[:,j])
            Sigma[:,:,j] = sum([w[i,j]*np.outer((X[i,:]-mu[:,j]),(X[i,:]-mu[:,j])) for i in range(n)])
            Sigma[:,:,j] = Sigma[:,:,j]/sum_w_vec[j]
        loss_cur = loss(X,w,mu,Sigma,phi) # z not important actually
        #print('mu_1',mu[:,0])
        #print('mu_2',mu[:,1])
    print('End Results:')
    print('Sigma1: ',Sigma[:,:,0])
    print('Sigma2: ',Sigma[:,:,1])
    print('mu_1',mu[:,0])
    print('mu_2',mu[:,1])
    print('phi: ',phi)

    # Below only used for 2-D case
    #mu_1 = mu[:,0]
    #mu_2 = mu[:,1]

    #self.theta = (mu_1.T-mu_0.T)@inv_Sigma
    #self.theta_0 = (1/2)*(mu_0.T@inv_Sigma@mu_0-mu_1.T@inv_Sigma@mu_1)+np.log(phi/(1-phi))

    #x_eval,y_eval = util.load_dataset(valid_path, add_intercept=False)
    #util.plot(x_eval, y_eval, theta, save_path[:-4]+'.jpg', correction=1.0)


if __name__ == "__main__":
    train_path = 'ds2_train.csv'
    valid_path = 'ds2_valid.csv'
    save_path = 'unsupervised_plot.txt'
    main(train_path, valid_path, save_path)
