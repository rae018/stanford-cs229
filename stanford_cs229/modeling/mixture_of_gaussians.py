import numpy as np

import sys
sys.path.append('../utils')
import util
import csv_plotter
import math
# Below is used to evalue the probability based on Gaussian PDF
def gaus_pdf(x,mu,Sigma):
    d = len(x)
    prob = np.exp(-(1/2)*(x-mu).T@np.linalg.solve(Sigma,(x-mu)))
    prob = prob*(np.linalg.det(Sigma)**(-1/2))/((2*np.pi)**(d/2))
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
    if type(train_path) is list: # Sometimes we want to combine data
        X = []
        for path in train_path:
            X.append(util.load_our_data(path))
        X = np.vstack(X)
    else:
        X = util.load_our_data(train_path)

    X = X[:,:3] # We are only using accelerometer data
    #print(X)
    n = len(X[:,0])
    """
    X_merged = [] # Now we combine every 3 rows
    for i in range(n):
        new_row = X[3*i:3*i+3,:].flatten()
        if len(new_row) == 3*3:
            X_merged.append(X[3*i:3*i+3,:].flatten())
    X_merged = np.vstack(X_merged)
    X = X_merged
    print(X_merged)
    """
    print(X)
    # k is the number of classes
    k = 5
    n = len(X[:,0])
    d = len(X[0,:])

    loss_prev = 10**(5) # Initialize to absurdity
    loss_cur = 0

    # w is size (total data-points, total classes)
    # mu is size (dim, total classes)
    # Sigma is size (dim, dim, total classes)
    w = np.ones((n,k))/k
    #mu = np.random.randn(d,k) # Random Gaussian centroids
    mu = 20*np.random.randn(d,k)+350
    Sigma = np.zeros((d,d,k))
    phi = np.ones(k)/k # We start with no prior distribution
    for j in range(k):
        Sigma[:,:,j] = 50*np.identity(d) # Initialize Sigma to identity for invertibility

    while np.abs(loss_prev-loss_cur)>10**(-3):
        loss_prev = loss_cur
        # E step
        for i in range(n):
            #print('gaus',np.array([gaus_pdf(X[i,:],mu[:,j],Sigma[:,:,j]) for j in range(k)]))
            w[i,:] = phi*np.array([gaus_pdf(X[i,:],mu[:,j],Sigma[:,:,j]) for j in range(k)])
            w[i,:] = w[i,:]/sum(w[i,:]) # Now normalize
            #for elem in w[i,:]:
            #    if math.isnan(elem):
            #        print(w[i,:])
        #print(w)
        # M step
        #print('w',w)
        sum_w_vec = np.zeros(k)
        nan_exists = False
        for i in range(n):
            nan_exists = False # Reset nan exists flag
            for elem in w[i,:]:
                if math.isnan(elem):
                    nan_exists = True
            if nan_exists:
                continue # Don't include this row if there is a nan
            sum_w_vec += w[i,:]
        #print(sum_w_vec)
        #sum_w_vec = np.sum(w, axis=0) # Sum along the rows
        #print('sum_w_vec',sum_w_vec)
        #print(sum_w_vec)
        phi = (1./n)*sum_w_vec
        #print(phi)
        for j in range(k):
            mu[:,j] = sum([w[i,j]*X[i,:] if not math.isnan(w[i,j]) else 0 for i in range(n)]) # MLE for mu
            mu[:,j] = (1./sum_w_vec[j])*mu[:,j]
            Sigma[:,:,j] = sum([w[i,j]*np.outer((X[i,:]-mu[:,j]),(X[i,:]-mu[:,j])) if not math.isnan(w[i,j]) else 0 for i in range(n)])
            Sigma[:,:,j] = Sigma[:,:,j]/sum_w_vec[j]
        loss_cur = loss(X,w,mu,Sigma,phi) # z not important actually
        #print('mu_1',mu[:,0])
        #print('mu_2',mu[:,1])
        #print(Sigma)
        #print(mu)
        print(loss_cur)
    #print('End Results:')
    for i in range(1,k+1):
        print('Sigma_{0}: {1}'.format(i-1,Sigma[:,:,i-1]))

    for i in range(1,k+1):
        print("mu_{0}: {1}".format(i-1,mu[:,i-1]))
    print('phi: ',phi)

    # Below only used for 2-D case
    #mu_1 = mu[:,0]
    #mu_2 = mu[:,1]

    #self.theta = (mu_1.T-mu_0.T)@inv_Sigma
    #self.theta_0 = (1/2)*(mu_0.T@inv_Sigma@mu_0-mu_1.T@inv_Sigma@mu_1)+np.log(phi/(1-phi))

    #x_eval,y_eval = util.load_dataset(valid_path, add_intercept=False)
    #util.plot(x_eval, y_eval, theta, save_path[:-4]+'.jpg', correction=1.0)
    print('Now we find the chosen z values')
    z = np.zeros(n)
    for i in range(n):
        z[i] = np.argmax(w[i,:])
    print(z)
    #z = np.repeat(z, 3)
    t = np.arange(len(z))*(1./60)

    csv_plotter.plot(t=t,data=z,xlabel=r'time ($s$)',ylabel='class',title='Classification')
    csv_plotter.acc_plot(train_path)

if __name__ == "__main__":
    train_path = [r"..\data\front_back_overhand-60.csv"]
    train_path.append(r"..\data\up_down_overhand-60.csv")
    #train_path = r"..\data\up_down_weight-60.csv"
    valid_path = 'ds2_valid.csv'
    save_path = 'unsupervised_plot.txt'
    main(train_path, valid_path, save_path)
    #csv_plotter.acc_plot(train_path)
