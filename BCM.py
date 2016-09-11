# %load BCM.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
# Define BCM class by quadratic local learning rule
# Relu activation function
# Update weight per sample
# Enable batch normalization
# Enable objective function

class bcm:
    """BCM learning
    Parameter:
    eta: float, learning rate (between 0.0 - 1.0)
    n_iter: int, passes over the training dataset
    ny: number of output neurons
    batchsize: float, percentage of data that are used to update the weight once
    
    Attributes:
    w_: 1d-array, weights after fitting
    error_: list, number of misclassification in every epoch
    """
    def __init__(self, eta = 0.1, n_epoch=10, ny=1, batch=10, tau=100, thres=0, p = 2,random_state = None, shuffle = True, nonlinear = None, obj_type = 'QBCM'):
        self.eta = eta
        self.n_epoch = n_epoch
        self.ny = ny
        self.tau = tau  # Time constant for calculating thresholds
        self.thres = thres * np.ones(ny)
        self.p = p    # int, power for threshold computation
        self.y_thres = []      # Storaged y for studying effect of threshold
        self.shuffle = shuffle
        self.nonlinear = nonlinear
        self.batch = batch
        self.obj_type = obj_type
        
        if random_state:
            np.random.seed(random_state)
            
    def fit(self, X):
        """fitting training data
        Parameter:
        X: {array-like}, shape = [n_samples,n_features]
        Returns: self:object
        ny: value, number of output neurons
        """
        # Weights initialized as normal distribution
        self.w_ = np.random.randn(X.shape[1],self.ny) # 2*1
        self.w_track = []
        self.obj = []
        
        # Use elementwise training 
        threshold = self.thres
        self.thres = []
        self.obj = []
        obj_x1 = np.zeros(self.ny)   # Two iterative terms in objective function
        obj_x2 = np.zeros(self.ny)   # Two iterative terms in objective function
        bcm_obj = np.zeros(self.ny)
        for _ in range(self.n_epoch):
            if self.shuffle:
                X = self._shuffle(X)
            if self.batch:
                X_batch = []
                for k in range(0,X.shape[0],self.batch):
                    X_batch.append(np.mean(X[k:k+self.batch,:],axis = 0))
                X_batch = np.vstack(X_batch)
            for i, xi in enumerate(X_batch):   # elementwise training for all samples
                y = np.zeros(self.ny)
                for j in range(self.ny):
                    y[j] = self._activation(np.dot(xi,self.w_[:,j]),nonlinear = self.nonlinear)
                    if self.obj_type == 'QBCM':
                        if self.nonlinear == 'Sigmoid':
                            self.w_[:,j][:,None] = self.w_[:,j][:,None]+ self.eta * xi[:,None]* _dsigmoid(y[j])*(y[j] - threshold[j])
                        else:
                            self.w_[:,j][:,None] = self.w_[:,j][:,None]+ self.eta * xi[:,None]* y[j]*(y[j] - threshold[j])
                    elif self.obj_type == 'kurtosis':
                        if self.nonlinear == 'Sigmoid':
                            self.w_[:,j][:,None] = self.w_[:,j][:,None]+ 4 * self.eta * xi[:,None]* _dsigmoid(y[j])* y[j]*(y[j]**2 - threshold[j])
                            self.w_[:,j] = self.w_[:,j]/np.sqrt(np.sum((self.w_[:,j])**2))
                        else:
                            self.w_[:,j][:,None] = self.w_[:,j][:,None]+ 4 * self.eta * xi[:,None]* y[j]*(y[j]**2 - threshold[j])
                            self.w_[:,j] = self.w_[:,j]/np.sqrt(np.sum((self.w_[:,j])**2))
                    threshold[j] = self._ema(x = threshold[j],y = y[j],power = self.p)  #
                    bcm_obj[j] = _obj(X,w = self.w_[:,j],obj_type = self.obj_type,nonlinear = self.nonlinear)
                w_tmp = np.concatenate(self.w_.T, axis=0)    # make 2*2 matrix into 1*4, preparing for weight tracking
                self.y_thres.append(y)
                self.thres.append(threshold.tolist())
                self.w_track.append(w_tmp.tolist())
                self.obj.append(bcm_obj.tolist())
        return self
    
    def _shuffle(self,X):
        r = np.random.permutation(len(X))
        return X[r]
    
    def _ema(self,x,y,power = 2):    #Good way to implement exponential moving average
        # x is the iterative variable, y is the function being averaged
        h = np.exp(-1/self.tau)
        return x*h+(y**power)*(1-h)

    def _activation(self,y,nonlinear = 'Relu'):
        if self.nonlinear == 'Sigmoid':
            y = _sigmoid (y)
        elif self.nonlinear == 'Relu':
            y = (y>=0)* y
        return y

def _dsigmoid(z):
    return _sigmoid(z)*(1-(_sigmoid(z)))

def _sigmoid(z):
    return 1/(1+np.exp(-z))

# objective function using all data
def _obj(X,w,obj_type='QBCM',nonlinear='Relu'):
    c = np.dot(X,w)
    if nonlinear == 'Sigmoid':
        c = _sigmoid (c)
    elif nonlinear == 'Relu':
        c = (1-(c<0))* c
        
    if obj_type == 'QBCM':
        obj1 = (c**3).mean(axis = 0)
        obj2 = (c**2).mean(axis = 0)
        obj = obj1/3 - obj2**2/4
    elif obj_type == 'kurtosis':
        obj1 = (c**4).mean(axis = 0)
        obj2 = (c**2).mean(axis = 0)
        obj = obj1 - obj2**2*3
    elif obj_type == 'skewness':
        obj1 = (c**3).mean(axis = 0)
        obj2 = (c**2).mean(axis = 0)
        obj = np.divide(obj1,obj2**1.5)
        
    return obj

# Perform zca whitening
def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    #ZCAMatrix = np.dot(U, np.diag(1.0/np.sqrt(S + epsilon))) # [M x M]
    return ZCAMatrix


# Plot trained BCM output, threshold, weights and objective function
def bcm_train(s_rt_wt,eta = 0.0001, n_epoch = 10, batch = 10, ny = 2,tau = 200, thres = 0, p = 2,random_state = None, shuffle = True, nonlinear = 'Relu', obj_type = 'QBCM'):
    BCM_laplace = bcm(eta = eta,n_epoch = n_epoch,batch = batch,ny = ny,tau = tau, thres = thres, p = p,random_state=random_state , shuffle = shuffle, nonlinear = nonlinear, obj_type = obj_type)

    t0 = time()
    BCM_laplace.fit(s_rt_wt)
    print("done in %0.3fs" % (time()-t0))

    #x_plot = range(n_iter * len(X))
    plt_range = n_epoch * len(s_rt_wt)
    BCM_laplace_thres = np.vstack(BCM_laplace.thres)
    BCM_laplace_out = np.vstack(BCM_laplace.y_thres)
    BCM_laplace_w = np.vstack(BCM_laplace.w_track)
    BCM_laplace_obj = np.vstack(BCM_laplace.obj)

    n_row = 1
    n_column = ny

    BCM_laplace_titles= ["BCM unit %d" % i for i in range (ny)]
    # Plot y and threshold
    for i in range(ny):
        plt.subplot(n_row,n_column,i+1)
        plt.plot(BCM_laplace_out[:plt_range,i],'g-',label = 'output')
        plt.plot(BCM_laplace_thres[:plt_range,i],'r--',label = 'threshold')
        plt.title(BCM_laplace_titles[i])

    plt.legend(loc = 'upper right')

    plt.figure()
    # Plot weights

    BCM_laplace_titles= ["BCM weights %d" % i for i in range (len(BCM_laplace_w.T))]

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.plot(BCM_laplace_w[:,0],'r',label = 'weight')
    ax1.set_title("BCM unit %d , weights %d" % (0,0))
    ax2.plot(BCM_laplace_w[:,2],'r',label = 'weight')
    ax2.set_title("BCM unit %d , weights %d" % (1,0))
    ax3.plot(BCM_laplace_w[:,1],'r',label = 'weight')
    ax3.set_title("BCM unit %d , weights %d" % (0,1))
    ax3.set_xlabel('# of iterations')
    ax4.plot(BCM_laplace_w[:,3],'r',label = 'weight')
    ax4.set_title("BCM unit %d , weights %d" % (1,1))
    ax4.set_xlabel('# of iterations')

    BCM_laplace_titles= ["BCM %d objective function " % i for i in range (len(BCM_laplace_w.T))]

    # plot the objective function
    plt.figure()
    for i in range(ny):
        plt.subplot(n_row,n_column,i+1)
        plt.plot(BCM_laplace_obj[:plt_range,i])
        plt.title(BCM_laplace_titles[i])

def bcm_obj(s_rt_wt,w_min,w_max,reso,para):
    w = np.linspace(w_min,w_max,reso)
    wx,wy = np.meshgrid(w,w)
    w = np.vstack((wx.ravel(),wy.ravel()))
    obj_choice = ['QBCM','kurtosis']
    nonlinear_choice = ['Relu','Sigmoid',None]

    #obj = _obj(s_rt_wt,w,obj_type = 'QBCM',nonlinear = 'Relu')
    ny = para[0]    # 2 output neurons
    n_epoch = para[1]
    p = para[2]
    eta = para[3]
    tau = para[4]
    batch = para[5]
    # Plot a gallery of images
    n_row = len(obj_choice)
    n_col = len(nonlinear_choice)

    fig, ax = plt.subplots(n_row, n_col,figsize=(12,12), sharex=True, sharey=True)
    for i in range(n_row):
        for j in range(n_col):
            #plt.subplot(n_row, n_col, i*j+1 )    
            obj = _obj(s_rt_wt,w,obj_type = obj_choice[i],nonlinear = nonlinear_choice[j])
            nbins=20
            levels = np.percentile(obj, np.linspace(0,100,nbins))
            with sns.axes_style('white'):
                c = ax[i,j].contour(wx,wy,obj.reshape(wx.shape),levels=levels, zorder=-10, cmap=plt.cm.get_cmap('viridis'))
                ax[i,j].plot(s_rt_wt[:,0],s_rt_wt[:,1],'.k', ms=4)

            plt.grid('on')
            plt.colorbar(c, ax=ax[i,j])
            
            # Parameter for training
            nonlinear = nonlinear_choice[j]
            obj_type = obj_choice[i]

            # Traing with BCM local learning rule
            BCM_laplace = bcm(eta = eta,n_epoch = n_epoch,batch = batch,ny = ny,tau = tau, thres = 0, p = p,random_state = None, shuffle = True, nonlinear = nonlinear, obj_type = obj_type)
            BCM_laplace.fit(s_rt_wt)
            BCM_laplace_w = np.vstack(BCM_laplace.w_track)
            
            ax[i,j].plot(BCM_laplace_w[:,0],BCM_laplace_w[:,1],'g')
            ax[i,j].plot(BCM_laplace_w[-1,0],BCM_laplace_w[-1,1],'y*',ms = 15)
            ax[i,j].plot(BCM_laplace_w[:,2],BCM_laplace_w[:,3],'r')
            ax[i,j].set_title((obj_choice[i],nonlinear_choice[j]))
            ax[i,j].plot(BCM_laplace_w[-1,2],BCM_laplace_w[-1,3],'y*',ms = 15)