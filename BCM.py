# %load BCM.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
# Define BCM class by quadratic local learning rule
# Relu/Sigmoid activation function
# Update weight per sample
# Enable 'QBCM' and 'kurtosis' objective function
# Hasn't applied batch learning
# Enable decat time constant
class bcm:
    """BCM learning
    Parameter:
    eta: float, learning rate (between 0.0 - 1.0)
    n_epoch: int, passes over the training dataset
    ny: int, number of output neurons
    batchsize: int, percentage of data that are used to update the weight once
    tau: float, time constant for BCM leanring rule
    thres: float, initial BCM threshold
    p: int, power of BCM threshold function
    random_state: int, seed for random number generator
    shuffle: Boolean, whether shuffle whole datasize for each epoch
    nonlinear: String, type of activation function, can be choosen from 'Relu', 'Sigmoid' amd None
    obj_type: type of local learning rule, can be choosen from 'QBCM' and 'kurtosis'
    decay: float, decay time constant, 0.0 means no decay

    Attributes:
    w_: array input dimension * num of output neurons
    w_track: list, trakcing the trajectory of weights, list length is number of weight updats, and each list contains array with input dimension * num of output neurons
    y_thres: list, trakcing the trajectory of output, list length is number of weight updats, and each list contains array with 1 * num of output neurons
    obj: list, trakcing the trajectory of values of certain objective function, list length is number of weight updats, and each list contains array with 1 * num of output neurons

    """
    def __init__(self, eta = 0.1, n_epoch=10, ny=1, batch=1, tau=100.0, thres=0, p = 2,random_state = None, shuffle = True, nonlinear = None, obj_type = 'QBCM',decay = 0.0):
        self.eta = eta
        self.n_epoch = n_epoch
        self.ny = ny
        self.batch = batch
        self.tau = tau  
        self.thres = thres * np.ones(ny)
        self.p = p    
        self.shuffle = shuffle
        self.nonlinear = nonlinear
        self.obj_type = obj_type
        self.y_thres = []      # Storaged y for studying effect of threshold
        self.decay = decay

        if random_state:
            np.random.seed(random_state)
            
    def fit(self, X):
    
        # Weights initialized as normal distribution
        self.w_ = np.random.randn(X.shape[1],self.ny) # 2*1
        self.w_track = []
        self.obj = []
        
        # Use elementwise training 
        threshold = self.thres    # Auxillary veriables for calculating thresholds
        self.thres = []
        obj_x1 = np.zeros(self.ny)   # Two iterative terms in objective function
        obj_x2 = np.zeros(self.ny)   # Two iterative terms in objective function
        bcm_obj = np.zeros(self.ny)
        for _ in range(self.n_epoch):
            if self.shuffle:
                X = self._shuffle(X)
            for i, xi in enumerate(X):   # elementwise training for all samples
                y = np.zeros(self.ny)
                for j in range(self.ny):
                    y[j] = self._activation(np.dot(xi, self.w_[:, j]), nonlinear=self.nonlinear)
                    if self.obj_type == 'QBCM':
                        if self.nonlinear == 'Sigmoid':
                            self.w_[:, j][:, None] = self.w_[:, j][:, None]+ self.eta * xi[:, None] * dsigmoid(y[j]) * y[j] * (y[j] - threshold[j])- self.eta * self.decay * self.w_[:, j][:, None]
                        # self.w_[:, j][:, None] = self.w_[:, j][:, None]- self.eta * xi[:, None] * dsigmoid(y[j]) * y[j]
                        elif (self.nonlinear == 'Relu') | (self.nonlinear == None):
                            self.w_[:, j][:, None] = self.w_[:, j][:, None]+ self.eta * xi[:, None] * y[j] * (y[j] - threshold[j])- self.eta * self.decay * self.w_[:, j][:, None]
                        else:
                            print('Wrong nonlinearty')
                    elif self.obj_type == 'kurtosis':
                        if self.nonlinear == 'Sigmoid':
                            self.w_[:,j][:, None] = self.w_[:, j][:, None]+ 4 * self.eta * xi[:, None] * dsigmoid(y[j]) * y[j] * (y[j] ** 2 - threshold[j])- self.eta * self.decay * self.w_[:, j][:, None]
                            self.w_[:,j] = self.w_[:, j]/np.sqrt(np.sum((self.w_[:, j])**2))               # L2 norm of kurtosis learnin rule
                        elif (self.nonlinear == 'Relu')|(self.nonlinear == None):
                            self.w_[:,j][:, None] = self.w_[:, j][:, None]+ 4 * self.eta * xi[:, None] * y[j]*(y[j]**2 - threshold[j]) - self.eta * self.decay * self.w_[:, j][:, None]
                            self.w_[:,j] = self.w_[:, j]/np.sqrt(np.sum((self.w_[:, j])**2))     # L2 norm of kurtosis learnin rule
                        else:
                            print('Wrong nonlinearty')
                    else:
                        print('Wrong objective function')

                    threshold[j] = self._ema(x = threshold[j],y = y[j],power = self.p)       
                    bcm_obj[j] = obj(X,w = self.w_[:,j],obj_type = self.obj_type,nonlinear = self.nonlinear)
                w_tmp = np.concatenate(self.w_.T, axis=0)    # make 2*2 matrix into 1*4, preparing for weight tracking
                self.y_thres.append(y)
                self.thres.append(threshold.tolist())
                self.w_track.append(w_tmp.tolist())
                self.obj.append(bcm_obj.tolist())
        return self
    
    def _shuffle(self,X):
        r = np.random.permutation(len(X))
        return X[r]
    
    #Implementing exponential moving average
    def _ema(self, x, y, power = 2):
        # x is the iterative variable, y is the function being averaged
        h = np.exp(-1/self.tau)
        return x*h+(y**power)*(1-h)

    def _activation(self,y,nonlinear = None):
        if nonlinear == 'Sigmoid':
            y = sigmoid(y)
        elif nonlinear == 'Relu':
            y = (y>=0)* y
        return y

# Implement differentiation of sigmoid function
def dsigmoid(z):
    return sigmoid(z)*(1-(sigmoid(z)))

# Implement sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# objective function using all data
def obj(X, w, obj_type = 'QBCM', nonlinear = 'Sigmoid'):
    c = np.dot(X, w)
    if nonlinear == 'Sigmoid':
        c = sigmoid(c)
    elif nonlinear == 'Relu':
        c = (c>=0)* c

    obj = 0

    if obj_type == 'QBCM':
        obj1 = (c**3).mean(axis = 0)
        obj2 = (c**2).mean(axis = 0)
        obj = obj1/3 - obj2**2/4
    #    obj = - obj2/2
    elif obj_type == 'kurtosis':
        obj1 = (c**4).mean(axis = 0)
        obj2 = (c**2).mean(axis = 0)
        obj = obj1 - obj2**2*3
    elif obj_type == 'skewness':
        obj1 = (c**3).mean(axis = 0)
        obj2 = (c**2).mean(axis = 0)
        obj = np.divide(obj1,obj2**1.5)
    else:
        print('Wrong objective function')

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
def bcm_train(s_rt_wt,eta = 0.0001, n_epoch = 10, batch = 1, ny = 2,tau = 200, thres = 0, p = 2,random_state = None, shuffle = True, nonlinear = 'Relu', obj_type = 'QBCM',decay = 0.0):
    # Initialize a BCM class onject
    BCM_data = bcm(eta = eta,n_epoch = n_epoch,batch = batch,ny = ny,tau = tau, thres = thres, p = p,random_state=random_state , shuffle = shuffle, nonlinear = nonlinear, obj_type = obj_type, decay = decay)

    # Track traning time
    t0 = time()
    # Train teh model with data s_rt_wt
    BCM_data.fit(s_rt_wt)
    print("done in %0.3fs" % (time()-t0))

    # Convert list object to array
    plt_range = n_epoch * len(s_rt_wt)
    BCM_data_thres = np.vstack(BCM_data.thres)
    BCM_data_out = np.vstack(BCM_data.y_thres)
    BCM_data_w = np.vstack(BCM_data.w_track)
    BCM_data_obj = np.vstack(BCM_data.obj)

    n_row = 1
    n_column = ny

    BCM_data_titles= ["BCM unit %d" % i for i in range (ny)]

    # Plot y and threshold over iterations
    for i in range(ny):
        plt.subplot(n_row,n_column,i+1)
        plt.plot(BCM_data_out[:plt_range,i],'g-',label = 'output')
        plt.plot(BCM_data_thres[:plt_range,i],'r--',label = 'threshold')
        plt.title(BCM_data_titles[i])

    plt.legend(loc = 'upper right')

    plt.figure()

    # Plot weights over iterations 

    BCM_data_titles= ["BCM weights %d" % i for i in range (len(BCM_data_w.T))]

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.plot(BCM_data_w[:,0],'r',label = 'weight')
    ax1.set_title("BCM unit %d , weights %d" % (0,0))
    ax2.plot(BCM_data_w[:,2],'r',label = 'weight')
    ax2.set_title("BCM unit %d , weights %d" % (1,0))
    ax3.plot(BCM_data_w[:,1],'r',label = 'weight')
    ax3.set_title("BCM unit %d , weights %d" % (0,1))
    ax3.set_xlabel('# of iterations')
    ax4.plot(BCM_data_w[:,3],'r',label = 'weight')
    ax4.set_title("BCM unit %d , weights %d" % (1,1))
    ax4.set_xlabel('# of iterations')
    BCM_data_titles= ["BCM %d objective function " % i for i in range (len(BCM_data_w.T))]

    # plot the objective function
    plt.figure()
    for i in range(ny):
        plt.subplot(n_row,n_column,i+1)
        plt.plot(BCM_data_obj[:plt_range,i])
        plt.title(BCM_data_titles[i])
        plt.xlabel('# of iterations')
        #plt.axis([0,plt_range,-1000,1000])

# Plot the weights trajectory on top of objective function landscape

def bcm_obj(s_rt_wt , w_min , w_max , reso , para, obj_select = None, nonlinear_select = None, ori_w = 0):

    """
    Parameter: 
    s_rt_wt: input data, num of samples * dimension
    w_min: mininum range of objective function landscape
    w_max: maximun range of objective function landscape
    reso: resolution of weights grid
    para: parameters for training local learnin rule 
    ori_w: for laplace data, plot the original weights
    obj_select: str, type of objective function if specified, sweep across all combination if none
    Nolinear_select: str, type of nonlinearity if specified, sweep across all combination if none
    """

    w = np.linspace(w_min, w_max, reso)
    wx,wy = np.meshgrid(w, w)
    w = np.vstack((wx.ravel(), wy.ravel()))
    obj_choice = ['QBCM', 'kurtosis']
    nonlinear_choice = ['Relu', 'Sigmoid', None]

    p = para[0]   
    ny = para[1]
    tau = para[2]
    batch = para[3]
    n_epoch = para[4]
    decay = para[5]
    eta = para[6]
    # Plot a gallery of images
    if obj_select == None:
        n_row = len(obj_choice)
    else: 
        n_row = 1
        obj_index = obj_choice.index(obj_select)
        nonlinear_index = nonlinear_choice.index(nonlinear_select)

    if nonlinear_select == None:
        n_col = len(nonlinear_choice)
    else:
        n_col = 1

    fig, ax = plt.subplots(n_row, n_col, figsize=(12, 6), sharex=True, sharey=True)
    ori_w = ori_w * (w_max ** 0.5)

    for i in range(n_row):
        for j in range(n_col):
            if (n_row + n_col) > 2:
                obj_type = obj_choice[i]
                nonlinear = nonlinear_choice[j]
                para_index = i * n_row + j
            else:
                obj_type = obj_select
                nonlinear = nonlinear_select
                para_index = obj_index * len(obj_choice) + nonlinear_index

            obj_landscape = obj(s_rt_wt, w, obj_type=obj_type, nonlinear=nonlinear)
            title_set = [obj_type, nonlinear]

            nbins=20
            levels = np.percentile(obj_landscape, np.linspace(0,100,nbins))


            with sns.axes_style('white'):
                if (n_row + n_col) > 2:
                    g = ax[i, j]
                else:
                    g = ax

                c = g.contour(wx, wy, obj_landscape.reshape(wx.shape), levels=levels, zorder=-10, cmap=plt.cm.get_cmap('viridis'))
                g.plot(s_rt_wt[:, 0], s_rt_wt[:, 1], '.k', ms=4)
                g.set_aspect(1)

            plt.grid('on')
            plt.colorbar(c, ax=g)

            # Training with BCM local learning rule

            bcm_data = bcm(eta=eta[para_index], n_epoch=n_epoch[para_index], batch=batch, ny=ny, tau=tau, thres=0, p=p, random_state=None, shuffle=True, nonlinear=nonlinear, obj_type=obj_type, decay=decay[para_index])

            bcm_data.fit(s_rt_wt)
            bcm_data_w = np.vstack(bcm_data.w_track)

            g.plot([0, ori_w[0][0]], [0, ori_w[0][1]])
            g.plot([0, ori_w[1][0]], [0, ori_w[1][1]])
            g.plot(bcm_data_w[:, 0], bcm_data_w[:, 1], 'g')
            g.plot(bcm_data_w[-1, 0], bcm_data_w[-1, 1], 'y*', ms=15)
            g.plot(bcm_data_w[:, 2], bcm_data_w[:, 3], 'r')
            g.set_title(title_set)
            g.plot(bcm_data_w[-1, 2], bcm_data_w[-1, 3], 'y*', ms=15)