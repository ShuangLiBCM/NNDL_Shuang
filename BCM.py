# Define BCM class by quadratic local learning rule
# Relu activation function
# Update weight per sample
# Enable batch normalization

class BCM:
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
    def __init__(self, eta = 0.1,n_epoch = 10,ny = 1,batch = 10, tau = 100,thres = 0, p = 2,random_state = None, shuffle = True,nonlinear = None):
        self.eta = eta
        self.n_epoch = n_epoch
        self.ny = ny
        self.tau = tau  # Time constant for calculating thresholds
        self.thres = [thres*np.ones(ny)]
        self.p = p    # int, power for threshold computation
        self.y_thres = []      # Storaged y for studying effect of threshold
        self.shuffle = shuffle
        self.nonlinear = nonlinear
        self.batch = batch

        if random_state:
            seed(random_state)

    def fit(self,X):
        """fitting training data
        Parameter:
        X: {array-like}, shape = [n_samples,n_features]
        Returns: self:object
        ny: value, number of output neurons
        """
        # Weights initialized as normal distribution
        self.w_ = np.random.randn(X.shape[1],self.ny) # 2*1
        self.w_track = []

        # Use elementwise training
        threshold = np.zeros(2)
        for _ in range(self.n_epoch):
            if self.shuffle:
                X = self._shuffle(X)
            if self.batch:
                X_batch = []
                for k in range(0,X.shape[0],self.batch):
                    X_batch.append(np.mean(X[k:k+self.batch,:],axis = 0))
                X_batch = np.vstack(X_batch)
            for i, xi in enumerate(X_batch):   # elementwise training for all samples
                theta = np.zeros(2)
                y = np.zeros(2)
                for j in range(self.ny):
                    y[j] = np.dot(xi,self.w_[:,j])
                    if self.nonlinear == 'Sigmoid':
                        y[j] = self._sigmoid (y[j])
                    elif self.nonlinear == 'Relu':
                        y[j] = (1-(y[j]<0))* y[j]
                    self.w_[:,j][:,None] = self.w_[:,j][:,None]+ self.eta * xi[:,None]* y[j]*(y[j] - threshold[j])
                    h = np.exp(-1/self.tau)
                    y_power = y[j]**2
                    theta[j] = threshold[j]*h+y_power*(1-h)  # Good way to implement exponential moving average
                w_tmp = np.concatenate(self.w_.T, axis=0)    # make 2*2 matrix into 1*4, preparing for weight tracking
                self.y_thres.append(y)
                self.thres.append(theta)
                self.w_track.append(w_tmp.tolist())
                threshold = theta
        return self

    def _shuffle(self,X):
        r = np.random.permutation(len(X))
        return X[r]

    def _sigmoid(self,z):
        return 1/(1+np.exp(-z))