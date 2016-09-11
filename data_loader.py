import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
import pandas as pd

def load_Iris(whiten = True):
	
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
	df.tail()   # Visualize in the table last rows of the dataframe

	y = df.iloc[:, 4].values  # pandas dataframe, select data by location index
	y = np.where(y == 'Iris-setosa',1,-1)
	s_rt = df.iloc[:,[0,2]].values
	# Remove the mean
	s_rt = s_rt - s_rt.mean(axis = 0)

	if whiten:
		ZCAMatrix = zca_whitening_matrix(s_rt.T)
		s_rt_wt = np.dot(s_rt,ZCAMatrix)
	else: 
		s_rt_wt = s_rt

	# Generate 2d satter plot of the whitened data
	df = pd.DataFrame({'x':s_rt_wt[:,0],'y':s_rt_wt[:,1]})
	g = sns.jointplot(x="x", y="y", data=df)
	g.plot_joint(plt.scatter, c="gray", s=10, linewidth=.1, marker=".")
	g.ax_joint.collections[0].set_alpha(0)
	g.set_axis_labels("Dimension 1", "Dimension 2")

	plt.scatter(s_rt_wt[:50,0],s_rt_wt[:50,1],color = 'red',marker = 'o',label = 'setosa')
	plt.scatter(s_rt_wt[50:150,0],s_rt_wt[50:150,1],color = 'blue', marker = '*', label = 'versicolor')
	plt.xlabel('petal length')
	plt.ylabel('sepal length')
	plt.legend(loc = 'upper left')

	return s_rt_wt

def load_laplace(loc = 0, scale = 1, sample_size = 1000,dimension = 2,skew = False, whiten = True):
		# Sample from the above distribution
	s = np.random.laplace(loc,scale,[sample_size,dimension])  # 5000*2, 0 mean, unit variance

	# make data skewed with a half-squaring
	if skew:
	    s[:,1] = stats.skewnorm.rvs(14., size=len(s))

	# Conduct rotation and mixture
	# Generate rotation matrix
	#if rotation
	#	theta = np.pi/4      # 45 degree rotation
	#A = np.array(((mt.cos(theta),-mt.sin(theta)),(mt.sin(theta),mt.cos(theta))))

	A = np.random.randn(dimension,dimension)
	#A = np.dot(A, A.T)

	s_rt = np.dot(s,A)

	if whiten:
		ZCAMatrix = zca_whitening_matrix(s_rt.T)
		s_rt_wt = np.dot(s_rt,ZCAMatrix)
	else: 
		s_rt_wt = s_rt

	# plot original distribution
	df = pd.DataFrame({'x':s[:,0],'y':s[:,1]})
	g = sns.jointplot(x="x", y="y", data=df)
	g.plot_joint(plt.scatter, c="gray", s=10, linewidth=.1, marker=".")
	g.ax_joint.collections[0].set_alpha(0)
	g.set_axis_labels("Dimension 1", "Dimension 2")

	# plot mixed distribution
	df = pd.DataFrame({'x':s_rt[:,0],'y':s_rt[:,1]})
	g = sns.jointplot(x="x", y="y", data=df)
	g.plot_joint(plt.scatter, c="gray", s=10, linewidth=.1, marker=".")
	g.ax_joint.collections[0].set_alpha(0)
	g.set_axis_labels("Dimension 1", "Dimension 2")

	return s_rt_wt



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