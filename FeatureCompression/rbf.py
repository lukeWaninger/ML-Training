from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation
   
    Parameters
    -------------
    X: {NumPy ndarray}, shape = {n_samples, n_features}
    gamma: float : Tuning parameter of the RBF kernel
    n_components: int : Number of PCs to return

    Returns
    -------------
    x_pc: {NumPy ndarray}, shape = [n_samples, k_features : projected dataset
    lampdas: list : eigenvalues
    """

    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset
    sq_distances = pdist(X, 'sqeuclidean')

    # convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_distances)

    # compute the symmetric kernel  matrix
    K = exp(-gamma * mat_sq_dists)

    # center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # obtain the eigenpairs from the centered kernel matrix
    # numpy eigh returns them in sorted order (which is needed to 
    # determine the principle components
    eigvals, eigvecs = eigh(K)

    # collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
    
    # collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]

    return alphas, lambdas