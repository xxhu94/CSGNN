import pandas as pd
import numpy as np
import torch


def cost_table_calc(cost_matrix):
    """Creates a table of values from the cost matrix.
    Write the matrix form cost matrix in the form of coordinates + cost value

    Parameters
    ----------
    cost_matrix : array-like of shape = [n_classes, n_classes]

    Returns
    -------
    df : dataframe of shape = [n_classes * n_classes, 3]

    """
    table = np.empty((0, 3))

    for (x, y), value in np.ndenumerate(cost_matrix.numpy()):
        # table = np.vstack((table, np.array([x + 1, y + 1, value])))
        table = np.vstack((table, np.array([x, y, value])))

    return pd.DataFrame(table, columns=['row', 'column', 'cost'])

labels=torch.tensor([1,1,0,0,1])
# labels=torch.rand([5,1])
feat=torch.rand([5,3])
cost_matrix=torch.rand([2,2])
cost_table= cost_table_calc(cost_matrix)
num_classes=len(torch.unique(labels, return_counts=True)[0])

# class_count=torch.unique(labels, return_counts=True)[1]

# H = pd.DataFrame(np.zeros((len(class_count),len(class_count))))
# H = pd.DataFrame(np.zeros((num_classes,num_classes)))
#
# h = (class_count / torch.sum(class_count).item()).numpy()
#
# for i in range(num_classes):
#     for j in range(num_classes):
#         if i == j:
#             H.iloc[i, j] = h[i]
#         else:
#             H.iloc[i, j] = max(h[i], h[j])
# H = torch.tensor(H.values)
# print(H)



# from __future__ import print_function, division
import matplotlib.pyplot as plt
from utils import calculate_covariance_matrix, normalize, standardize


class MultiClassLDA():
    """Enables dimensionality reduction for multiple
    class distributions. It transforms the features space into a space where
    the between class scatter is maximized and the within class scatter is
    minimized.
    Parameters:
    -----------
    solver: str
        If 'svd' we use the pseudo-inverse to calculate the inverse of matrices
        when doing the transformation.
    """
    def __init__(self, solver="svd"):
        self.solver = solver

    def _calculate_scatter_matrices(self, X, y):
        n_features = np.shape(X)[1]
        labels = np.unique(y)

        # Within class scatter matrix:
        # SW = sum{ (X_for_class - mean_of_X_for_class)^2 }
        #   <=> (n_samples_X_for_class - 1) * covar(X_for_class)
        # SW = np.empty((n_features, n_features))
        intra_dis={}
        for label in labels:
            _X = X[y == label]
            # SW += (len(_X) - 1) * calculate_covariance_matrix(_X)
            # SW += (len(_X) - 1) * np.trace(calculate_covariance_matrix(_X))
            intra_dis[label]=  np.trace(calculate_covariance_matrix(_X))
        SW=np.array(list(intra_dis.values()))
        SW=np.tile(SW, (len(SW), 1))
        # Between class scatter:
        # SB = sum{ n_samples_for_class * (mean_for_class - total_mean)^2 }
        total_mean = np.mean(X, axis=0)
        inter_dis={}
        class_mean={}
        # SB = np.empty((n_features, n_features))
        SB=np.zeros((len(labels),len(labels)))
        for label_i in labels:
            _X_i = X[y == label_i]
            _mean_i = np.mean(_X_i, axis=0)
            for label_j in labels:
                _X_j = X[y == label_j]
                _mean_j = np.mean(_X_j, axis=0)
                SB[label_i][label_j]=np.linalg.norm(_mean_i-_mean_j)
            # _X = X[y == label]
            # _mean = np.mean(_X, axis=0)
            # class_mean[label]=_mean
            # inter_dis[label]=  (_mean - total_mean).dot((_mean - total_mean).T)
        row, col = np.diag_indices_from(SB)
        SB[row, col] = 1
        # SB=np.array(list(inter_dis.values()))
        S=SW.T*(1/SB)

        # S=np.transpose([SW]).dot(1 / SB.reshape(-1, len(SB)))

        return S

    def transform(self, X, y, n_components):
        SW, SB = self._calculate_scatter_matrices(X, y)

        # Determine SW^-1 * SB by calculating inverse of SW
        A = np.linalg.inv(SW).dot(SB)

        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = eigenvectors[:, idx][:, :n_components]

        # Project the data onto eigenvectors
        X_transformed = X.dot(eigenvectors)

        return X_transformed


    def plot_in_2d(self, X, y, title=None):
        """ Plot the dataset X and the corresponding labels y in 2D using the LDA
        transformation."""
        X_transformed = self.transform(X, y, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y)
        if title: plt.title(title)
        plt.show()

A=MultiClassLDA()
A.plot_in_2d(feat.numpy(),labels.numpy())
