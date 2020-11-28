import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

data_path = '../datafiles/dataset/'

n_bins = 200

# Load in data set
X = np.load(data_path + 'cat_dog_X_nbins%d.npy' % n_bins)

#pca = PCA(n_components='mle')
#X_new = pca.fit_transform(X)
#print(X_new.shape)
#print("Eigenvector of largest eigenvalue")
#print(pca.components_.T[:, 0])
#print(pca.explained_variance_ratio_.shape)

pca = PCA(n_components=0.95)  # 95% of variance
X_reduced = pca.fit_transform(X)
print(X_reduced.shape)

# TODO: maybe consider plotting stuff?

X = np.save(data_path + 'pca/cat_dog_X_nbins%d_pca%d.npy' % (n_bins, X_reduced.shape[1]), X_reduced)

