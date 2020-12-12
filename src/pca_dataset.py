import numpy as np
from sklearn.decomposition import PCA


data_path = '../datafiles/dataset/'  # bootstrap/'
n_bins = 200

# Load in data set
X = np.load(data_path + 'cat_dog_X_nbins%d.npy' % n_bins)

# Perform PCA
pca = PCA(n_components=0.95)  # 95% of variance
X_reduced = pca.fit_transform(X)
print(X_reduced.shape)

# Save new data set to file
X = np.save(data_path + 'cat_dog_X_nbins%d_pca%d.npy' % (n_bins, X_reduced.shape[1]), X_reduced)
