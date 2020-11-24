import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data_path = '../datafiles/'
fig_path = '../figures/'

test_size = 0.2
n_bins = 200

# Load in data
X = np.load(data_path + 'cat_dog_X_nbins%d.npy' % n_bins)
y = np.load(data_path + 'cat_dog_y_nbins%d.npy' % n_bins)
y = y.ravel()

# Split into train and test data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

clf = RandomForestClassifier()

t_start = time.time()
clf.fit(X_train, y_train)
t_end = time.time()

y_fit = clf.predict(X_train)
y_pred = clf.predict(X_test)

print('Train:')
for i in range(len(y_train)):
    print(y_train[i], y_fit[i])
print('\nTest:')
for i in range(len(y_test)):
    print(y_test[i], y_pred[i])


print('Train ', clf.score(X_train, y_train))
print('Test ', clf.score(X_test, y_test))

t_tot = t_end - t_start
print('Total time elapsed: %.2f s' % t_tot)