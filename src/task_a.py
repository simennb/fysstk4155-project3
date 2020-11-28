import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import scikitplot as skplt
import functions as fun
import time
import os
import joblib

data_path = '../datafiles/'
fig_path = '../figures/random_forest/'
pickle = data_path + 'random_forest/'
# TODO: Fix pickle with n_bins and folder creation!

if not os.path.exists(fig_path):
    os.mkdir(fig_path)
if not os.path.exists(pickle):
    os.mkdir(pickle)

test_size = 0.2
n_bins = 200
n_pca = 35

#load_data = False
load_data = True

# Default
'''
n_estimators = 100
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
max_features = 100#'auto'
max_leaf_nodes = None
'''

'''
n_estimators = [10*i for i in range(1, 21)]
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
max_features = 100#'auto'
max_leaf_nodes = None
'''

# Create the parameter grid based on the results of random search
# TODO: Find and decide on parameter sets to use
param_grid = {
    'n_estimators': [2, 5, 7, 10, 15, 25, 50, 100],
    'max_depth': [80, 90, 100, 110, 120, 130],
    'min_samples_split': [3, 4, 5],
    'min_samples_leaf': [8, 10, 12],
    'max_features': [7, 10, 13, 16],
    'max_leaf_nodes': [None]
}

#n_params = [len(param_grid['n_estimators']), len(param_grid['max_depth']), len(param_grid['min_samples_split']),
 #           len(param_grid['min_samples_leaf']), len(param_grid['max_features']), len(param_grid['max_leaf_nodes'])]

# Alphabetical order
n_params = [len(param_grid['max_depth']), len(param_grid['max_features']), len(param_grid['max_leaf_nodes']),
            len(param_grid['min_samples_leaf']), len(param_grid['min_samples_split']), len(param_grid['n_estimators'])]
param_string = ''.join([str(i) for i in n_params])

# Set random seed for consistency
seed = 4155  # 42
seed = 42
np.random.seed(4155)

# Load in data
if n_pca != 0:
    X = np.load(data_path + 'dataset/cat_dog_X_nbins%d_pca%d.npy' % (n_bins, n_pca))
else:
    X = np.load(data_path + 'dataset/cat_dog_X_nbins%d.npy' % n_bins)
y = np.load(data_path + 'dataset/cat_dog_y_nbins%d.npy' % n_bins)
y = y.ravel()

# Split into train and test data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
'''
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf, max_features=max_features,
                             max_leaf_nodes=max_leaf_nodes, n_jobs=-1)

t_start = time.time()
clf.fit(X_train, y_train)
'''

if not load_data:
    # Create model
    rf = RandomForestClassifier(random_state=seed)
    # Instantiate the grid search model TODO fix
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=1, iid='deprecated')

    t_start = time.time()
    # Fit the grid search to the data TODO fix
    grid_search.fit(X, y)
    t_end = time.time()

    t_tot = t_end - t_start
    print('Total time elapsed: %.2f s' % t_tot)

    # Dump data or load
    # TODO: add seed and nbins dependency on pickle names
    joblib.dump(grid_search, pickle + 'grid_search_%s.pkl' % param_string)
else:
    grid_search = joblib.load(pickle + 'grid_search_%s.pkl' % param_string)

y_fit = grid_search.predict(X_train)
y_pred = grid_search.predict(X_test)

print(grid_search.best_params_)
best_params = grid_search.best_params_

best_grid = grid_search.best_estimator_
print('aaaa', grid_search.best_score_)

best_index = grid_search.best_index_
print(best_index)

# Results matrix for all parameter combinations
cv_results = grid_search.cv_results_
test_score = cv_results['mean_test_score']
test_score = test_score.reshape(n_params)
print(test_score.shape)
i_, j_, k_, l_, m_, n_ = np.unravel_index(best_index, n_params)

'''
grid_accuracy = evaluate(best_grid, test_features, test_labels)
Model Performance
Average Error: 3.6561 degrees.
Accuracy = 93.83%.
print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
Improvement of 0.50%.
'''


#y_fit = clf.predict(X_train)
#y_pred = clf.predict(X_test)

'''
print('Train:')
for i in range(len(y_train)):
    print(y_train[i], y_fit[i])
print('\nTest:')
for i in range(len(y_test)):
    print(y_test[i], y_pred[i])
'''

#print('Train accuracy = %.3f' % clf.score(X_train, y_train))
#print('Test accuracy = %.3f' % clf.score(X_test, y_test))


# Creating y-arrays with labels due to oddities related to scikitplot version that was installed
y_1 = ['cat' if i == 0 else 'dog' for i in y_test]
y_2 = ['cat' if i == 0 else 'dog' for i in y_pred]

y_pred2 = np.zeros((y_pred.shape[0], 2))
y_pred2[y_pred == 0, 0] = 1
y_pred2[y_pred == 1, 1] = 1  # could simplify

y_fit2 = np.zeros((y_fit.shape[0], 2))
y_fit2[y_fit == 0, 0] = 1
y_fit2[y_fit == 1, 1] = 1  # could simplify


########################################################################################################
# Plotting
save = 'nbins%d_pca%d_seed%d_ts%.2f' % (n_bins, n_pca, seed, test_size)
fs = 12

# Confusion matrix
skplt.metrics.plot_confusion_matrix(y_1, y_2, normalize=True, text_fontsize=fs, title_fontsize=fs+1)
plt.savefig(fig_path + 'confusion_matrix_%s.png' % save)

skplt.metrics.plot_roc(y_test, y_pred2, text_fontsize=fs, title_fontsize=fs+1)
plt.savefig(fig_path + 'roc_%s.png' % save)
#scikitplot.metrics.plot_roc(y_true, y_probas, title='ROC Curves', plot_micro=True, plot_macro=True, classes_to_plot=None, ax=None, figsize=None, cmap='nipy_spectral', title_fontsize='large', text_fontsize='medium')

skplt.metrics.plot_cumulative_gain(y_test, y_pred2, title='Cumulative Gains Curve - Test',
                                   text_fontsize=fs, title_fontsize=fs+1)
plt.savefig(fig_path + 'cumulative_gain_test_%s.png' % save)

skplt.metrics.plot_cumulative_gain(y_train, y_fit2, title='Cumulative Gains Curve - Train',
                                   text_fontsize=fs, title_fontsize=fs+1)
plt.savefig(fig_path + 'cumulative_gain_train_%s.png' % save)


# Plot heatmaps
# TODO: This could be automated
param_names = ['max features', 'max features', 'max leaf nodes',
               'min samples leaf', 'min samples split', 'n estimators']
pnamestr = [p.replace(' ', '_') for p in param_names]

for i in range(len(n_params)):
    for j in range(i + 1, len(n_params)):
        # Dynamically change which dimension to iterate over
        dims = [i_, j_, k_, l_, m_, n_]
        dims[i] = slice(None)
        dims[j] = slice(None)
        dims = tuple(dims)

        if n_params[i] > 1 and n_params[j] > 1:
            print(i, j)
            fun.plot_heatmap(param_grid[pnamestr[i]], param_grid[pnamestr[j]], test_score[dims].T,
                                        param_names[i], param_names[j], 'accuracy', 'Test accuracy',
                                        save + '_%s_%s_%s' % ('accuracy', pnamestr[i], pnamestr[j]),
                                        fig_path + 'heatmaps/', xt='int', yt='int')

'''
# Max depth vs max features
fun.plot_heatmap(param_grid['max_depth'], param_grid['max_features'], test_score[:, :, k_, l_, m_, n_].T,
                 'max depth', 'max features', 'accuracy', 'Test accuracy',
                 save+'_%s_%s' % ('accuracy', 'max_depth_max_features'), fig_path + 'heatmaps/', xt='int', yt='int')

# Max depth vs n estimators
fun.plot_heatmap(param_grid['max_depth'], param_grid['n_estimators'], test_score[:, j_, k_, l_, m_, :].T,
                 'max depth', 'N estimators', 'accuracy', 'Test accuracy',
                 save+'_%s_%s' % ('accuracy', 'max_depth_n_estimators'), fig_path + 'heatmaps/', xt='int', yt='int')

# Max features vs n estimators
fun.plot_heatmap(param_grid['max_features'], param_grid['n_estimators'], test_score[i_, :, k_, l_, m_, :].T,
                 'max features', 'N estimators', 'accuracy', 'Test accuracy',
                 save+'_%s_%s' % ('accuracy', 'max_features_n_estimators'), fig_path + 'heatmaps/', xt='int', yt='int')
'''

plt.show()
