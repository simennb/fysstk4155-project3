import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import scikitplot as skplt
import functions as fun
import time
import os
import joblib
import pprint


########################
###### PARAMETERS ######
########################
parallel = -1  # number of jobs to run, -1 = all

data_path = '../datafiles/'
dataset_path = data_path + '/dataset/'
fig_path = '../figures/random_forest/'
pickle = data_path + 'random_forest/'

if not os.path.exists(fig_path):
    os.mkdir(fig_path)
if not os.path.exists(pickle):
    os.mkdir(pickle)

test_size = 0.2
n_bins = 200
n_pca = 35  # 0 for full data set

load_data = False  # set to True to bypass running the analysis and load the pickled file (if exists)
#load_data = True

# Set random seed for consistency
seed = 4155
np.random.seed(seed)  # unsure if matters here

# Create the parameter grid based on the results of random search
param_grid = {
    'n_estimators': [2, 5, 7, 10, 15, 25, 50, 100],
    'max_depth': [3, 5, 7, 9, 12],
    'min_samples_split': [2, 3, 4, 5, 7, 9],
    'min_samples_leaf': [5, 8, 10, 12],
    'max_features': [3, 5, 7, 10, 13]
}

# Other parameters to not loop over
param_init = {
    'max_leaf_nodes': None,
    'random_state': seed
}

########################
########################
########################
print('n_bins: %d , n_pca: %d\n' % (n_bins, n_pca))

# Alphabetical order
param_names = list(param_grid.keys())
param_names.sort()
n_params = [len(param_grid[key]) for key in param_names]
param_string = ''.join([str(i) for i in n_params])

# Load in data
if n_pca != 0:
    X = np.load(dataset_path + 'cat_dog_X_nbins%d_pca%d.npy' % (n_bins, n_pca))
else:
    X = np.load(dataset_path + 'cat_dog_X_nbins%d.npy' % n_bins)
y = np.load(dataset_path + 'cat_dog_y_nbins%d.npy' % n_bins)
y = y.ravel()

# Split into train and test data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

if not load_data:
    # Create model
    rf = RandomForestClassifier(**param_init)
    # Initialize grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=parallel, verbose=1, iid='deprecated',
                               return_train_score=True, scoring='accuracy')

    t_start = time.time()
    # Fit the grid search to the data
    grid_search.fit(X, y)
    t_end = time.time()

    t_tot = t_end - t_start
    print('Total time elapsed: %.2f s' % t_tot)

    # Dump data or load
    joblib.dump(grid_search, pickle + 'grid_search_%s_seed%d_nbins%d_pca%d.pkl' % (param_string, seed, n_bins, n_pca))
else:
    grid_search = joblib.load(pickle + 'grid_search_%s_seed%d_nbins%d_pca%d.pkl' % (param_string, seed, n_bins, n_pca))

#y_fit = grid_search.predict(X_train)
#y_pred = grid_search.predict(X_test)

best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
best_index = grid_search.best_index_

i_, j_, k_, l_, m_ = np.unravel_index(best_index, n_params)  # indices of best params

# Results matrix for all parameter combinations
cv_results = grid_search.cv_results_
test_score = cv_results['mean_test_score']
test_score = test_score.reshape(n_params)

print('Mean fit time = %.2e s' % np.mean(cv_results['mean_fit_time']))

print('### GRID SEARCH RESULTS ###')
print('Best params:')
pprint.pprint(grid_search.best_params_)
print('\nTrain accuracy = %.3f' % np.max(cv_results['mean_train_score']))
print('Test accuracy = %.3f' % grid_search.best_score_)

# Accuracy using our splitted dataset and not CV results
best_estimator.fit(X_train, y_train)
y_fit = best_estimator.predict(X_train)
y_pred = best_estimator.predict(X_test)
y_proba = best_estimator.predict_proba(X_test)

print('\n### NORMAL DATA TRAIN TEST SPLIT ###')  # TODO: do
print('Train accuracy = %.3f' % best_estimator.score(X_train, y_train))
print('Test accuracy = %.3f' % best_estimator.score(X_test, y_test))

# Creating y-arrays with labels due to oddities related to scikitplot version that was installed
y_1 = ['cat' if i == 0 else 'dog' for i in y_test]
y_2 = ['cat' if i == 0 else 'dog' for i in y_pred]

########################################################################################################
# Plotting
save = 'nbins%d_pca%d_seed%d_ts%.2f' % (n_bins, n_pca, seed, test_size)
fs = 12

# Confusion matrix
skplt.metrics.plot_confusion_matrix(y_1, y_2, normalize=True, text_fontsize=fs, title_fontsize=fs+1)
plt.savefig(fig_path + 'confusion_matrix_%s.png' % save)

# ROC curve
skplt.metrics.plot_roc(y_1, y_proba, text_fontsize=fs, title_fontsize=fs+1)
plt.savefig(fig_path + 'roc_%s.png' % save)

# Plot hyperparameter heatmaps in an automatic fashion
tick_type = ['int', 'int', 'int', 'int', 'int']
for i in range(len(n_params)):
    for j in range(i + 1, len(n_params)):
        # Dynamically change which dimension to iterate over
        dims = [i_, j_, k_, l_, m_]
        dims[i] = slice(None)
        dims[j] = slice(None)
        dims = tuple(dims)

        if n_params[i] > 1 and n_params[j] > 1:
            fun.plot_heatmap(param_grid[param_names[i]], param_grid[param_names[j]], test_score[dims].T,
                             param_names[i].replace('_', ' '), param_names[j].replace('_', ' '), 'accuracy',
                             'Test accuracy', save + '_%s_%s_%s' % ('accuracy', param_names[i], param_names[j]),
                             fig_path + 'heatmaps/', xt=tick_type[i], yt=tick_type[j])

plt.show()
