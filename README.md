# FYS-STK4155: Project 3
A look at audio classification in fourier space. The data set used is the set of cat and dog sounds from [here.](https://www.kaggle.com/mmoreaux/audio-cats-and-dogs)
The classifiers used in this project is a simple feed-forward neural network, random forests, and XGBoost.

## Report
The report for this project can be found [here.](https://github.com/simennb/fysstk4155-project3/blob/main/report/project3.pdf)

## Usage
### Creating data set:
Running the file `create_dataset.py` allows you to create and plot the data set with the specified number of frequency bins. Requires updating the path to where the data set is stored, as data set is not included in the repository.

The file `pca_dataset.py` can be used to perform principal component analysis on the data set to create a dimensionally reduced version of the data set.

### Analysis:
The file `task_a.py` performs the analysis using random forests.

The file `task_b.py` performs the analysis using feed-forward neural nets.

The file `task_c.py` performs the analysis using XGBoost.

## Benchmarks
A set of benchmark results can be found in the benchmarks folder, with one file corresponding to each of the main programs of the project.

## License
[MIT](https://choosealicense.com/licenses/mit/)
