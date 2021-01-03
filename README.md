Evaluation of the ideal binning techniques for the piecewise constant approximation of the normalized Unexplained Variance (nUV) measure
===

Contents
---
The package contains the following files.

1) General files:
    * ```requirements.txt```: the package specifications to reproduce the results.
    * ```README.md```: this file.
    * ```LICENSE```: the license of the code.
2) Python codes:
    * ```binning.py```: implementation of the binning techniques.
    * ```config.py```: configuration parameters.
    * ```data_generation.py```: all the functions related to the random generation or computation of quantities appearing in the paper.
    * ```evaluation.py```: the script generating plots and tables.
    * ```nuv.py```: the implementation of the normalized Unexplained Variance dissimilarity measure and the exact approximations of the various expected values.
    * ```simulation.py```: the script implementing the simulations used in the Tests and Results section of the paper.
    * ```feature_selection.py```: the codes implementing the feature selection experiment.
    * ```utils/*```: some further utility functions.
3) Notebooks:
    * ```evaluation_of_simulations.ipynb```: evaluation of the pattern recognition simulations
    * ```evaluation_of_feature_selection.ipynb```: evaluation of feature selection.
4) Data files:
    * ```results_general.csv```: all the simulation results for general distortions.
    * ```results_spherical.csv```: all the simulation results for spherical distortions. 
    * ```feature_selection_results.csv```: r2 scores of the feature selection experiment.
    * ```feature_selection_rankings.csv```: rankings in the feature selection experiment.
    * ```feature_selection_runtimes.csv```: runtimes of the feature selection experiment.
5) Plots:
    * ```auc_general.pdf```: AUC scores with general distortions.
    * ```auc_spherical.pdf```: AUC scores with spherical distortions.
    * ```fit_general.pdf```: alignment of theoretical results and measurements for general distortions.
    * ```fit_spherical.pdf```: alignment of theoretical results and measurements for spherical distortions.
    * ```fs_results.pdf```: feature selection results.

Reproducing the results
---

The following steps can be used to reproduce the results of the study in either Windows/Linux or Mac environments.

It is assumed that ```git``` and ```Anaconda``` are installed and properly configured.

1) Clone the repository
```bash
> git clone https://github.com/gykovacs/ideal_binning_nuv.git
```
2) Enter the repository
```bash
> cd ideal_binning_nuv
```
3) Create a conda Python 3.7 conda environment
```bash
> conda create -n nuv-test python==3.7
```
4) Activate the environment
```bash
> conda activate nuv-test
```
5) Install the packages in ```requirements.txt```
```bash
> pip install -r requirements.txt
```
6) Edit the ```config.py``` file to set the path of the working directory, one can also modify the parameters of the simulation.
7) Execute the ```simulation.py``` file to generate the CSV files of results in the working directory.
```bash
> python simulation.py
```
8) Execute the ```feature_selection.py``` file to generate the plots and tables appearing in the paper
```
> python feature_selection.py
```
9) Run the notebooks ```evaluation_of_simulations.ipynb``` and ```evaluation_of_feature_selection.ipynb``` to generate the plots and statistical test results.

Remarks
---

1) The execution time with the default settings in ```config.py``` is about 10 hours on an average computer.
2) The codes might run with different package configurations than the ones specified in ```requirements.txt```, however, due to possible changes in the pseudo-number generators, other package configurations might give slightly different results.
