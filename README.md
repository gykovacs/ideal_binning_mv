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
    * ```utils/*```: some further utility functions.
3) Data files:
    * ```results_general.csv```: all the simulation results for general distortions.
    * ```results_spherical.csv```: all the simulation results for spherical distortions. 

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
8) Execute the ```evaluation.py``` file to generate the plots and tables appearing in the paper
```
> python evaluation.py
```

Remarks
---

1) The execution time with the default settings in ```config.py``` is about 10 hours on an average computer.
2) The codes might run with different package configurations than the ones specified in ```requirements.txt```, however, due to possible changes in the pseudo-number generators, other package configurations might give slightly different results.

Raw simulation results
---

The raw results of the simulations are available for download at the following links:

1) general distortions: https://drive.google.com/file/d/1Y8IOEfRe_WW9NwSUxQjMoBQpRwnmx2gN/view?usp=sharing
2) spherical distortions: https://drive.google.com/file/d/1VQd3iN6uMyERHXIbrf21epBs7i59T5PF/view?usp=sharing