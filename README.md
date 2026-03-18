# Code for revealing_entanglement_through_local_features_of_phase-space_distributions
This is the repository for all the code used in the article _[Revealing entanglement through local features of phase-space distributions](https://arxiv.org/abs/2602.21688)_ by Elena Callus, Martin Gärttner & Tobias Haas, arXiv:2602.21688.

All the code is written in Python version 2.7.14, except for one file that uses Mathematica.

The code is organised as follows:

### NOON states
* noon_optimal.nb - a Mathematica notebook to locate the coordinates for minima of the Husimi-based criterion for various N
* noon_data.py - generates the data used for plotting the Husimi-based criterion for various N, as well as the performance of the criterion for a range of $\sigma$ and loss parameter $\tau$
* noon_plots.py - generates plots using the previously generated data

### Entangled cat states
* entangled_cat_states.py - generates the data (which is also saved in .csv files) and the plots for the two-mode entangled cat states

### Random number states
* random_states_data-py - randomly samples the value of the $\sigma$-parametrized criterion for various Hilbert space dimensions at two different phase-space coordinates
* random_states_plots.py - generates plots using the previously generated data
