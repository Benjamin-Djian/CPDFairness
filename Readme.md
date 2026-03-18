# CPDFairness

## Description

The goal of this projet is to use Computational Profile Distance (CPD) in the context of fairness of deep neural
networks (DNN).

The project is initially build to measure difference in CPD distribution between two demographic groups in the context
of binary classification.
More detailed articles can be found [here](https://onlinelibrary.wiley.com/doi/10.1111/coin.70124)
and [here](https://www.sciencedirect.com/science/article/pii/S0950584925002150).

The code creates a binary classificator, preprocessed data and train the predictor on it.
Then, activation of neurons of last hidden are collected to build histograms.
Finally, CPD of inputs are calculated using these histograms.
Various graphs can be plotted to represent CPD distributions, or histograms.

## Technologies used

The project uses mainly [numpy](https://numpy.org/) and [pandas](https://pandas.pydata.org/) for data manipulation,
and [Pytorch](https://pytorch.org/) for neural networks.

For some experiments, we
used [Correlation Remover](https://fairlearn.org/main/api_reference/generated/fairlearn.preprocessing.CorrelationRemover.html)
from Fairlearn
and [Disparate Impact Remover](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.DisparateImpactRemover.html)
from AIF360.

## Requirements and Installation instructions

Soon available

# Roadmap

- [ ] Personalize preprocessing in config file
- [ ] CR and DIR
- [ ] Post experiment
- [ ] Update main.py

