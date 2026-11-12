# Statistical Machine Learning

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![TensorFlow Probability](https://img.shields.io/badge/TF_Probability-0.23-FF6F00.svg?style=flat)](https://www.tensorflow.org/probability)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An in-depth curriculum covering foundational statistical machine learning, Bayesian inference, probabilistic graphical models, and sampling methods. Each laboratory balances rigorous mathematical derivations with practical implementation in Python, NumPy, SciPy, and TensorFlow Probability.

Managed with [**uv**](https://github.com/astral-sh/uv), the blazingly fast Python package and environment manager.

---

## Overview

| Lab | Topic | Mathematical & Algorithmic Foundations | Core Tools |
| :---: | :--- | :--- | :--- |
| [**01**](./lab1/lab1.ipynb) | **Monte Carlo Foundations** | Monte Carlo Integration, Hypersphere Volumes, Uniform Sampling in $d$-Ball, Von Neumann Coin Extractor | NumPy, Matplotlib |
| [**02**](./lab2/lab2.ipynb) | **Linear Gaussian Models & State Estimation** | Multivariate Gaussian Distributions, Radar Echo Localization, Bayes' Rule, Kalman-style Sequential Updating | NumPy, SciPy |
| [**03**](./lab3/lab3.ipynb) | **Linear & Bayesian Regression** | Ordinary Least Squares (OLS), Conjugate Gaussian Priors, Posterior Weight Distributions, MAP Estimation | NumPy, SciPy |
| [**04**](./lab4/lab4.ipynb) | **Gaussian Process Regression** | Covariance Kernels (RBF, Matérn), GP Priors/Posteriors, Marginal Likelihood, Bayesian Optimization | NumPy, SciPy |
| [**05**](./lab5/lab5.ipynb) | **Multinomial Logistic Regression** | Softmax Formulation, Cross-Entropy Loss, Vectorized Gradient Descent from Scratch on MNIST | NumPy, MNIST |
| [**06**](./lab6/lab6.ipynb) | **Probabilistic Programming & Importance Sampling** | TensorFlow Probability (TFP), Joint Distributions, Importance Sampling, Normalizing Constant Estimation | TensorFlow, TFP |
| [**07**](./lab7/lab7.ipynb) | **MCMC & Hierarchical Bayesian Models** | Markov Chain Monte Carlo, Metropolis-Hastings, Hamiltonian Monte Carlo (HMC), Change-Point Models | TensorFlow, TFP, Seaborn |

---

## Detailed Lab Breakdown

<details open>
<summary><b>Lab 1: Monte Carlo Foundations</b></summary>

- **Focus**: Sampling-based numerical approximation, geometric probability, and randomness extraction.
- **Key Concepts**:
  - Estimating $\pi$ via uniform rejection sampling and analyzing convergence rates ($\mathcal{O}(1/\sqrt{N})$).
  - High-dimensional geometry: The ratio between the volume of an inscribed $d$-dimensional ball and hypercube, demonstrating concentration of measure and the curse of dimensionality.
  - Correct uniform sampling within a $d$-dimensional unit sphere using Gaussian vectors and radial scaling.
  - The Von Neumann randomness extractor: Generating unbiased Bernoulli trials from a biased coin.
- **Notebook**: [`lab1/lab1.ipynb`](./lab1/lab1.ipynb)
</details>

<details>
<summary><b>Lab 2: Linear Gaussian Models & Object Tracking</b></summary>

- **Focus**: Analytical Bayesian inference with Gaussian likelihoods and priors.
- **Key Concepts**:
  - Modeling noisy physical sensors and radar range/bearing measurements.
  - Joint and conditional multivariate Gaussian distributions.
  - Analytical posterior derivation for object location from single and multiple noisy radar echoes.
  - Sequential Bayesian updating as an introduction to Kalman filtering.
- **Notebook**: [`lab2/lab2.ipynb`](./lab2/lab2.ipynb)
</details>

<details>
<summary><b>Lab 3: Linear & Bayesian Regression</b></summary>

- **Focus**: Comparing frequentist parameter estimation with full posterior probability densities.
- **Key Concepts**:
  - Maximum Likelihood Estimation (MLE) and Ordinary Least Squares (OLS) closed-form solutions.
  - Conjugate Gaussian priors on linear weights: Derivation of the weight posterior distribution $p(\mathbf{w} \mid \mathcal{D})$.
  - Maximum A Posteriori (MAP) estimation and its equivalence to $L_2$ Ridge regularization.
  - Predictive distributions: Visualizing epistemic vs. aleatoric uncertainty across feature space.
- **Notebook**: [`lab3/lab3.ipynb`](./lab3/lab3.ipynb)
</details>

<details>
<summary><b>Lab 4: Gaussian Process Regression & Bayesian Optimization</b></summary>

- **Focus**: Non-parametric function estimation and sample-efficient global optimization.
- **Key Concepts**:
  - Properties of covariance functions: Radial Basis Function (RBF), Matérn, and periodic kernels.
  - Definition of Gaussian Processes: Any finite collection of evaluation points is jointly Gaussian.
  - Conditioning Gaussian Processes on observed data points: Analytical mean and variance functions.
  - Application to **Bayesian Optimization**: Acquisition functions (Upper Confidence Bound - UCB, Expected Improvement - EI) for tuning expensive black-box objectives.
- **Notebook**: [`lab4/lab4.ipynb`](./lab4/lab4.ipynb)
</details>

<details>
<summary><b>Lab 5: Multinomial Logistic Regression</b></summary>

- **Focus**: Vectorized probabilistic classification derived from first principles.
- **Key Concepts**:
  - Multinomial logistic regression (Softmax regression) for multi-class classification.
  - Derivation of the multi-class negative log-likelihood (cross-entropy) loss function.
  - Analytical computation of gradients with respect to weight matrices and bias vectors.
  - Batch and mini-batch gradient descent optimization implemented in pure NumPy and tested on the MNIST handwritten digits dataset.
- **Notebook**: [`lab5/lab5.ipynb`](./lab5/lab5.ipynb)
</details>

<details>
<summary><b>Lab 6: Probabilistic Programming with TFP & Importance Sampling</b></summary>

- **Focus**: Declarative probabilistic modeling and Monte Carlo integration.
- **Key Concepts**:
  - Introduction to **TensorFlow Probability** (TFP) distributions and bijectors.
  - Building joint log-probability functions for hierarchical Bayesian models.
  - **Importance Sampling**: Computing expectations with respect to intractable target distributions using proposal distributions.
  - Evaluating proposal quality, diagnostics, and calculating effective sample size (ESS).
- **Notebook**: [`lab6/lab6.ipynb`](./lab6/lab6.ipynb)
</details>

<details>
<summary><b>Lab 7: MCMC Inference in Hierarchical Bayesian Models</b></summary>

- **Focus**: Markov Chain Monte Carlo algorithms for intractable posterior inference.
- **Key Concepts**:
  - Foundations of MCMC: Transition kernels, detailed balance, and stationary distributions.
  - Metropolis-Hastings (MH) and Hamiltonian Monte Carlo (HMC) sampling in TensorFlow Probability.
  - Diagnosing convergence: Trace plots, autocorrelation time, and Gelman-Rubin $\hat{R}$.
  - Application to Change-Point and Interval Change detection in non-stationary time series data.
- **Notebook**: [`lab7/lab7.ipynb`](./lab7/lab7.ipynb)
</details>

---

## Getting Started

Ensure you have [uv](https://docs.astral.sh/uv/) installed.

### 1. Clone & Sync Environment

Clone the repository and run `uv sync`. This automatically provisions Python 3.11, creates a local `.venv`, and installs all locked dependencies:

```bash
git clone https://github.com/barhanc/statistical-machine-learning.git
cd statistical-machine-learning

# Synchronize the virtual environment with uv.lock in seconds
uv sync
```

### 2. Launch Jupyter

Run JupyterLab or Jupyter Notebook directly through `uv`:

```bash
# Launch JupyterLab
uv run jupyter lab

# Or classic Jupyter Notebook
uv run jupyter notebook
```
