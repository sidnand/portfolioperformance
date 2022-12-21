# Testing Portfolio Optimization Methods

This is a tool to test the out-of-sample performance using the Sharpe ratio of different portfolio methods against each other. This is based on the paper "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" by Victor DeMiguel, Lorenzo Garlappi and Raman Uppal, and it uses the same models.

## Models Built-in Currently

* Equal weighted model
* Mean-varaince (Markowitz) model
* Minimum-variance
* Minimum-varaince with shortsale constraints model
* Minimum-varaince with generalized constraints (Jagannathan Ma) model
* Kan Zhou equal weighted "three-fund" model

## Data

``` bash
    data/new/orig/sp_sector # contains the uncleaned data from various industries in the S&P500
```