# Portfolio_Performance

:construction:  This README is a work in progress

Tool to  test the out-of-sample performance of portfolio optimization models. Based on the paper: [Optimal Versus Naive Diversification:
How Inefficient is the 1/N Portfolio Strategy?](http://faculty.london.edu/avmiguel/DeMiguel-Garlappi-Uppal-RFS.pdf) by Victor DeMiguel, Lorenzo Garlappi and Raman Uppal

I have also developed a playground website for this package so you can easily run these models on your own data. [Link to website](https://sidnand.github.io/Portfolio-Optimization-Interface/)

## Usage

### Installation

`pip install portfolio-performance`

### Models

```python
# Each model takes a name parameter
ew = EqualWeight("Equal Weight")
minVar = MinVar("Minimum Variance")
JagannathanMa = JagannathanMa("Jagannathan Ma")
minVarShortSellCon = MinVarShortSellCon("Minimum Variance with Short Sell Constrains")
kanZhouEw = KanZhouEw("Kan Zhou EW")

meanVar = MeanVar("Mean Variance (Markowitz)")
meanVarShortSellCon = MeanVarShortSellCon("Mean Variance with Short Sell Constrains")
kanZhou = KanZhou("Kan Zhou Three Fund")
bayesStein = BayesStein("Bayes Stein")
bayesSteinShortSellCon = BayesSteinShortSellCon("Bayes Stein with Short Sell Constrains")
macKinlayPastor = MacKinlayPastor("MacKinlay and Pastor")
```

### Example

```python
import numpy as np
from portfoliotest import *

# Risk aversion levels
GAMMAS = [1, 2, 3, 4, 5, 10]

# Time horizons
TIME_HORIZON = [60, 120]

benchmark = <benchmark model>

# List of models
models = [
    benchmark,
    # <add other list of models>
]

app = App(<data path>, GAMMAS, TIME_HORIZON,
              models, date=<true or false>, riskFactorPositions=[0-indexed positions for risk factor column], riskFreePosition=<0-indexed, risk free asset column>)

sr = app.getSharpeRatios()
sig = app.getStatisticalSignificanceWRTBenchmark(benchmark)
```

## API

:construction: Coming soon

## Development

### Setup

1. Clone project: `git clone https://github.com/sidnand/portfolio-performance`
2. Install packages: `pip install -r requirements.txt`
3. Make project: `setapp `
4. Make changes!

### Folder Structure

```bash
.
|-- __init__.py # import /src/app.py and all models
|-- /src
    |-- app.py # code for running all the optimiation models
    |-- model.py # parent class to all the models
    |-- modelNoGamma.py # class for models that don't take extra parameters
    |-- modelGamma.py # class for models that take an extra gamma parameter; gamma is a list of constants for the investors risk-aversion level
    |-- ./models # all the models
    |-- ./utils
        |-- filter.py # includes a function that is used to filter the parameters passed to a function
        |-- quadprog.py # quadratic programming
        |-- sharedOptions.py # models options that are in common with >2 models
        |-- statistics.py # statistics functions

```

### Create Pull-Request

Please create a pull-request to include your changes onto this repo. These changes will be reflected on the playground website linked above.

1. Run ``pip install -e .`` to install the package locally.
2. Make changes.
3. Update version number in ``setup.py``. Please use the Semantic Versioning 2.0.0 system. [Click to learn more](https://semver.org/).
4. Run ``python setup.py sdist bdist_wheel`` to create a python wheel.
5. Create a pull-request!