import numpy as np
import pytest
from usopp import utils


@pytest.fixture(params=[1, 5, 10])
def trend_data(request):
    np.random.seed(42)
    n_changepoints = request.param
    data, delta = utils.trend_data(n_changepoints, noise=0.0001)
    return data, delta, n_changepoints


@pytest.fixture(params=[1, 5, 10])
def logistic_growth_data(request):
    np.random.seed(42)
    n_changepoints = request.param
    data, delta = utils.logistic_growth_data(n_changepoints, noise=0.0001)
    return data, delta, n_changepoints


@pytest.fixture(params=[1, 5, 10])
def seasonal_data(request):
    np.random.seed(42)
    n_components = request.param
    data, beta = utils.seasonal_data(n_components, noise=0.0000000001)
    return data, beta, n_components


@pytest.fixture(params=[1, 5, 10])
def rbf_seasonal_data(request):
    np.random.seed(42)
    n_components = request.param
    data, beta = utils.rbf_seasonal_data(n_components, noise=0.0000000001)
    return data, beta, n_components

@pytest.fixture(params=[1, 5, 10])
def regressor_data(request):
    np.random.seed(42)
    n_features = request.param
    data, k = utils.regressor_data(n_features, noise=0.0000000001)
    return data, k, n_features

@pytest.fixture(params=[(5, 2, 2)])
def additive_timeseries_data(request):
    np.random.seed(42)
    n_components, n_changepoints, n_features = request.param
    data = utils.additive_timeseries_data(n_components=5)
    return data, n_components, n_changepoints, n_features

@pytest.fixture(params=[(5, 2, 2)])
def multiplicative_seasonality_data(request):
    np.random.seed(42)
    n_components, n_changepoints, n_features = request.param
    data = utils.multiplicative_seasonality_data(n_components=5)
    return data, n_components, n_changepoints, n_features
