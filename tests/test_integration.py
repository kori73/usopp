import numpy as np
from usopp import FourierSeasonality, LinearTrend, Regressor

def test_can_predict_additive(additive_timeseries_data):
    data, n_components, n_changepoints, n_features = additive_timeseries_data
    feature_names = [f"feature{i}" for i in range(n_features)]
    model = (
        FourierSeasonality(n=n_components) 
        + LinearTrend(n_changepoints=n_changepoints)
        + Regressor(on=feature_names)
    )
    model.fit(data[["t", *feature_names]], data["value"])
    res = model.predict(data[["t", *feature_names]])
    np.testing.assert_allclose(data.value, res.yhat.squeeze(), atol=0.1)

def test_can_predict_multiplicative(multiplicative_seasonality_data):
    data, n_components, n_changepoints, n_features = multiplicative_seasonality_data
    feature_names = [f"feature{i}" for i in range(n_features)]
    model = (
        FourierSeasonality(n=n_components)
        * LinearTrend(n_changepoints=n_changepoints)
        + Regressor(on=feature_names)
    )
    model.fit(data[["t", *feature_names]], data["value"])
    res = model.predict(data[["t", *feature_names]])
    np.testing.assert_allclose(data.value, res.yhat.squeeze(), atol=0.25)
