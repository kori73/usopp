from usopp import LogisticGrowth
from usopp.utils import MaxScaler
import numpy as np


def test_can_fit_generated_data(logistic_growth_data):
    data, true_delta, n_changepoints = logistic_growth_data
    data["value"] = data["value"] * 1000
    model = LogisticGrowth(capacity=1000, n_changepoints=n_changepoints)
    model.fit(data[["t"]], data["value"], y_scaler=MaxScaler)
    model_delta = model.trace_[model._param_name("delta")].mean(axis=0)
    res = model.predict(data[["t"]])
    np.testing.assert_allclose(res.yhat.squeeze(), data["value"], atol=1.0)
    np.testing.assert_allclose(model_delta, true_delta, atol=0.01)
