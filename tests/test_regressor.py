import numpy as np
import pandas as pd

from usopp import Regressor
from usopp.utils import IdentityScaler


def test_can_fit_generated_data(regressor_data):
    data, true_k, n_features = regressor_data
    feature_names = [f"feature{i}" for i in range(n_features)]
    model = Regressor(on=feature_names)
    model.fit(
        data[['t', *feature_names]],
        data['value'],
        y_scaler=IdentityScaler,
    )
    model_k = np.mean(model.trace_[model._param_name("k")], axis=0)
    res = model.predict(data[['t', *feature_names]])
    np.testing.assert_allclose(res.yhat.squeeze(), data['value'], atol=0.01)
    np.testing.assert_allclose(model_k, true_k, atol=0.12)
