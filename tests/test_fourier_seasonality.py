from usopp import FourierSeasonality
import numpy as np
from usopp.utils import IdentityScaler


def test_can_fit_generated_data(seasonal_data):
    data, true_beta, n_components = seasonal_data
    model = FourierSeasonality(n=n_components)
    model.fit(data[["t"]], data["value"], y_scaler=IdentityScaler)

    # No groups, so we can take mean over 1 as well
    model_beta = np.mean(model.trace_[model._param_name("beta")], axis=0)
    np.testing.assert_allclose(model_beta, true_beta, atol=0.12)
    res = model.predict(data[["t"]])
    np.testing.assert_allclose(res.yhat.squeeze(), data["value"], atol=0.5)
