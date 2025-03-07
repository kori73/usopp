import numpy as np
import pandas as pd

from usopp import RBFSeasonality
from usopp.utils import IdentityScaler, get_periodic_peaks


def test_can_fit_generated_data(rbf_seasonal_data):
    data, true_beta, n_components = rbf_seasonal_data
    ps = get_periodic_peaks(n_components)
    model = RBFSeasonality(peaks=ps, period=pd.Timedelta(days=365.25), sigma=0.015)
    model.fit(data[['t']], data['value'], y_scaler=IdentityScaler)
    model_beta = np.mean(model.trace_[model._param_name("beta")], axis=0)
    res = model.predict(data[['t']])
    np.testing.assert_allclose(res.yhat.squeeze(), data['value'], atol=0.01)
    np.testing.assert_allclose(model_beta, true_beta, atol=0.12)
