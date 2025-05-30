import numpy as np
from usopp.timeseries_model import TimeSeriesModel
from usopp.utils import add_subplot, get_group_definition
import pymc as pm
from xarray.core.dataset import Dataset

class Regressor(TimeSeriesModel):
    def __init__(self, on: str, scale: float = 1., name: str = None, pool_cols=None, pool_type='complete'):
        self.on = on
        self.scale = scale
        self.pool_cols = pool_cols
        self.pool_type = pool_type

        self.name = name or f"LinearRegressor(on={self.on}, scale={self.scale}, " \
                            f"pool_cols='{self.pool_cols}', pool_type='{self.pool_type}')"
        super().__init__()

    def definition(self, model, X, scale_factor):
        self.feature_indices_ = X.columns.get_indexer(self.on)
        self.shape_ = len(self.on)

        group, n_groups, self.groups_ = get_group_definition(X, self.pool_cols, self.pool_type)
        with model:
            if self.pool_type == "partial":
                sigma_k = pm.HalfCauchy(self._param_name('sigma_k'), beta=self.scale)
                offset_k = pm.Normal(self._param_name('offset_k'), mu=0, sigma=1, shape=(n_groups, self.shape_))
                k = pm.Deterministic(self._param_name("k"), offset_k * sigma_k)

            else:
                k = pm.Normal(self._param_name('k'), mu=0, sigma=self.scale, shape=(n_groups, self.shape_))
        return pm.math.sum(X[self.on].values * k[group], axis=1)

    def _predict(self, trace, t, pool_group=0):
        if isinstance(trace, Dataset):
            k = trace[self._param_name("k")][:, :, pool_group, :].values.reshape(self.shape_, -1)
        else:
            k = trace[self._param_name("k")][pool_group, :]
        X = t[:, self.feature_indices_]
        result = X @ k
        if result.ndim == 1:
            result = result[:, np.newaxis]
        return result

    def plot(self, trace, scaled_t, y_scaler, drawer):
        ax = drawer.add_subplot()
        ax.set_title(str(self))
        # ax.set_xticks([])
        trend_return = np.empty((len(scaled_t), len(self.groups_)))
        plot_data = []
        for group_code, group_name in self.groups_.items():
            y_hat = np.mean(self._predict(trace, scaled_t, group_code), axis=1)
            trend_return[:, group_code] = y_hat
            plot_data.append((group_name, y_hat[0]))
        ax.bar(*zip(*plot_data))
        ax.axhline(0, c='k', linewidth=3)

        return trend_return

    def __repr__(self):
        return f"LinearRegressor(on={self.on}, pool_cols={self.pool_cols}, pool_type={self.pool_type})"
