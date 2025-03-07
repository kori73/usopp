import numpy as np
import pytensor.tensor as pt
import pytensor
from usopp.timeseries_model import TimeSeriesModel
from usopp.utils import add_subplot, get_group_definition
import pymc as pm


class LogisticGrowth(TimeSeriesModel):
    def __init__(
            self, capacity: float, name: str = None, n_changepoints=None,
            changepoints_prior_scale=0.05, growth_prior_scale=1, pool_cols=None,
            pool_type='complete'
    ):
        self.cap = capacity
        self.n_changepoints = n_changepoints
        self.changepoints_prior_scale = changepoints_prior_scale
        self.growth_prior_scale = growth_prior_scale
        self.pool_cols = pool_cols
        self.pool_type = pool_type
        self.name = name or f"LogisticGrowth(n_changepoints={n_changepoints})"
        super().__init__()

    def definition(self, model, X, scale_factor):

        def update_gamma(j, gamma, i, delta, offset, rate, change_point):
            return pt.set_subtensor(
                gamma[i, j],
                (change_point[j] - offset[i] - pt.sum(gamma[i, :j])) *
                (1 - (rate[i] + pt.sum(delta[i, :j])) / (rate[i] + pt.sum(delta[i, :j+1])))
                )

        def get_gamma(i, gamma_init, delta, m, k, s):
            gamma, _ = pytensor.scan(
              update_gamma,
              sequences=[
                  pt.arange(gamma_init.shape[1]),
              ],
              outputs_info=gamma_init,
              non_sequences=[i, delta, m, k, s],
            )
            return gamma[-1]

        t = X["t"].values
        self.cap_scaled = self._y_scaler_.transform(self.cap)
        group, n_groups, self.groups_ = get_group_definition(X, self.pool_cols, self.pool_type)
        self.s = np.linspace(0, np.max(t), self.n_changepoints + 2)[1:-1]

        with model:
            A = (t[:, None] > self.s) * 1.0

            if self.pool_type == 'partial':
                sigma_k = pm.HalfCauchy(self._param_name('sigma_k'), beta=self.growth_prior_scale)
                offset_k = pm.Normal(self._param_name('offset_k'), mu=0, sigma=1, shape=n_groups)
                k = pm.Deterministic(self._param_name("k"), offset_k * sigma_k)

                sigma_delta = pm.HalfCauchy(
                    self._param_name('sigma_delta'), beta=self.changepoints_prior_scale
                )
                offset_delta = pm.Laplace(
                    self._param_name('offset_delta'), 0, 1, shape=(n_groups, self.n_changepoints)
                )
                delta = pm.Deterministic(self._param_name("delta"), offset_delta * sigma_delta)

            else:
                delta = pm.Laplace(self._param_name("delta"), 0, self.changepoints_prior_scale,
                                   shape=(n_groups, self.n_changepoints))
                k = pm.Normal(self._param_name("k"), 0, self.growth_prior_scale,
                              shape=n_groups, initval=np.ones(n_groups))

            m = pm.Normal(self._param_name("m"), 0, 5, shape=n_groups)

            gamma_init = pt.zeros_like(delta)
            gamma, _ = pytensor.scan(
                get_gamma,
                sequences=[pt.arange(gamma_init.shape[0])],
                outputs_info=gamma_init,
                non_sequences=[delta, m, k, self.s],
            )
            gamma = gamma[-1]
            growth = (
                (k[group] + pm.math.sum(A * delta[group], axis=1)) *
                (t - (m[group] + pm.math.sum(A * gamma[group], axis=1)))
            )
            growth = self.cap_scaled / (1 + pm.math.exp(-growth))
        return growth

    def _predict(self, trace, t, pool_group=0):
        t = t.squeeze()
        delta = trace[self._param_name("delta")]
        if isinstance(delta, np.ndarray):
            delta = delta[pool_group].reshape(1, self.n_changepoints)
            k = trace[self._param_name("k")][pool_group]
            m = trace[self._param_name("m")][pool_group]
            gamma = np.zeros(delta.T.shape)
        else:
            delta = delta.values[:, :, pool_group, :].reshape(-1, self.n_changepoints)
            k = trace[self._param_name("k")].values[:, :, pool_group].reshape(1, -1)
            m = trace[self._param_name("m")].values[:, :, pool_group].reshape(1, -1)
            gamma = np.zeros(delta.T.shape)

        A = (t[:, None] > self.s) * 1
        
        for i in range(gamma.shape[0]):
            gamma[i] = (
                (self.s[i] - m - gamma[:i].sum(axis=0)) * (1 - ((k + delta[:, :i].sum(axis=1)) / (k + delta[:, :i+1].sum(axis=1))))
            )
        g = (
            (k + A @ delta.T) *
            (t[:, None] - (m + A @ gamma))
        )
        return self.cap_scaled / (1 + np.exp(-g))

    def plot(self, trace, scaled_t, y_scaler, drawer):
        ax = add_subplot()
        ax.set_title(str(self))
        ax.set_xticks([])
        growth_return = np.empty((len(scaled_t), len(self.groups_)))
        for group_code, group_name in self.groups_.items():
            scaled_growth = self._predict(trace, scaled_t, group_code)
            growth = y_scaler.inv_transform(scaled_growth)
            ax.plot(scaled_t, growth.mean(axis=1), label=group_name)
            growth_return[:, group_code] = scaled_growth.mean(axis=1)

        for changepoint in self.s:
            ax.axvline(changepoint, linestyle="--", alpha=0.2, c="k")
        ax.legend()
        return growth_return
