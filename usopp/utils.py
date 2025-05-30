import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


def dot(a, b):
    return (a * b[None, :]).sum(axis=-1)


class IdentityScaler:
    def fit(self, data):
        self.scale_factor_ = 1
        return self

    def transform(self, data):
        return data

    def inv_transform(self, data):
        return data

    def fit_transform(self, data):
        return data


class MinMaxScaler:
    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            self.max_ = data.max(axis=0)
            self.min_ = data.min(axis=0)
            self.scale_factor_ = (self.max_ - self.min_).where(self.max_ != self.min_, 1)
        if isinstance(data, np.ndarray):
            self.max_ = data.max(axis=0)[None, ...]
            self.min_ = data.min(axis=0)[None, ...]
            self.scale_factor_ = np.where(self.max_ != self.min_, self.max_ - self.min_, 1)
        if isinstance(data, pd.Series):
            self.max_ = data.max()
            self.min_ = data.min()
            self.scale_factor_ = self.max_ - self.min_

        return self

    def transform(self, series):
        return ((series - self.min_) / self.scale_factor_).astype("float")

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)

    def inv_transform(self, series):
        return series * self.scale_factor_ + self.min_


class MaxScaler:
    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            self.scale_factor_ = data.max(axis=0)
        if isinstance(data, np.ndarray):
            self.scale_factor_ = data.max(axis=0)[None, ...]
        if isinstance(data, pd.Series):
            self.scale_factor_ = data.max()
        return self

    def transform(self, series):
        return series / self.scale_factor_

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)

    def inv_transform(self, series):
        return series * self.scale_factor_


class StdScaler:
    def fit(self, data):
        if isinstance(data, pd.Series):
            self.mean_ = data.mean()
            self.std_ = data.std()

        return self

    def transform(self, series):
        return (series - self.mean_) / self.std_

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)

    def inv_transform(self, series):
        return series * self.std_ + self.mean_


def add_subplot(height=5):
    fig = plt.gcf()
    n = len(fig.axes)
    for i in range(n):
        fig.axes[i].change_geometry(n + 1, 1, i + 1)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h + height)
    return fig.add_subplot(len(fig.axes) + 1, 1, len(fig.axes) + 1)


class Drawer():

    def __init__(self) -> None:
        """Constructs a Drawer for one column and multiple rows.

        Note that subplots are added to the bottom.
        """
        self.fig = plt.figure()

        # Start with one subplot
        self.row = 0

    def add_subplot(self) -> None:
        """Plots the data to a new subplot at the bottom."""
        self.row += 1
        gs = gridspec.GridSpec(self.row, 1)

        # Reposition existing subplots
        for i, ax in enumerate(self.fig.axes):
            ax.set_position(gs[i].get_position(self.fig))
            ax.set_subplotspec(gs[i])

        # Add new subplot
        new_ax = self.fig.add_subplot(gs[self.row-1])
        return new_ax

    def show(self) -> None:
        plt.show()


def trend_data(n_changepoints, location="spaced", noise=0.001):
    delta = np.random.laplace(size=n_changepoints)

    t = np.linspace(0, 1, 1000)

    if location == "random":
        s = np.sort(np.random.choice(t, n_changepoints, replace=False))
    elif location == "spaced":
        s = np.linspace(0, np.max(t), n_changepoints + 2)[1:-1]
    else:
        raise ValueError('invalid `location`, should be "random" or "spaced"')

    A = (t[:, None] > s) * 1

    k, m = 0, 0

    growth = k + A @ delta
    gamma = -s * delta
    offset = m + A @ gamma
    trend = growth * t + offset + np.random.randn(len(t)) * noise

    return (
        pd.DataFrame({"t": pd.date_range("2018-1-1", periods=len(t)), "value": trend}),
        delta,
    )


def logistic_growth_data(n_changepoints, location="spaced", noise=0.001, loc=0, scale=0.2):
    delta = np.random.laplace(size=n_changepoints, loc=loc, scale=scale)
    gamma = np.zeros(n_changepoints)

    t = np.linspace(0, 1, 1000)
    if location == "random":
        s = np.sort(np.random.choice(t, n_changepoints, replace=False))
    elif location == "spaced":
        s = np.linspace(0, np.max(t), n_changepoints + 2)[1:-1]
    else:
        raise ValueError('invalid `location`, should be "random" or "spaced"')

    A = (t[:, None] > s) * 1
    k, m = 2.5, 0

    for i in range(n_changepoints):
        left = (s[i] - m - np.sum(gamma[:i]))
        right = (1 - (k + np.sum(delta[:i])) / (k + np.sum(delta[:i+1])))
        gamma[i] = left * right

    g = (k + np.sum(A * delta, axis=1)) * (t - (m + np.sum(A * gamma, axis=1)))
    logistic_growth = 1 / (1 + np.exp(-g)) + np.random.randn(len(t)) * noise
    return (
        pd.DataFrame({"t": pd.date_range("2018-1-1", periods=len(t)), "value": logistic_growth}),
        delta,
    )


def seasonal_data(n_components, noise=0.001):
    def X(t, p=365.25, n=10):
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate((np.cos(x), np.sin(x)), axis=1)

    t = np.linspace(0, 1, 1000)
    beta = np.random.normal(size=2 * n_components)

    seasonality = X(t, 365.25 / len(t), n_components) @ beta + np.random.randn(len(t)) * noise

    return (
        pd.DataFrame(
            {"t": pd.date_range("2018-1-1", periods=len(t)), "value": seasonality}
        ),
        beta,
    )


def rbf_seasonal_data(n_components, sigma=0.015, noise=0.001):
    def X(t, peaks, sigma, year):
        mod = np.array((t % year))[:, None]
        left_difference = np.sqrt((mod - peaks[None, :]) ** 2)
        right_difference = np.abs(year - left_difference)
        return np.exp(- ((np.minimum(left_difference, right_difference)) ** 2) / (2 * sigma**2))

    t = pd.Series(pd.date_range("2010-01-01", "2014-01-01"))
    scaler = MinMaxScaler()
    scaled_t = scaler.fit_transform(t)
    scale_factor = t.max() - t.min()
    beta = np.random.normal(size=n_components)
    peaks = get_periodic_peaks(n_components)
    peaks = np.array([p / scale_factor for p in peaks])
    period = pd.Timedelta(days=365.25)
    seasonality = X(scaled_t, peaks, sigma, period / scale_factor) @ beta + np.random.randn(len(t)) * noise
    return (
        pd.DataFrame(
            {"t": pd.date_range("2018-1-1", periods=len(t)), "value": seasonality}
        ),
        beta,
    )


def regressor_data(n_features, loc=0., scale=1., noise=0.001, binary=False):
    t = np.linspace(0, 1, 1000)

    k = np.random.normal(loc, scale, size=(n_features))
    if binary:
        features = np.random.binomial(n=1, p=0.1, size=(len(t), n_features))
    else:
        features = np.random.normal(0, scale, size=(len(t), n_features))
    value  = features @ k + np.random.randn(len(t)) * noise

    df = pd.DataFrame(
        {"t": pd.date_range("2018-1-1", periods=len(t)), "value": value}
    )
    for i in range(n_features):
        df[f"feature{i}"] = features[:, i]
    return df, k

def additive_timeseries_data(n_components=5, n_changepoints=2, n_features=2):

    df_seasonal, _ = seasonal_data(n_components=n_components)
    df_trend, _ = trend_data(n_changepoints=n_changepoints)
    df_regressor, _ = regressor_data(loc=2.0, n_features=n_features, binary=True)

    df_seasonal = df_seasonal.rename(columns={"value": "value_seasonal"})
    df_trend = df_trend.rename(columns={"value": "value_trend"})
    df_trend = df_trend.drop(columns=["t"])

    df_regressor = df_regressor.rename(columns={"value": "value_regressor"})
    df_regressor = df_regressor.drop(columns=["t"])

    df = pd.concat([df_seasonal, df_trend, df_regressor], axis=1)
    df["value"] = df["value_seasonal"] + df["value_trend"] + df["value_regressor"]
    df = df.drop(columns=["value_seasonal", "value_trend", "value_regressor"], axis=1)
    return df

def multiplicative_seasonality_data(n_components=5, n_changepoints=2, n_features=2):

    df_seasonal, _ = seasonal_data(n_components=n_components)
    df_trend, _ = trend_data(n_changepoints=n_changepoints)
    df_trend["value"] = df_trend["value"] + 1.0
    df_regressor, _ = regressor_data(loc=0.0, n_features=n_features, binary=True)

    df_seasonal = df_seasonal.rename(columns={"value": "value_seasonal"})
    df_trend = df_trend.rename(columns={"value": "value_trend"})
    df_trend = df_trend.drop(columns=["t"])

    df_regressor = df_regressor.rename(columns={"value": "value_regressor"})
    df_regressor = df_regressor.drop(columns=["t"])

    df = pd.concat([df_seasonal, df_trend, df_regressor], axis=1)
    df["value"] = df["value_seasonal"] * df["value_trend"] + df["value_regressor"]
    df = df.drop(columns=["value_seasonal", "value_trend", "value_regressor"], axis=1)
    return df

def get_group_definition(X, pool_cols, pool_type):
    if pool_type == 'complete':
        group = np.zeros(len(X), dtype='int')
        group_mapping = {0: 'all'}
        n_groups = 1
    else:
        group = X[pool_cols].cat.codes.values
        group_mapping = dict(enumerate(X[pool_cols].cat.categories))
        n_groups = X[pool_cols].nunique()
    return group, n_groups, group_mapping


def get_periodic_peaks(
        n: int = 20,
        period: pd.Timedelta = pd.Timedelta(days=365.25)):
    """
    Returns n periodic peaks that repeats each period. Return value
    can be used in RBFSeasonality.
    """
    return np.array([period * i / n for i in range(n)])
