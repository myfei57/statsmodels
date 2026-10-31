from statsmodels.compat.pandas import MONTH_END

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import statsmodels.datasets
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.testing import warns_or_none
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.statespace.exponential_smoothing import (
    ExponentialSmoothing,
)


@pytest.fixture(scope="module")
def data(request):
    rs = np.random.RandomState(987654321)
    err = rs.standard_normal(500)
    index = pd.date_range("1980-1-1", freq=MONTH_END, periods=500)
    fourier = Fourier(12, 1)
    terms = fourier.in_sample(index)
    det = np.squeeze(np.asarray(terms @ np.array([[2], [1]])))
    for i in range(1, 500):
        err[i] += 0.9 * err[i - 1] + det[i]
    return pd.Series(err, index=index)


def test_smoke(data):
    stlf = STLForecast(data, ARIMA, model_kwargs={"order": (2, 0, 0)})
    res = stlf.fit(fit_kwargs={})
    res.forecast(37)
    assert isinstance(res.summary().as_text(), str)
    assert isinstance(res.stl, STL)
    assert isinstance(res.result, DecomposeResult)
    assert isinstance(res.model, ARIMA)
    assert hasattr(res.model_result, "forecast")
    assert res.result.nobs == 500


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_sharex(data, close_figures):
    stlf = STLForecast(data, ARIMA, model_kwargs={"order": (2, 0, 0)})
    res = stlf.fit(fit_kwargs={})
    plt = res.result.plot()
    grouper_view = plt.axes[0].get_shared_x_axes()
    sibs = grouper_view.get_siblings(plt.axes[1])
    assert len(sibs) == 4


MODELS = [
    (ARIMA, {"order": (2, 0, 0), "trend": "c"}),
    (ExponentialSmoothing, {"trend": True}),
    (AutoReg, {"lags": 2}),
    (ETSModel, {}),
]
MODELS = MODELS[-1:]
IDS = [str(c[0]).split(".")[-1][:-2] for c in MODELS]


@pytest.mark.parametrize("config", MODELS, ids=IDS)
@pytest.mark.parametrize("horizon", [1, 7, 23])
def test_equivalence_forecast(data, config, horizon, close_figures):
    model, kwargs = config

    stl = STL(data)
    stl_fit = stl.fit()
    resids = data - stl_fit.seasonal
    mod = model(resids, **kwargs)
    fit_kwarg = {}
    if model is ETSModel:
        fit_kwarg["disp"] = False
    res = mod.fit(**fit_kwarg)
    stlf = STLForecast(data, model, model_kwargs=kwargs).fit(fit_kwargs=fit_kwarg)

    seasonal = np.asarray(stl_fit.seasonal)[-12:]
    seasonal = np.tile(seasonal, 1 + horizon // 12)
    fcast = res.forecast(horizon) + seasonal[:horizon]
    actual = stlf.forecast(horizon)
    assert_allclose(actual, fcast, rtol=1e-4)
    if not hasattr(res, "get_prediction"):
        return
    pred = stlf.get_prediction(data.shape[0], data.shape[0] + horizon - 1)
    assert isinstance(pred, PredictionResults)
    assert_allclose(pred.predicted_mean, fcast, rtol=1e-4)

    half = data.shape[0] // 2
    stlf.get_prediction(half, data.shape[0] + horizon - 1)
    stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=True)
    stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=half // 2)
    if hasattr(data, "index"):
        loc = data.index[half + half // 2]
        a = stlf.get_prediction(
            half, data.shape[0] + horizon - 1, dynamic=loc.strftime("%Y-%m-%d")
        )
        b = stlf.get_prediction(
            half, data.shape[0] + horizon - 1, dynamic=loc.to_pydatetime()
        )
        c = stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=loc)
        assert_allclose(a.predicted_mean, b.predicted_mean, rtol=1e-4)
        assert_allclose(a.predicted_mean, c.predicted_mean, rtol=1e-4)


def test_exceptions(data):
    class BadModel:
        def __init__(self, *args, **kwargs):
            pass

    with pytest.raises(AttributeError, match="model must expose"):
        STLForecast(data, BadModel)

    class NoForecast(BadModel):
        def fit(self, *args, **kwargs):
            return BadModel()

    with pytest.raises(AttributeError, match="The model's result"):
        STLForecast(data, NoForecast).fit()

    class BadResult:
        def forecast(self, *args, **kwargs):
            pass

    class FakeModel(BadModel):
        def fit(self, *args, **kwargs):
            return BadResult()

    with pytest.raises(AttributeError, match="The model result does not"):
        STLForecast(data, FakeModel).fit().summary()

    class BadResultSummary(BadResult):
        def summary(self, *args, **kwargs):
            return object()

    class FakeModelSummary(BadModel):
        def fit(self, *args, **kwargs):
            return BadResultSummary()

    with pytest.raises(TypeError, match="The model result's summary"):
        STLForecast(data, FakeModelSummary).fit().summary()


@pytest.fixture
def sunspots():
    df = statsmodels.datasets.sunspots.load_pandas().data
    df.index = np.arange(df.shape[0])
    return df.iloc[:, 0]


def test_get_prediction(sunspots):
    # GH7309
    stlf_model = STLForecast(
        sunspots, model=ARIMA, model_kwargs={"order": (2, 2, 0)}, period=11
    )
    # The default optimizer may stop short of convergence on some platforms
    with warns_or_none(ConvergenceWarning, match="Maximum Likelihood optimization"):
        stlf_res = stlf_model.fit()
    pred = stlf_res.get_prediction()
    assert pred.predicted_mean.shape == (309,)
    assert pred.var_pred_mean.shape == (309,)


@pytest.mark.parametrize("not_implemented", [True, False])
def test_no_var_pred(sunspots, not_implemented, close_figures):
    class DummyPred:
        def __init__(self, predicted_mean, row_labels):
            self.predicted_mean = predicted_mean
            self.row_labels = row_labels

            def f():
                raise NotImplementedError

            if not_implemented:
                self.forecast = property(f)

    class DummyRes:
        def __init__(self, res):
            self._res = res

        def forecast(self, *args, **kwargs):
            return self._res.forecast(*args, **kwargs)

        def get_prediction(self, *args, **kwargs):
            pred = self._res.get_prediction(*args, **kwargs)

            return DummyPred(pred.predicted_mean, pred.row_labels)

    class DummyMod:
        def __init__(self, y):
            self._mod = ARIMA(y)

        def fit(self, *args, **kwargs):
            res = self._mod.fit(*args, **kwargs)
            return DummyRes(res)

    stl_mod = STLForecast(sunspots, model=DummyMod, period=11)
    stl_res = stl_mod.fit()
    with pytest.warns(UserWarning, match="The variance of"):
        pred = stl_res.get_prediction()
    assert np.all(np.isnan(pred.var_pred_mean))


class RecordingResults:
    """Deterministic stand-in results for a fitted forecasting model."""

    def __init__(self, endog):
        self._last = float(np.asarray(endog)[-1])

    def forecast(self, steps=1, **kwargs):
        RecordingModel.forecast_steps.append(steps)
        RecordingModel.forecast_kwargs.append(kwargs)
        return self._last + np.arange(1, steps + 1, dtype=float)


class RecordingModel:
    """Model recording the training data and every fit/forecast call."""

    train_indices: list = []
    init_kwargs: list = []
    fit_kwargs: list = []
    forecast_steps: list = []
    forecast_kwargs: list = []

    @classmethod
    def reset(cls):
        cls.train_indices.clear()
        cls.init_kwargs.clear()
        cls.fit_kwargs.clear()
        cls.forecast_steps.clear()
        cls.forecast_kwargs.clear()

    def __init__(self, endog, **kwargs):
        self._endog = endog
        index = getattr(endog, "index", pd.RangeIndex(len(endog)))
        type(self).train_indices.append(list(index))
        type(self).init_kwargs.append(kwargs)

    def fit(self, **kwargs):
        type(self).fit_kwargs.append(kwargs)
        return RecordingResults(self._endog)


@pytest.fixture
def backtest_data():
    rs = np.random.default_rng(20240521)
    nobs = 72
    index = pd.date_range("2000-01-01", periods=nobs, freq=MONTH_END)
    seasonal = 2.5 * np.sin(np.arange(nobs) * 2 * np.pi / 12)
    trend = 0.02 * np.arange(nobs)
    return pd.Series(seasonal + trend + rs.standard_normal(nobs), index=index)


def test_backtest_expanding(backtest_data):
    RecordingModel.reset()
    stlf = STLForecast(backtest_data, RecordingModel, period=12)
    bt = stlf.backtest(steps=4, initial=40, step=5)

    lengths = [len(idx) for idx in RecordingModel.train_indices]
    assert lengths == [40, 45, 50, 55, 60, 65]
    assert RecordingModel.train_indices[0] == list(backtest_data.index[:40])
    assert RecordingModel.train_indices[-1] == list(backtest_data.index[:65])
    assert RecordingModel.forecast_steps == [4] * 6

    frame = bt.frame
    assert frame.shape == (24, 3)
    assert list(frame.columns) == ["step", "actual", "forecast"]
    assert frame.index.names == ["origin", "target"]
    origins = backtest_data.index[[39, 44, 49, 54, 59, 64]]
    assert list(bt.origins) == list(origins)

    first = frame.xs(origins[0], level="origin")
    assert list(first.index) == list(backtest_data.index[40:44])
    assert list(first["step"]) == [1, 2, 3, 4]
    assert_allclose(first["actual"], backtest_data.iloc[40:44])

    stl_fit = STL(backtest_data.iloc[:40], period=12).fit()
    last = float((stl_fit.trend + stl_fit.resid).iloc[-1])
    cycle = np.asarray(stl_fit.seasonal)[-12:]
    expected = last + np.arange(1, 5) + cycle[:4]
    assert_allclose(first["forecast"], expected)


def test_backtest_fixed_window(backtest_data):
    RecordingModel.reset()
    stlf = STLForecast(backtest_data, RecordingModel, period=12)
    bt = stlf.backtest(steps=4, initial=40, step=5, window=20)

    lengths = [len(idx) for idx in RecordingModel.train_indices]
    assert lengths == [20] * 6
    assert RecordingModel.train_indices[0] == list(backtest_data.index[20:40])
    assert RecordingModel.train_indices[-1] == list(backtest_data.index[45:65])
    # Every origin still validates four observations after its training end.
    targets = bt.frame.index.get_level_values("target")
    expected = list(backtest_data.index[40:44]) + list(
        backtest_data.index[45:49]
    )
    expected += list(backtest_data.index[50:54])
    expected += list(backtest_data.index[55:59])
    expected += list(backtest_data.index[60:64])
    expected += list(backtest_data.index[65:69])
    assert list(targets) == expected


def test_backtest_gap(backtest_data):
    RecordingModel.reset()
    bt = STLForecast(backtest_data, RecordingModel, period=12).backtest(
        steps=3, initial=40, step=5, gap=2
    )
    # The model always forecasts gap + steps observations.
    assert RecordingModel.forecast_steps == [5] * 6

    frame = bt.frame
    first_origin = backtest_data.index[39]
    first = frame.xs(first_origin, level="origin")
    assert list(first.index) == list(backtest_data.index[42:45])
    assert_allclose(first["actual"], backtest_data.iloc[42:45])

    stl_fit = STL(backtest_data.iloc[:40], period=12).fit()
    last = float((stl_fit.trend + stl_fit.resid).iloc[-1])
    cycle = np.asarray(stl_fit.seasonal)[-12:]
    tiled = np.tile(cycle, 5 // 12 + 1)[:5]
    expected = last + np.arange(1, 6) + tiled
    assert_allclose(first["forecast"], expected[2:5])


@pytest.mark.parametrize("partial", ["drop", "keep"])
def test_backtest_partial_window(backtest_data, partial):
    data = backtest_data.iloc[:50]
    RecordingModel.reset()
    bt = STLForecast(data, RecordingModel, period=12).backtest(
        steps=8, initial=30, step=15, partial=partial
    )
    lengths = [len(idx) for idx in RecordingModel.train_indices]
    if partial == "drop":
        assert lengths == [30]
        assert bt.frame.shape[0] == 8
    else:
        assert lengths == [30, 45]
        assert bt.frame.shape[0] == 13
        last_origin = data.index[44]
        last = bt.frame.xs(last_origin, level="origin")
        assert list(last["step"]) == [1, 2, 3, 4, 5]
        assert_allclose(last["actual"], data.iloc[45:50])
        # The first origin covers all eight steps, so every MAE is finite.
        assert np.all(np.isfinite(bt.mae))
        assert np.all(np.isfinite(bt.rmse))


def test_backtest_partial_missing_horizons(backtest_data):
    # The first (and only) origin is itself incomplete: only five of the
    # eight requested forecasts have an actual value.
    data = backtest_data.iloc[:50]
    bt = STLForecast(data, RecordingModel, period=12).backtest(
        steps=8, initial=45, step=10, partial="keep"
    )
    assert bt.frame.shape[0] == 5
    assert np.all(np.isfinite(bt.mae.loc[:5]))
    assert np.all(np.isnan(bt.mae.loc[6:]))
    assert np.all(np.isnan(bt.rmse.loc[6:]))


def test_backtest_partial_no_origin(backtest_data):
    data = backtest_data.iloc[:50]
    stlf = STLForecast(data, RecordingModel, period=12)
    # No complete validation window exists.
    with pytest.raises(ValueError, match="too short"):
        stlf.backtest(steps=8, initial=45, step=10)
    # A single observation can still be validated when kept.
    keep = stlf.backtest(steps=8, initial=49, step=10, partial="keep")
    assert keep.frame.shape[0] == 1
    with pytest.raises(ValueError, match="too short"):
        stlf.backtest(steps=8, initial=50, step=10, partial="keep")


def test_backtest_no_future_leakage(backtest_data):
    RecordingModel.reset()
    stlf = STLForecast(backtest_data, RecordingModel, period=12)
    first_origin = backtest_data.index[39]
    first = stlf.backtest(steps=4, initial=40, step=10).frame.xs(
        first_origin, level="origin"
    )

    # The recorded training samples never extend beyond their origin.
    for end, labels in zip(
        [40, 50, 60], RecordingModel.train_indices, strict=False
    ):
        assert labels == list(backtest_data.index[:end])

    # Changing future observations does not affect earlier origins.
    altered = backtest_data.copy()
    altered.iloc[40:] = altered.iloc[40:] * 100.0 + 7.0
    RecordingModel.reset()
    first_altered = (
        STLForecast(altered, RecordingModel, period=12)
        .backtest(steps=4, initial=40, step=10)
        .frame.xs(first_origin, level="origin")
    )
    assert_allclose(first["forecast"], first_altered["forecast"])
    # The validation actuals are the (altered) values at the target dates.
    assert_allclose(first_altered["actual"], altered.iloc[40:44])
    assert_allclose(first["actual"], backtest_data.iloc[40:44])


def test_backtest_does_not_modify_model(backtest_data):
    stlf = STLForecast(
        backtest_data, ARIMA, model_kwargs={"order": (1, 0, 0)}, period=12
    )
    res = stlf.fit()
    before = np.asarray(res.forecast(5)).copy()
    stlf.backtest(steps=3, initial=60, step=5)
    after = np.asarray(res.forecast(5))
    assert_allclose(before, after)


def test_backtest_kwargs(backtest_data):
    RecordingModel.reset()
    stlf = STLForecast(
        backtest_data,
        RecordingModel,
        model_kwargs={"flag": 7},
        period=12,
    )
    stlf.backtest(
        steps=2,
        initial=60,
        step=10,
        inner_iter=1,
        outer_iter=0,
        fit_kwargs={"optimizer": "none"},
        forecast_kwargs={"future": 3},
    )
    assert RecordingModel.init_kwargs == [{"flag": 7}] * 2
    assert RecordingModel.fit_kwargs == [{"optimizer": "none"}] * 2
    assert RecordingModel.forecast_kwargs == [{"future": 3}] * 2
    assert RecordingModel.forecast_steps == [2, 2]


def test_backtest_metrics(backtest_data):
    bt = STLForecast(backtest_data, RecordingModel, period=12).backtest(
        steps=4, initial=40, step=5
    )
    frame = bt.frame
    errors = frame["actual"] - frame["forecast"]
    expected_mae = errors.abs().groupby(frame["step"]).mean()
    expected_rmse = np.sqrt(errors.pow(2).groupby(frame["step"]).mean())
    assert_allclose(bt.mae.values, expected_mae.loc[1:4].values)
    assert_allclose(bt.rmse.values, expected_rmse.loc[1:4].values)
    assert bt.mae.name == "MAE"
    assert bt.rmse.name == "RMSE"
    assert list(bt.mae.index) == [1, 2, 3, 4]

    text = bt.summary().as_text()
    assert "Rolling-Origin Backtest" in text
    assert "All" in text
    assert "MAE" in text
    assert "RMSE" in text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0, "initial": 40},
        {"steps": -1, "initial": 40},
        {"steps": 4.0, "initial": 40},
        {"steps": 4, "initial": 0},
        {"steps": 4, "initial": 40, "step": 0},
        {"steps": 4, "initial": 40, "gap": -1},
        {"steps": 4, "initial": 40, "window": 0},
        {"steps": 4, "initial": 20, "window": 30},
        {"steps": 4, "initial": 40, "partial": "other"},
        {"steps": 4, "initial": 72},
    ],
)
def test_backtest_validation(backtest_data, kwargs):
    stlf = STLForecast(backtest_data, RecordingModel, period=12)
    with pytest.raises(ValueError):
        stlf.backtest(**kwargs)


def test_backtest_date_initial(backtest_data):
    label = backtest_data.index[39]
    for initial in (label, label.strftime("%Y-%m-%d")):
        RecordingModel.reset()
        STLForecast(backtest_data, RecordingModel, period=12).backtest(
            steps=4, initial=initial, step=5
        )
        lengths = [len(idx) for idx in RecordingModel.train_indices]
        assert lengths == [40, 45, 50, 55, 60, 65]


def test_backtest_ndarray(backtest_data):
    values = backtest_data.to_numpy()
    bt = STLForecast(values, RecordingModel, period=12).backtest(
        steps=3, initial=40, step=10
    )
    frame = bt.frame
    assert frame.index.names == ["origin", "target"]
    assert list(bt.origins) == [39, 49, 59]
    first = frame.xs(39, level="origin")
    assert list(first.index) == [40, 41, 42]
    assert_allclose(first["actual"], values[40:43])
    assert np.all(np.isfinite(first["forecast"]))


def test_backtest_arima_equivalence(backtest_data):
    order = (2, 0, 0)
    stlf = STLForecast(
        backtest_data, ARIMA, model_kwargs={"order": order}, period=12
    )
    with warns_or_none(ConvergenceWarning, match="Maximum Likelihood"):
        bt = stlf.backtest(steps=5, initial=60, step=100)

    train = backtest_data.iloc[:60]
    stl_fit = STL(train, period=12).fit()
    mod = ARIMA(stl_fit.trend + stl_fit.resid, order=order)
    with warns_or_none(ConvergenceWarning, match="Maximum Likelihood"):
        res = mod.fit()
    cycle = np.tile(np.asarray(stl_fit.seasonal)[-12:], 1 + 5 // 12)[:5]
    expected = np.asarray(res.forecast(5)) + cycle

    assert bt.frame.shape[0] == 5
    assert list(bt.frame.index.get_level_values("target")) == list(
        backtest_data.index[60:65]
    )
    assert_allclose(bt.frame["forecast"], expected, rtol=1e-4)
    assert_allclose(bt.frame["actual"], backtest_data.iloc[60:65])
