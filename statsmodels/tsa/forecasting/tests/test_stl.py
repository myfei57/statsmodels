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


HORIZON = 13


@pytest.fixture(scope="module")
def ets_result(data):
    stlf = STLForecast(data, ETSModel, model_kwargs={})
    return stlf.fit(fit_kwargs={"disp": False})


def test_forecast_seasonal_scenario(data, ets_result):
    horizon = HORIZON
    default = ets_result.forecast(horizon)
    tiled = np.asarray(ets_result._seasonal_forecast(horizon, None))
    rng = np.random.default_rng(1234)
    values = rng.normal(scale=2.0, size=horizon)
    expected = np.asarray(default) - tiled + values

    # Array and other array_like inputs are aligned positionally
    assert_allclose(ets_result.forecast(horizon, seasonal_scenario=values), expected)
    assert_allclose(
        ets_result.forecast(horizon, seasonal_scenario=list(values)), expected
    )

    # Pandas input aligns with the out-of-sample forecast index
    scenario = pd.Series(values, index=default.index)
    aligned = ets_result.forecast(horizon, seasonal_scenario=scenario)
    assert isinstance(aligned, pd.Series)
    assert_allclose(aligned, expected)
    assert aligned.index.equals(default.index)

    # Values follow their labels even if the order is reversed
    assert_allclose(
        ets_result.forecast(horizon, seasonal_scenario=scenario.iloc[::-1]),
        expected,
    )

    # A zero scenario leaves the deseasonalized model forecast
    assert_allclose(
        ets_result.forecast(horizon, seasonal_scenario=np.zeros(horizon)),
        ets_result.model_result.forecast(horizon),
        rtol=1e-10,
    )


def test_scenario_matching_default_tiling(ets_result):
    horizon = 25
    tiled = np.asarray(ets_result._seasonal_forecast(horizon, None))
    assert_allclose(
        ets_result.forecast(horizon, seasonal_scenario=tiled),
        ets_result.forecast(horizon),
    )


def test_get_prediction_seasonal_scenario(data, ets_result):
    nobs = data.shape[0]
    horizon = HORIZON
    start, end = nobs - 7, nobs + horizon - 1
    default = ets_result.get_prediction(start, end)
    rng = np.random.default_rng(4321)
    values = rng.normal(scale=2.0, size=horizon)
    scenario = pd.Series(values, index=default.row_labels[-horizon:])
    result = ets_result.get_prediction(start, end, seasonal_scenario=scenario)

    tiled = np.asarray(ets_result._seasonal_forecast(horizon, None))
    shift = values - tiled

    # Only the out-of-sample mean changes, and only by the seasonal delta
    assert_allclose(
        np.asarray(result.predicted_mean)[-horizon:]
        - np.asarray(default.predicted_mean)[-horizon:],
        shift,
    )
    assert_allclose(
        np.asarray(result.predicted_mean)[:-horizon],
        np.asarray(default.predicted_mean)[:-horizon],
    )
    # The variance is taken from the deseasonalized model unchanged
    assert_allclose(result.var_pred_mean, default.var_pred_mean)

    default_ci = default.conf_int()
    result_ci = result.conf_int()

    def width(ci):
        return np.asarray(ci["upper"] - ci["lower"])

    assert_allclose(width(result_ci), width(default_ci))
    upper_shift = np.asarray(result_ci["upper"] - default_ci["upper"])
    assert_allclose(upper_shift[:-horizon], 0.0)
    assert_allclose(upper_shift[-horizon:], shift)

    # The accepted scenario is stored and labelled
    assert isinstance(result.seasonal_scenario, pd.Series)
    assert result.seasonal_scenario.index.equals(default.row_labels[-horizon:])
    assert_allclose(result.seasonal_scenario, values)
    assert not hasattr(default, "seasonal_scenario")


def test_seasonal_scenario_dynamic(data, ets_result):
    nobs = data.shape[0]
    horizon = HORIZON
    start = nobs - 20
    end = nobs + horizon - 1
    default = ets_result.get_prediction(start, end, dynamic=True)
    values = np.linspace(-1.0, 1.0, horizon)
    result = ets_result.get_prediction(
        start, end, dynamic=True, seasonal_scenario=values
    )

    # In-sample dynamic predictions keep their tiled seasonal values
    assert_allclose(
        np.asarray(result.predicted_mean)[:-horizon],
        np.asarray(default.predicted_mean)[:-horizon],
    )
    assert not np.allclose(
        np.asarray(result.predicted_mean)[-horizon:],
        np.asarray(default.predicted_mean)[-horizon:],
    )
    assert_allclose(result.var_pred_mean, default.var_pred_mean)
    assert len(result.seasonal_scenario) == horizon


def test_seasonal_scenario_validation(data, ets_result):
    nobs = data.shape[0]
    horizon = HORIZON
    oos_index = ets_result.forecast(horizon).index
    rng = np.random.default_rng(7)
    values = rng.standard_normal(horizon)

    with pytest.raises(ValueError, match="exactly"):
        ets_result.forecast(horizon, seasonal_scenario=values[:-1])
    missing = values.copy()
    missing[3] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        ets_result.forecast(horizon, seasonal_scenario=missing)
    with pytest.raises(ValueError, match="one-dimensional"):
        ets_result.forecast(horizon, seasonal_scenario=np.zeros((horizon, 2)))
    with pytest.raises(TypeError, match="numeric"):
        ets_result.forecast(horizon, seasonal_scenario=["x"] * horizon)
    with pytest.raises(ValueError, match="in-sample"):
        ets_result.forecast(
            horizon,
            seasonal_scenario=pd.Series(values, index=data.index[-horizon:]),
        )
    wrong_labels = pd.DatetimeIndex(
        list(oos_index[1:]) + [oos_index[-1] + pd.offsets.MonthEnd(1)]
    )
    with pytest.raises(ValueError, match="exactly match"):
        ets_result.forecast(
            horizon, seasonal_scenario=pd.Series(values, index=wrong_labels)
        )
    duplicate_labels = pd.DatetimeIndex(list(oos_index[:-1]) + [oos_index[0]])
    with pytest.raises(ValueError, match="unique"):
        ets_result.forecast(
            horizon,
            seasonal_scenario=pd.Series(values, index=duplicate_labels),
        )

    start, end = nobs - 5, nobs + horizon - 1
    with pytest.raises(ValueError, match="exactly"):
        ets_result.get_prediction(start, end, seasonal_scenario=values[:-1])
    with pytest.raises(ValueError, match="in-sample"):
        ets_result.get_prediction(
            start,
            end,
            seasonal_scenario=pd.Series(values, index=data.index[-horizon:]),
        )
    # A scenario for an in-sample-only prediction must have zero length
    with pytest.raises(ValueError, match="exactly 0"):
        ets_result.get_prediction(0, nobs - 1, seasonal_scenario=values)


def test_seasonal_scenario_not_passed_to_model(data):
    received = {}

    class RecordingResults:
        def __init__(self, results):
            self._results = results

        def forecast(self, steps=1, **kwargs):
            received["forecast"] = kwargs
            return self._results.forecast(steps=steps, **kwargs)

        def get_prediction(self, *args, **kwargs):
            received["get_prediction"] = kwargs
            return self._results.get_prediction(*args, **kwargs)

        def __getattr__(self, item):
            return getattr(self._results, item)

    class RecordingModel:
        def __init__(self, endog):
            self._model = ETSModel(endog)

        def fit(self, *args, **kwargs):
            return RecordingResults(self._model.fit(*args, **kwargs))

    res = STLForecast(data, RecordingModel, period=12).fit(fit_kwargs={"disp": False})
    horizon = 5
    values = np.zeros(horizon)
    res.forecast(horizon, seasonal_scenario=values)
    assert "seasonal_scenario" not in received["forecast"]
    nobs = data.shape[0]
    res.get_prediction(nobs, nobs + horizon - 1, seasonal_scenario=values)
    assert "seasonal_scenario" not in received["get_prediction"]


def test_seasonal_scenario_ndarray_endog(data):
    endog = np.asarray(data)
    nobs = endog.shape[0]
    res = STLForecast(endog, ETSModel, model_kwargs={}, period=12).fit(
        fit_kwargs={"disp": False}
    )
    horizon = 7
    values = np.linspace(0.5, 1.5, horizon)
    got = res.forecast(horizon, seasonal_scenario=values)
    assert isinstance(got, np.ndarray)
    assert_allclose(got, np.asarray(res.model_result.forecast(horizon)) + values)

    # A Series aligned to the integer labels of the out-of-sample steps
    labelled = pd.Series(values, index=pd.RangeIndex(nobs, nobs + horizon))
    assert_allclose(res.forecast(horizon, seasonal_scenario=labelled), got)

    # A default RangeIndex starts at zero and therefore covers in-sample
    with pytest.raises(ValueError, match="in-sample"):
        res.forecast(horizon, seasonal_scenario=pd.Series(values))


def test_seasonal_scenario_period_index(data):
    period_data = data.to_period(freq="M")
    res = STLForecast(period_data, ETSModel, model_kwargs={}, period=12).fit(
        fit_kwargs={"disp": False}
    )
    horizon = 6
    default = res.forecast(horizon)
    values = np.linspace(-2.0, 2.0, horizon)
    got = res.forecast(
        horizon, seasonal_scenario=pd.Series(values, index=default.index)
    )
    assert isinstance(got.index, pd.PeriodIndex)
    expected = np.asarray(res.model_result.forecast(horizon)) + values
    assert_allclose(np.asarray(got), expected)
    with pytest.raises(ValueError, match="in-sample"):
        res.forecast(
            horizon,
            seasonal_scenario=pd.Series(values, index=period_data.index[-horizon:]),
        )


def test_seasonal_scenario_leaves_fitted_component(ets_result):
    before = np.asarray(ets_result.result.seasonal).copy()
    ets_result.forecast(HORIZON, seasonal_scenario=np.ones(HORIZON))
    ets_result.get_prediction(490, 499 + HORIZON, seasonal_scenario=np.ones(HORIZON))
    assert_allclose(np.asarray(ets_result.result.seasonal), before)
