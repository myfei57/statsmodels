from __future__ import annotations

from statsmodels.compat.pandas import Substitution, is_int_index

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from statsmodels.base.data import PandasData
from statsmodels.iolib.summary import SimpleTable, Summary
from statsmodels.tools.docstring import Docstring, Parameter, indent
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.base.tsa_model import get_index_loc, get_prediction_index
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.statespace.kalman_filter import _check_dynamic

DateLike = int | str | dt.datetime | pd.Timestamp | np.datetime64

ds = Docstring(STL.__doc__)
ds.insert_parameters(
    "endog",
    Parameter(
        "model",
        "Model",
        [
            (
                "The model used to forecast endog after the seasonality has been "
                "removed using STL"
            )
        ],
    ),
)
ds.insert_parameters(
    "model",
    Parameter(
        "model_kwargs",
        "dict, optional",
        [
            (
                "Any additional arguments needed to initialized the model using "
                "the residuals produced by subtracting the seasonality."
            )
        ],
    ),
)
_stl_forecast_params = ds.extract_parameters(
    [
        "endog",
        "model",
        "model_kwargs",
        "period",
        "seasonal",
        "trend",
        "low_pass",
        "seasonal_deg",
        "trend_deg",
        "low_pass_deg",
        "robust",
        "seasonal_jump",
        "trend_jump",
        "low_pass_jump",
    ]
)

ds = Docstring(STL.fit.__doc__)
_fit_params = ds.extract_parameters(["inner_iter", "outer_iter"])


def _get_endog_index(endog: Any, nobs: int) -> pd.Index:
    """Normalize the row labels of an endog object to a pandas index."""
    index = getattr(endog, "index", pd.RangeIndex(nobs))
    if not (
        isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex))
        or is_int_index(index)
    ):
        try:
            index = pd.to_datetime(index)
        except ValueError:
            index = pd.RangeIndex(nobs)
    return index


def _seasonal_forecast(seasonal: np.ndarray, steps: int) -> np.ndarray:
    """Extend a single full seasonal cycle forward using a random walk."""
    period = seasonal.shape[0]
    seasonal = np.tile(seasonal, steps // period + ((steps % period) != 0))
    return seasonal[:steps]


@Substitution(stl_forecast_params=indent(_stl_forecast_params, "    "))
class STLForecast:
    r"""
    Model-based forecasting using STL to remove seasonality

    Forecasts are produced by first subtracting the seasonality
    estimated using STL, then forecasting the deseasonalized
    data using a time-series model, for example, ARIMA.

    Parameters
    ----------
%(stl_forecast_params)s

    See Also
    --------
    statsmodels.tsa.arima.model.ARIMA
        ARIMA modeling.
    statsmodels.tsa.ar_model.AutoReg
        Autoregressive modeling supporting complex deterministics.
    statsmodels.tsa.exponential_smoothing.ets.ETSModel
        Additive and multiplicative exponential smoothing with trend.
    statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothing
        Additive exponential smoothing with trend.

    Notes
    -----
    If :math:`\hat{S}_t` is the seasonal component, then the deseasonalize
    series is constructed as

    .. math::

        Y_t - \hat{S}_t

    The trend component is not removed, and so the time series model should
    be capable of adequately fitting and forecasting the trend if present. The
    out-of-sample forecasts of the seasonal component are produced as

    .. math::

        \hat{S}_{T + h} = \hat{S}_{T - k}

    where :math:`k = m - h + m \lfloor (h-1)/m \rfloor` tracks the period
    offset in the full cycle of 1, 2, ..., m where m is the period length.

    This class is mostly a convenience wrapper around ``STL`` and a
    user-specified model. The model is assumed to follow the standard
    statsmodels pattern:

    * ``fit`` is used to estimate parameters and returns a results instance,
      ``results``.
    * ``results`` must expose a method ``forecast(steps, **kwargs)`` that
      produces out-of-sample forecasts.
    * ``results`` may also expose a method ``get_prediction`` that produces
      both in- and out-of-sample predictions.

    See the notebook `Seasonal Decomposition
    <../examples/notebooks/generated/stl_decomposition.html>`__ for an
    overview.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from statsmodels.tsa.api import STLForecast
    >>> from statsmodels.tsa.arima.model import ARIMA
    >>> from statsmodels.datasets import macrodata
    >>> ds = macrodata.load_pandas()
    >>> data = np.log(ds.data.m1)
    >>> base_date = f"{int(ds.data.year[0])}-{3*int(ds.data.quarter[0])+1}-1"
    >>> data.index = pd.date_range(base_date, periods=data.shape[0], freq="QS")

    Generate forecasts from an ARIMA

    >>> stlf = STLForecast(data, ARIMA, model_kwargs={"order": (2, 1, 0)})
    >>> res = stlf.fit()
    >>> forecasts = res.forecast(12)

    Generate forecasts from an Exponential Smoothing model with trend

    >>> from statsmodels.tsa.statespace import exponential_smoothing
    >>> ES = exponential_smoothing.ExponentialSmoothing
    >>> config = {"trend": True}
    >>> stlf = STLForecast(data, ES, model_kwargs=config)
    >>> res = stlf.fit()
    >>> forecasts = res.forecast(12)
    """

    def __init__(
        self,
        endog,
        model,
        *,
        model_kwargs=None,
        period=None,
        seasonal=7,
        trend=None,
        low_pass=None,
        seasonal_deg=1,
        trend_deg=1,
        low_pass_deg=1,
        robust=False,
        seasonal_jump=1,
        trend_jump=1,
        low_pass_jump=1,
    ):
        self._endog = endog
        self._stl_kwargs = {
            "period": period,
            "seasonal": seasonal,
            "trend": trend,
            "low_pass": low_pass,
            "seasonal_deg": seasonal_deg,
            "trend_deg": trend_deg,
            "low_pass_deg": low_pass_deg,
            "robust": robust,
            "seasonal_jump": seasonal_jump,
            "trend_jump": trend_jump,
            "low_pass_jump": low_pass_jump,
        }
        self._model = model
        self._model_kwargs = {} if model_kwargs is None else model_kwargs
        if not hasattr(model, "fit"):
            raise AttributeError("model must expose a ``fit``  method.")

    @Substitution(fit_params=indent(_fit_params, " " * 8))
    def fit(self, *, inner_iter=None, outer_iter=None, fit_kwargs=None):
        """
        Estimate STL and forecasting model parameters

        Parameters
        ----------\n%(fit_params)s
        fit_kwargs : dict, optional
            Any additional keyword arguments to pass to ``model``'s ``fit``
            method when estimating the model on the decomposed residuals.

        Returns
        -------
        STLForecastResults
            Results with forecasting methods.
        """
        fit_kwargs = {} if fit_kwargs is None else fit_kwargs
        stl = STL(self._endog, **self._stl_kwargs)
        stl_fit: DecomposeResult = stl.fit(inner_iter=inner_iter, outer_iter=outer_iter)
        model_endog = stl_fit.trend + stl_fit.resid
        mod = self._model(model_endog, **self._model_kwargs)
        res = mod.fit(**fit_kwargs)
        if not hasattr(res, "forecast"):
            raise AttributeError(
                "The model's result must expose a ``forecast`` method."
            )
        return STLForecastResults(stl, stl_fit, mod, res, self._endog)

    def backtest(
        self,
        steps: int,
        initial: int | DateLike,
        *,
        step: int = 1,
        window: int | None = None,
        gap: int = 0,
        partial: str = "drop",
        inner_iter: int | None = None,
        outer_iter: int | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        forecast_kwargs: dict[str, Any] | None = None,
    ) -> STLForecastBacktestResults:
        """
        Rolling-origin backtest of the STL-based forecast

        At every origin, a fresh ``STL`` decomposition is computed using
        only the observations that are available up to the origin, and a
        fresh copy of the underlying model is fit to the decomposed data.
        The decomposition is never performed on the complete series, so
        observations in the validation interval cannot leak into the fit
        through the STL trend or seasonal component. This model instance
        and any of its previously fitted results are not modified.

        Parameters
        ----------
        steps : int
            The forecast horizon, i.e. the number of validated
            observations after the gap at each origin. The model is
            always asked for ``gap + steps`` forecasts, and the first
            ``gap`` forecasts are discarded.
        initial : int or date-like
            The length of the training sample at the first origin. If an
            integer, the number of observations from the start of the
            series. A date-like label (str, datetime, Timestamp or
            Period) is interpreted as the label of the last training
            observation, and so requires a date-indexed ``endog``.
        step : int, optional
            The number of observations between two consecutive origins.
            Default is 1, which evaluates every origin.
        window : int, optional
            Fixed length of the (rolling) training window used at every
            origin. If None (the default), the training sample is
            expanding and contains all observations up to the origin.
        gap : int, optional
            The number of observations between the end of the training
            sample and the beginning of the validation interval. The
            model still forecasts ``gap + steps`` observations, and the
            validation values are the final ``steps`` forecasts.
            Default is 0.
        partial : {"drop", "keep"}, optional
            Whether to evaluate the final origin when its validation
            interval extends beyond the observed sample. "drop"
            (the default) skips origins without a complete validation
            interval, while "keep" retains the observations that are
            available.
        inner_iter : int, optional
            Number of inner iterations to pass to each ``STL.fit``.
            If None, the STL default is used at every origin.
        outer_iter : int, optional
            Number of outer iterations to pass to each ``STL.fit``.
            If None, the STL default is used at every origin.
        fit_kwargs : dict, optional
            Keyword arguments passed to the underlying model's ``fit``
            method at every origin. If None, no keyword arguments are
            passed.
        forecast_kwargs : dict, optional
            Keyword arguments passed to the fitted model's ``forecast``
            method at every origin. If None, no keyword arguments are
            passed.

        Returns
        -------
        STLForecastBacktestResults
            Results containing one row per origin and forecast step,
            the actual and forecast values, and step-wise MAE and RMSE.

        Raises
        ------
        ValueError
            If an integer argument is outside its valid range, if
            ``partial`` is not "drop" or "keep", if the training window
            is longer than the initial sample, or if not a single
            origin with a usable validation interval exists.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from statsmodels.tsa.api import STLForecast
        >>> from statsmodels.tsa.ar_model import AutoReg
        >>> rs = np.random.default_rng(0)
        >>> index = pd.date_range("2000-01-01", periods=96, freq="MS")
        >>> y = pd.Series(np.sin(np.arange(96) * 2 * np.pi / 12)
        ...               + rs.standard_normal(96), index=index)
        >>> stlf = STLForecast(y, AutoReg, model_kwargs={"lags": 2}, period=12)
        >>> bt = stlf.backtest(steps=6, initial=60, step=6, gap=1)
        >>> bt.rmse
        step
        1    ...
        2    ...
        3    ...
        4    ...
        5    ...
        6    ...
        Name: RMSE, dtype: float64
        """
        if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool):
            raise ValueError("steps must be a positive integer.")
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be a positive integer.")
        if not isinstance(step, (int, np.integer)) or isinstance(step, bool):
            raise ValueError("step must be a positive integer.")
        step = int(step)
        if step < 1:
            raise ValueError("step must be a positive integer.")
        if not isinstance(gap, (int, np.integer)) or isinstance(gap, bool):
            raise ValueError("gap must be a non-negative integer.")
        gap = int(gap)
        if gap < 0:
            raise ValueError("gap must be a non-negative integer.")
        if partial not in ("drop", "keep"):
            raise ValueError('partial must be either "drop" or "keep".')
        if window is not None:
            if not isinstance(window, (int, np.integer)) or isinstance(
                window, bool
            ):
                raise ValueError("window must be a positive integer or None.")
            window = int(window)
            if window < 1:
                raise ValueError("window must be a positive integer or None.")

        endog = self._endog
        values = np.asarray(endog)
        nobs = values.shape[0]
        index = _get_endog_index(endog, nobs)
        if isinstance(initial, (int, np.integer)) and not isinstance(
            initial, bool
        ):
            initial = int(initial)
            if initial < 1:
                raise ValueError("initial must be a positive integer.")
        else:
            loc, _, _ = get_index_loc(initial, index)
            initial = loc + 1
        if window is not None and window > initial:
            raise ValueError(
                "window must not be larger than the initial training length."
            )

        # Positions immediately after the last training observation.
        origin_ends = list(range(initial, nobs - gap - steps + 1, step))
        next_end = initial + len(origin_ends) * step
        if partial == "keep" and next_end + gap < nobs:
            origin_ends.append(next_end)
        if not origin_ends:
            what = "complete" if partial == "drop" else "usable"
            raise ValueError(
                f"The series is too short for a {what} validation interval "
                "given the initial length, gap and steps values."
            )

        fit_kwargs = {} if fit_kwargs is None else fit_kwargs
        forecast_kwargs = {} if forecast_kwargs is None else forecast_kwargs
        blocks = []
        for train_end in origin_ends:
            train_start = 0 if window is None else train_end - window
            train = endog[train_start:train_end]
            stl = STL(train, **self._stl_kwargs)
            stl_fit = stl.fit(inner_iter=inner_iter, outer_iter=outer_iter)
            model_endog = stl_fit.trend + stl_fit.resid
            mod = self._model(model_endog, **self._model_kwargs)
            mod_res = mod.fit(**fit_kwargs)
            point = np.asarray(
                mod_res.forecast(steps=gap + steps, **forecast_kwargs)
            )
            if point.shape[0] != gap + steps:
                raise ValueError(
                    "The model's forecast method did not return the "
                    f"requested {gap + steps} forecasts."
                )
            period = stl.period
            train_len = train_end - train_start
            cycle = np.asarray(stl_fit.seasonal)[train_len - period :]
            point = point + _seasonal_forecast(cycle, gap + steps)

            available = min(steps, nobs - train_end - gap)
            target_loc = np.arange(
                train_end + gap, train_end + gap + available
            )
            block = pd.DataFrame(
                {
                    "step": np.arange(1, available + 1, dtype=np.int64),
                    "actual": values[target_loc],
                    "forecast": point[gap : gap + available],
                },
                index=pd.MultiIndex.from_arrays(
                    [
                        np.full(available, index[train_end - 1]),
                        index[target_loc],
                    ],
                    names=["origin", "target"],
                ),
            )
            blocks.append(block)
        frame = pd.concat(blocks)
        return STLForecastBacktestResults(frame, steps=steps, gap=gap)


class STLForecastBacktestResults:
    """
    Results of a rolling-origin backtest of an ``STLForecast``

    Parameters
    ----------
    frame : DataFrame
        Long-format data frame with a MultiIndex named "origin" and
        "target", containing the label of the last training observation
        and the label of the validated observation. The columns are
        "step" (1-based forecast horizon within the validation
        interval), "actual" and "forecast".
    steps : int
        The requested forecast horizon.
    gap : int
        The number of observations skipped between the training sample
        and the validation interval.

    Notes
    -----
    Accuracy metrics are computed from the "actual" and "forecast"
    columns of :attr:`frame`. When a partial final window was kept,
    horizons without any validated observation report ``NaN``.
    """

    def __init__(self, frame: pd.DataFrame, steps: int, gap: int) -> None:
        self._frame = frame
        self._steps = steps
        self._gap = gap

    @property
    def frame(self) -> pd.DataFrame:
        """The actual and forecast values for every origin and step."""
        return self._frame

    @property
    def steps(self) -> int:
        """The requested forecast horizon."""
        return self._steps

    @property
    def gap(self) -> int:
        """The gap between the training sample and the validation interval."""
        return self._gap

    @property
    def origins(self) -> pd.Index:
        """The label of the last training observation at every origin."""
        return self._frame.index.get_level_values("origin").unique()

    def _errors(self) -> pd.Series:
        return self._frame["actual"] - self._frame["forecast"]

    @property
    def mae(self) -> pd.Series:
        """Mean absolute error grouped by forecast step."""
        errors = self._errors()
        mae = errors.abs().groupby(self._frame["step"]).mean()
        mae = mae.reindex(np.arange(1, self._steps + 1))
        mae.name = "MAE"
        return mae

    @property
    def rmse(self) -> pd.Series:
        """Root mean squared error grouped by forecast step."""
        errors = self._errors()
        rmse = np.sqrt(errors.pow(2).groupby(self._frame["step"]).mean())
        rmse = rmse.reindex(np.arange(1, self._steps + 1))
        rmse.name = "RMSE"
        return rmse

    def summary(self) -> Summary:
        """
        Step-wise backtest accuracy

        Returns
        -------
        Summary
            A table with the number of origins and the MAE and RMSE at
            every forecast step, followed by a row pooling all steps.
        """

        def fmt(value: float) -> str:
            return "" if np.isnan(value) else f"{value:8.6f}"

        mae = self.mae
        rmse = self.rmse
        counts = self._frame.groupby("step").size()
        counts = counts.reindex(np.arange(1, self._steps + 1)).fillna(0)
        data = []
        for horizon in range(1, self._steps + 1):
            data.append(
                [
                    str(horizon),
                    f"{int(counts.loc[horizon]):d}",
                    fmt(mae.loc[horizon]),
                    fmt(rmse.loc[horizon]),
                ]
            )
        errors = self._errors()
        data.append(
            [
                "All",
                f"{errors.shape[0]:d}",
                fmt(errors.abs().mean()),
                fmt(np.sqrt(errors.pow(2).mean())),
            ]
        )
        table = SimpleTable(
            data,
            headers=("Step", "Origins", "MAE", "RMSE"),
            title="STL Forecast Rolling-Origin Backtest",
        )
        summary = Summary()
        summary.tables.append(table)
        return summary


class STLForecastResults:
    """
    Results for forecasting using STL to remove seasonality

    Parameters
    ----------
    stl : STL
        The STL instance used to decompose the data.
    result : DecomposeResult
        The result of applying STL to the data.
    model : Model
        The time series model used to model the non-seasonal dynamics.
    model_result : Results
        Model results instance supporting, at a minimum, ``forecast``.
    endog : array_like
        The original data prior to seasonal decomposition.
    """

    def __init__(
        self, stl: STL, result: DecomposeResult, model, model_result, endog
    ) -> None:
        self._stl = stl
        self._result = result
        self._model = model
        self._model_result = model_result
        self._endog = np.asarray(endog)
        self._nobs = self._endog.shape[0]
        self._index = _get_endog_index(endog, self._nobs)

    @property
    def period(self) -> int:
        """The period of the seasonal component"""
        return self._stl.period

    @property
    def stl(self) -> STL:
        """The STL instance used to decompose the time series"""
        return self._stl

    @property
    def result(self) -> DecomposeResult:
        """The result of applying STL to the data"""
        return self._result

    @property
    def model(self) -> Any:
        """The model fit to the additively deseasonalized data"""
        return self._model

    @property
    def model_result(self) -> Any:
        """The result class from the estimated model"""
        return self._model_result

    def summary(self) -> Summary:
        """
        Summary of both the STL decomposition and the model fit

        Returns
        -------
        Summary
            The summary of the model fit and the STL decomposition.

        Notes
        -----
        Requires that the model's result class supports ``summary`` and
        returns a ``Summary`` object.
        """
        if not hasattr(self._model_result, "summary"):
            raise AttributeError("The model result does not have a summary attribute.")
        summary: Summary = self._model_result.summary()
        if not isinstance(summary, Summary):
            raise TypeError("The model result's summary is not a Summary object.")
        summary.tables[0].title = "STL Decomposition and " + summary.tables[0].title
        config = self._stl.config
        left_keys = ("period", "seasonal", "robust")
        left_data = []
        left_stubs = []
        right_data = []
        right_stubs = []
        for key in config:
            new = key.capitalize()
            new = new.replace("_", " ")
            if new in ("Trend", "Low Pass"):
                new += " Length"
            is_left = any(key.startswith(val) for val in left_keys)
            new += ":"
            stub = f"{new:<23s}"
            val = f"{config[key]!s:>13s}"
            if is_left:
                left_stubs.append(stub)
                left_data.append([val])
            else:
                right_stubs.append(" " * 6 + stub)
                right_data.append([val])
        delta = len(left_data) - len(right_data)
        adelta = abs(delta)
        if delta < 0:
            left_data += [[" " * len(left_data[0][0])] * adelta]
            left_stubs += [" " * len(left_stubs[0])] * adelta
        elif delta > 0:
            right_data += [[" " * len(right_data[0][0])] * adelta]
            right_stubs += [" " * len(right_stubs[0])] * adelta
        tab = SimpleTable(left_data, stubs=tuple(left_stubs), title="STL Configuration")
        right_data += [[""]] * (len(left_data) - len(right_data))
        tab.extend_right(SimpleTable(right_data, stubs=right_stubs))
        summary.tables.append(tab)
        return summary

    def _get_seasonal_prediction(
        self,
        start: DateLike | None,
        end: DateLike | None,
        dynamic: bool | DateLike,
    ) -> np.ndarray:
        """
        Get STL's seasonal in- and out-of-sample predictions

        Parameters
        ----------
        start : int, str, datetime, or None
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. If None, the zeroth observation is
            used.
        end : int, str, datetime, or None
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. If None, the last observation in
            the sample is used.
        dynamic : bool, int, str, or datetime
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.

        Returns
        -------
        ndarray
            Array containing the seasonal predictions.
        """
        data = PandasData(pd.Series(self._endog), index=self._index)
        if start is None:
            start = 0
        start, end, out_of_sample, prediction_index = get_prediction_index(
            start, end, self._nobs, self._index, data=data
        )

        if isinstance(dynamic, (str, dt.datetime, pd.Timestamp)):
            dynamic, _, _ = get_index_loc(dynamic, self._index)
            dynamic = dynamic - start
        elif dynamic is True:
            dynamic = 0
        elif dynamic is False:
            # If `dynamic=False`, then no dynamic predictions
            dynamic = None
        nobs = self._nobs
        dynamic, _ = _check_dynamic(dynamic, start, end, nobs)
        in_sample_end = end + 1 if dynamic is None else dynamic
        seasonal = np.asarray(self._result.seasonal)
        predictions = seasonal[start:in_sample_end]
        oos = np.empty((0,))
        if dynamic is not None:
            num = out_of_sample + end + 1 - dynamic
            oos = self._seasonal_forecast(num, None, offset=dynamic)
        elif out_of_sample:
            oos = self._seasonal_forecast(out_of_sample, None)
            oos_start = max(start - nobs, 0)
            oos = oos[oos_start:]
        predictions = np.r_[predictions, oos]
        return predictions

    def _seasonal_forecast(
        self, steps: int, index: pd.Index | None, offset=None
    ) -> pd.Series | np.ndarray:
        """
        Get the seasonal component of the forecast

        Parameters
        ----------
        steps : int
            The number of steps required.
        index : pd.Index or None
            A pandas index to use. If None, returns an ndarray.
        offset : int or None, optional
            The index of the first out-of-sample observation. If None, uses
            nobs.

        Returns
        -------
        seasonal : ndarray or Series
            The seasonal component.
        """

        period = self.period
        seasonal = np.asarray(self._result.seasonal)
        offset = self._nobs if offset is None else offset
        seasonal = _seasonal_forecast(seasonal[offset - period : offset], steps)
        if index is not None:
            seasonal = pd.Series(seasonal, index=index)
        return seasonal

    def forecast(
        self, steps: int = 1, **kwargs: dict[str, Any]
    ) -> np.ndarray | pd.Series:
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default is 1.
        **kwargs
            Additional arguments may be required for forecasting beyond the
            end of the sample. These arguments are passed into the time
            series model results' ``forecast`` method.

        Returns
        -------
        forecast : ndarray or Series
            Out of sample forecasts
        """
        forecast = self._model_result.forecast(steps=steps, **kwargs)
        index = forecast.index if isinstance(forecast, pd.Series) else None
        return forecast + self._seasonal_forecast(steps, index)

    def get_prediction(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
        dynamic: bool | DateLike = False,
        **kwargs: dict[str, Any],
    ):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        **kwargs
            Additional arguments may be required for forecasting beyond the
            end of the sample. These arguments are passed into the time
            series model results' ``get_prediction`` method.

        Returns
        -------
        PredictionResults
            PredictionResults instance containing in-sample predictions,
            out-of-sample forecasts, and prediction intervals.
        """
        pred = self._model_result.get_prediction(
            start=start, end=end, dynamic=dynamic, **kwargs
        )
        seasonal_prediction = self._get_seasonal_prediction(start, end, dynamic)
        mean = pred.predicted_mean + seasonal_prediction
        try:
            var_pred_mean = pred.var_pred_mean
        except (AttributeError, NotImplementedError):
            # Allow models that do not return var_pred_mean
            import warnings

            warnings.warn(
                "The variance of the predicted mean is not available using "
                f"the {self.model.__class__.__name__} model class.",
                UserWarning,
                stacklevel=2,
            )
            var_pred_mean = np.nan + mean.copy()
        return PredictionResults(
            mean, var_pred_mean, dist="norm", row_labels=pred.row_labels
        )
