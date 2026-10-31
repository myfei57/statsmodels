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

    Alternatively, ``forecast`` and ``get_prediction`` accept a deterministic
    ``seasonal_scenario`` that replaces these extrapolated seasonal values for
    the out-of-sample steps.

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
        self._index = getattr(endog, "index", pd.RangeIndex(self._nobs))
        if not (
            isinstance(self._index, (pd.DatetimeIndex, pd.PeriodIndex))
            or is_int_index(self._index)
        ):
            try:
                self._index = pd.to_datetime(self._index)
            except ValueError:
                self._index = pd.RangeIndex(self._nobs)

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
        seasonal_scenario: Any = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
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
        seasonal_scenario : array_like, pandas.Series, or None
            Deterministic seasonal values that replace the extrapolated
            seasonal component on the out-of-sample steps. If None, the
            seasonal component is extrapolated by repeating the final
            seasonal cycle.

        Returns
        -------
        predictions : ndarray
            Array containing the seasonal predictions.
        scenario : ndarray or None
            The validated out-of-sample seasonal scenario values, or None
            if no scenario was provided.
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
        scenario_values = None
        if seasonal_scenario is not None:
            # The scenario only replaces the true out-of-sample tail. The
            # leading entries are in-sample dates, including dates whose
            # seasonal values are produced by dynamic prediction.
            n_in_sample = max(nobs - start, 0)
            oos_index = (
                prediction_index[n_in_sample:] if prediction_index is not None else None
            )
            scenario_values = self._validate_seasonal_scenario(
                seasonal_scenario,
                oos_index,
                predictions.shape[0] - n_in_sample,
            )
            predictions[n_in_sample:] = scenario_values
        return predictions, scenario_values

    def _out_of_sample_index(self, steps: DateLike) -> pd.Index | None:
        """
        Get the index labels of the out-of-sample forecast steps

        Parameters
        ----------
        steps : int, str, or datetime
            The number of steps to forecast, or the label of the final
            out-of-sample observation.

        Returns
        -------
        pd.Index or None
            The index covering the out-of-sample forecast steps.
        """
        data = PandasData(pd.Series(self._endog), index=self._index)
        if isinstance(steps, (int, np.integer)) and not isinstance(steps, bool):
            end = self._nobs + int(steps) - 1
        else:
            end = steps
        _, _, _, prediction_index = get_prediction_index(
            self._nobs, end, self._nobs, self._index, data=data
        )
        return prediction_index

    def _validate_seasonal_scenario(
        self,
        seasonal_scenario: Any,
        forecast_index: pd.Index | None,
        steps: int,
    ) -> np.ndarray:
        """
        Validate a deterministic seasonal scenario

        Parameters
        ----------
        seasonal_scenario : array_like or pandas.Series
            Seasonal values for the out-of-sample forecast steps. A
            pandas Series is aligned using its index. Other array-like
            inputs are aligned positionally.
        forecast_index : pd.Index or None
            The labels of the out-of-sample forecast steps. Used to align
            pandas inputs. If None, pandas inputs cannot be label-aligned.
        steps : int
            The number of out-of-sample forecast steps.

        Returns
        -------
        ndarray
            The scenario values in the order of the forecast steps.

        Raises
        ------
        TypeError
            If the scenario does not contain numeric values.
        ValueError
            If the scenario is not one-dimensional, has missing values,
            has the wrong length, has a non-unique index, contains
            in-sample labels, or its index does not match the
            out-of-sample forecast index.
        """
        is_pandas = isinstance(seasonal_scenario, pd.Series)
        scenario_index = seasonal_scenario.index if is_pandas else None
        if is_pandas and not scenario_index.is_unique:
            raise ValueError(
                "The index of seasonal_scenario must contain unique labels."
            )
        try:
            values = np.asarray(seasonal_scenario, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("seasonal_scenario must contain numeric values.") from exc
        if values.ndim != 1:
            raise ValueError("seasonal_scenario must be one-dimensional.")
        if values.shape[0] != steps:
            raise ValueError(
                f"seasonal_scenario must contain exactly {steps} value(s), "
                f"one for each out-of-sample forecast step, but got "
                f"{values.shape[0]}."
            )
        if np.isnan(values).any():
            raise ValueError("seasonal_scenario must not contain missing values (NaN).")
        if is_pandas and forecast_index is not None:
            in_sample = scenario_index.isin(self._index)
            if in_sample.any():
                overlap = scenario_index[in_sample].tolist()
                raise ValueError(
                    "seasonal_scenario must only cover out-of-sample "
                    "observations, but its index contains in-sample "
                    f"labels: {overlap}."
                )
            missing = forecast_index.difference(scenario_index)
            unexpected = scenario_index.difference(forecast_index)
            if len(missing) or len(unexpected):
                raise ValueError(
                    "The index of seasonal_scenario must exactly match the "
                    "out-of-sample forecast index. Missing labels: "
                    f"{missing.tolist()}; unexpected labels: "
                    f"{unexpected.tolist()}."
                )
            # Reorder (or cast) to the forecast index while keeping the
            # values attached to their labels. The label sets match, so no
            # missing values can be introduced here.
            values = np.asarray(
                seasonal_scenario.reindex(forecast_index), dtype=np.float64
            )
        return values

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
        seasonal = seasonal[offset - period : offset]
        seasonal = np.tile(seasonal, steps // period + ((steps % period) != 0))
        seasonal = seasonal[:steps]
        if index is not None:
            seasonal = pd.Series(seasonal, index=index)
        return seasonal

    def forecast(
        self,
        steps: int = 1,
        seasonal_scenario: Any = None,
        **kwargs: dict[str, Any],
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
        seasonal_scenario : array_like or pandas.Series, optional
            Deterministic values of the seasonal component for the
            out-of-sample forecast steps. Must contain exactly one value
            per forecast step, and missing values are not allowed. If a
            pandas Series, its index must exactly match the index of the
            out-of-sample forecasts and must not contain in-sample
            labels. If None, the seasonal component is extrapolated by
            repeating the last seasonal cycle. The scenario is not passed
            to the underlying model fitted on the deseasonalized data.
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
        seasonal = self._seasonal_forecast(steps, index)
        if seasonal_scenario is not None:
            if index is None:
                index = self._out_of_sample_index(steps)
            scenario = self._validate_seasonal_scenario(
                seasonal_scenario, index, np.asarray(forecast).shape[0]
            )
            if isinstance(seasonal, pd.Series):
                seasonal = pd.Series(scenario, index=index)
            else:
                seasonal = scenario
        return forecast + seasonal

    def get_prediction(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
        dynamic: bool | DateLike = False,
        seasonal_scenario: Any = None,
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
        seasonal_scenario : array_like or pandas.Series, optional
            Deterministic values of the seasonal component for the
            out-of-sample forecast steps. Must contain exactly one value
            per out-of-sample step, and missing values are not allowed.
            If a pandas Series, its index must exactly match the index of
            the out-of-sample forecasts and must not contain in-sample
            labels. The scenario only replaces the out-of-sample
            seasonal values; in-sample predictions are unchanged. If
            None, the seasonal component is extrapolated by repeating
            the last seasonal cycle. The scenario is not passed to the
            underlying model fitted on the deseasonalized data.
        **kwargs
            Additional arguments may be required for forecasting beyond the
            end of the sample. These arguments are passed into the time
            series model results' ``get_prediction`` method.

        Returns
        -------
        PredictionResults
            PredictionResults instance containing in-sample predictions,
            out-of-sample forecasts, and prediction intervals. When
            ``seasonal_scenario`` is provided, the instance additionally
            stores the accepted scenario in its ``seasonal_scenario``
            attribute.

        Notes
        -----
        The values of ``seasonal_scenario`` are treated as fixed and known.
        The underlying model is still fit and used to forecast on the
        deseasonalized data, so its predicted mean and
        ``var_pred_mean`` are computed without any contribution from
        seasonal uncertainty. The scenario only shifts the predicted mean
        and the prediction intervals: their width is unchanged, and the
        intervals are conditional on the given seasonal scenario.
        """
        pred = self._model_result.get_prediction(
            start=start, end=end, dynamic=dynamic, **kwargs
        )
        seasonal_prediction, scenario = self._get_seasonal_prediction(
            start, end, dynamic, seasonal_scenario
        )
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
        results = PredictionResults(
            mean, var_pred_mean, dist="norm", row_labels=pred.row_labels
        )
        if scenario is not None:
            labels = results.row_labels
            if labels is not None:
                scenario_index = labels[len(labels) - scenario.shape[0] :]
                results.seasonal_scenario = pd.Series(
                    scenario, index=scenario_index, name="seasonal_scenario"
                )
            else:
                results.seasonal_scenario = scenario
        return results
