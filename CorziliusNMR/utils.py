"""
Fitting module for spectral analysis.

This module provides classes for fitting spectral data using the `lmfit` package.
It includes different types of fitters, such as `Fitter`, `Prefitter`, `GlobalFitter`, and `SingleFitter`.

Classes:
    Fitter: Base class for fitting spectral data.
    Prefitter: A specialized fitter that only fits a preselected spectrum.
    GlobalFitter: A fitter that enforces parameter constraints across multiple spectra.
    SingleFitter: A simple extension of `Fitter` with no additional functionality.
"""

import lmfit
import re
import numpy as np
import copy
import matplotlib.pyplot as plt
from pyDOE2 import lhs
from CorziliusNMR import functions


class Fitter:
    """
    Base class for spectral fitting using `lmfit`.

    This class handles parameter initialization and spectral fitting for a dataset.

    Attributes:
        dataset: The dataset containing spectra and peak information.
    """

    def __init__(self, dataset):
        """
        Initializes the Fitter with a dataset.

        Args:
            dataset: An object containing spectral data and peak list.
        """
        self.dataset = dataset

    def fit(self):
        """
        Performs spectral fitting using the `lmfit.minimize` function.

        Returns:
            lmfit.MinimizerResult: The result of the fitting process.
        """
        x_axis, y_axis = self._generate_axis_list()
        params = self._generate_params_list()
        params = self._set_param_expr(params)
        return lmfit.minimize(
            self._spectral_fitting, params, args=(x_axis, y_axis)
        )

    def _set_param_expr(self, params):
        """
        Modifies parameter expressions if needed. Default implementation returns parameters unchanged.

        Args:
            params (lmfit.Parameters): The parameters to be modified.

        Returns:
            lmfit.Parameters: The modified parameters.
        """
        return params

    def _generate_axis_list(self):
        """
        Generates lists of x-axis and y-axis values for all spectra in the dataset.

        Returns:
            tuple: Two lists containing x-axis and y-axis values for each spectrum.
        """
        x_axis, y_axis = [], []
        for spectrum in self.dataset.spectra:
            x_axis.append(spectrum.x_axis)
            y_axis.append(spectrum.y_axis)
        return x_axis, y_axis

    def _generate_params_list(self):
        """
        Generates initial fitting parameters based on peak information in the dataset.

        Returns:
            lmfit.Parameters: The initialized parameters for fitting.
        """
        params = lmfit.Parameters()
        spectra = self._get_spectra_list()
        lw_types = {
            "voigt": ["sigma", "gamma"],
            "gauss": ["sigma"],
            "lorentz": ["gamma"],
        }
        for spectrum_nr, _ in enumerate(spectra):
            for peak in self.dataset.peak_list:
                params.add(**self._get_amplitude_dict(peak, spectrum_nr))
                params.add(**self._get_center_dict(peak, spectrum_nr))

                for lw_type in lw_types.get(peak.fitting_type, []):
                    params.add(
                        **self._get_lw_dict(peak, spectrum_nr, lw_type)
                    )
        return params

    def _get_spectra_list(self):
        """
        Retrieves the appropriate spectra for fitting.

        Returns:
            list: A list of spectra to be fitted.
        """
        return (
            [self.dataset.spectra[self.dataset.props.spectrum_for_prefit]]
            if isinstance(self, Prefitter)
            else self.dataset.spectra
        )

    def _get_amplitude_dict(self, peak, nr):
        """
        Generates an amplitude parameter dictionary for a given peak.

        Args:
            peak: A peak object containing fitting information.
            nr (int): The spectrum index.

        Returns:
            dict: A dictionary defining the amplitude parameter.
        """
        return {
            "name": f"{peak.peak_label}_amp_{nr}",
            "value": 200 if peak.peak_sign == "+" else -200,
            "min": 0 if peak.peak_sign == "+" else -np.inf,
            "max": np.inf if peak.peak_sign == "+" else 0,
        }

    def _get_center_dict(self, peak, nr):
        """
        Generates a center parameter dictionary for a given peak.

        Args:
            peak: A peak object containing fitting information.
            nr (int): The spectrum index.

        Returns:
            dict: A dictionary defining the center parameter.
        """
        return {
            "name": f"{peak.peak_label}_cen_{nr}",
            "value": peak.peak_center,
            "min": peak.peak_center - 1,
            "max": peak.peak_center + 1,
        }

    def _get_lw_dict(self, peak, nr, lw):
        """
        Generates a linewidth parameter dictionary for a given peak.

        Args:
            peak: A peak object containing fitting information.
            nr (int): The spectrum index.
            lw (str): The linewidth type (e.g., 'sigma', 'gamma').

        Returns:
            dict: A dictionary defining the linewidth parameter.
        """
        return {
            "name": f"{peak.peak_label}_{lw}_{nr}",
            "value": (
                peak.line_broadening[lw]["min"]
                + peak.line_broadening[lw]["max"]
            )
            / 2,
            "min": peak.line_broadening[lw]["min"],
            "max": peak.line_broadening[lw]["max"],
        }

    def _spectral_fitting(self, params, x_axis, y_axis):
        """
        Computes the residual between the fitted and experimental spectra.

        Args:
            params (lmfit.Parameters): The fitting parameters.
            x_axis (list): List of x-axis values.
            y_axis (list): List of y-axis values.

        Returns:
            np.ndarray: The residual between the fitted and experimental spectra.
        """
        residual = copy.deepcopy(y_axis)
        params_dict_list = functions.generate_spectra_param_dict(params)
        for key, val_list in params_dict_list.items():
            for val in val_list:
                simspec = [0 for _ in range(len(x_axis[key]))]
                simspec = functions.calc_peak(x_axis[key], simspec, val)
                residual[key] -= simspec
        return np.concatenate(residual)


class Prefitter(Fitter):
    """Fitter for a single preselected spectrum."""

    def _generate_axis_list(self):
        spectrum_for_prefit = self.dataset.props.spectrum_for_prefit
        x_axis, y_axis = [], []
        x_axis.append(self.dataset.spectra[spectrum_for_prefit].x_axis)
        y_axis.append(self.dataset.spectra[spectrum_for_prefit].y_axis)
        return x_axis, y_axis


class GlobalFitter(Fitter):
    """Fitter with global parameter constraints across spectra."""

    def _set_param_expr(self, params):
        for keys in params.keys():
            splitted_keys = keys.split("_")
            if splitted_keys[-1] != "0" and splitted_keys[-2] != "amp":
                splitted_keys[-1] = "0"
                params[keys].expr = "_".join(splitted_keys)
        return params


class SingleFitter(Fitter):
    """A basic extension of Fitter with no modifications."""

    pass


class BuildupFitter:

    def __init__(self, dataset):
        self.dataset = dataset

    def perform_fit(self):
        result_list = []
        for peak in self.dataset.peak_list:
            default_param_dict = self._get_default_param_dict(peak)
            lhs_init_params = self._get_lhs_init_params(default_param_dict)
            best_result = None
            best_chisqr = np.inf
            for init_nr, init_params in enumerate(lhs_init_params):
                params = self._set_params(default_param_dict, init_params)
                try:
                    result = self._start_minimize(params, peak._buildup_vals)
                    best_result, best_chisqr = self._check_result_quality(
                        best_result, best_chisqr, result
                    )
                except:
                    pass
            result_list.append(best_result)
        return result_list

    def _get_lhs_init_params(self, default_param_dict, n_samples=1):
        param_bounds = []
        param_bounds = [
            self._get_param_bounds(default_param_dict[key])
            for key in default_param_dict
        ]
        if n_samples == 1:
            n_samples = len(default_param_dict.keys()) * 100
        lhs_samples = lhs(len(default_param_dict.keys()), samples=n_samples)
        return self._set_sample_params(lhs_samples, param_bounds)

    def _start_minimize(self, params, args):
        return lmfit.minimize(
            self._fitting_function,
            params,
            args=(args.tdel, args.intensity),
        )

    def _check_result_quality(self, best_result, best_chisqr, result):
        if result.chisqr < best_chisqr:
            return result, result.chisqr
        else:
            return best_result, best_chisqr

    def _get_param_bounds(self, params):
        return (params["min"], params["max"])

    def _set_sample_params(self, lhs_samples, param_bounds):
        sampled_params = []
        for sample in lhs_samples:
            scaled_sample = []
            for i, (low, high) in enumerate(param_bounds):
                scaled_sample.append(low + sample[i] * (high - low))
            sampled_params.append(scaled_sample)
        return sampled_params

    def _set_params(self, default_param_dict, init_params):
        params = lmfit.Parameters()
        for key_nr, key in enumerate(default_param_dict.keys()):
            default_param_dict[key]["value"] = init_params[key_nr]
            params.add(key, **default_param_dict[key])
        return params

    def _fitting_function(self, params, tdel, intensity):
        residual = copy.deepcopy(intensity)
        param_list = self._generate_param_list(params)
        intensity_sim = self._calc_intensity(tdel, param_list)
        return [a - b for a, b in zip(residual, intensity_sim)]

    def _generate_param_list(self, params):
        param_list = []
        for key in params:
            param_list.append((params[key].value))
        return param_list

    def _get_intensity_dict(self, peak):
        return (
            dict(value=10, min=0, max=max(peak.buildup_vals.intensity) * 3)
            if peak.peak_sign == "+"
            else dict(
                value=10, max=0, min=min(peak.buildup_vals.intensity) * 3
            )
        )

    def _get_time_dict(self, peak):
        return dict(value=5, min=0, max=max(peak.buildup_vals.tdel) * 3)


class BiexpFitter(BuildupFitter):
    """
    Class for fitting biexponential models to buildup data.
    """

    def _get_default_param_dict(self, peak):
        return {
            "A1": self._get_intensity_dict(peak),
            "A2": self._get_intensity_dict(peak),
            "t1": self._get_time_dict(peak),
            "t2": self._get_time_dict(peak),
        }

    def _calc_intensity(self, tdel, param):
        return functions.calc_biexponential(tdel, param)


class BiexpFitterWithOffset(BuildupFitter):
    """
    Class for fitting biexponential models with offsets to buildup data.
    """

    def _get_default_param_dict(self, peak):
        return {
            "A1": self._get_intensity_dict(peak),
            "A2": self._get_intensity_dict(peak),
            "t1": self._get_time_dict(peak),
            "t2": self._get_time_dict(peak),
            "x1": dict(value=0, min=-5, max=5),
        }

    def _calc_intensity(self, tdel, param):
        return functions.calc_biexponential_with_offset(tdel, param)


class ExpFitter(BuildupFitter):
    """
    Class for fitting exponential models to buildup data.
    """

    def _get_default_param_dict(self, peak):
        return {
            "A1": self._get_intensity_dict(peak),
            "t1": self._get_time_dict(peak),
        }

    def _calc_intensity(self, tdel, param):
        return functions.calc_exponential(tdel, param)


class ExpFitterWithOffset(BuildupFitter):
    """
    Class for fitting exponential models with offsets to buildup data.
    """

    def _get_default_param_dict(self, peak):
        return {
            "A1": self._get_intensity_dict(peak),
            "t1": self._get_time_dict(peak),
            "x1": dict(value=0, min=-5, max=5),
        }

    def _calc_intensity(self, tdel, param):
        return functions.calc_exponential_with_offset(tdel, param)
