from scipy.special import wofz
import lmfit
import re
import numpy as np
import copy
import matplotlib.pyplot as plt
from pyDOE2 import lhs


class Fitter:

    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self):
        x_axis, y_axis = self._generate_axis_list()
        params = self._generate_params_list()
        params = self._set_param_expr(params)
        return lmfit.minimize(
            self._spectral_fitting, params, args=(x_axis, y_axis)
        )

    def _set_param_expr(self, params):
        return params

    def _generate_axis_list(self):
        x_axis, y_axis = [], []
        for spectrum in self.dataset.spectra:
            x_axis.append(spectrum.x_axis)
            y_axis.append(spectrum.y_axis)
        return x_axis, y_axis

    def _generate_params_list(self):
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
        return (
            [self.dataset.spectra[self.dataset.props.spectrum_for_prefit]]
            if isinstance(self, Prefitter)
            else self.dataset.spectra
        )

    def _get_amplitude_dict(self, peak, nr):
        return {
            "name": f"{peak.peak_label}_amp_{nr}",
            "value": 200 if peak.peak_sign == "+" else -200,
            "min": 0 if peak.peak_sign == "+" else -np.inf,
            "max": np.inf if peak.peak_sign == "+" else 0,
        }

    def _get_center_dict(self, peak, nr):
        return {
            "name": f"{peak.peak_label}_cen_{nr}",
            "value": peak.peak_center,
            "min": peak.peak_center - 1,
            "max": peak.peak_center + 1,
        }

    def _get_lw_dict(self, peak, nr, lw):
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
        residual = copy.deepcopy(y_axis)
        y_sim = None
        params_dict_list = self._sort_params(params)
        for key, val_list in params_dict_list.items():
            for val in val_list:
                if len(val) == 5:
                    y_sim = voigt_profile(
                        x_axis[key],
                        val[1],
                        val[2],
                        val[3],
                        val[0],
                    )
                if len(val) == 3:
                    y_sim = gauss_profile(
                        x_axis[key],
                        val[1],
                        val[2],
                        val[0],
                    )
                if len(val) == 4:
                    y_sim = lorentz_profile(
                        x_axis[key],
                        val[1],
                        val[2],
                        val[0],
                    )
                residual[key] -= y_sim
        return np.concatenate(residual)

    def _sort_params(self, params):
        param_dict = {}
        prefix, lastfix = None, None
        dict_index = -1
        param_value_list = []
        for param in params:
            parts = re.split(r"_(cen|amp|sigma|gamma)_", param)
            if prefix != parts[0]:
                if param_value_list:
                    param_dict[dict_index].append(param_value_list)
                prefix = parts[0]
                param_value_list = []
            if lastfix != parts[2]:
                if param_value_list:
                    param_dict[dict_index].append(param_value_list)
                    param_value_list = []
                lastfix = parts[2]
                dict_index += 1
            if dict_index not in param_dict:
                param_dict[dict_index] = []
            param_value_list.append(float(params[param].value))
            if parts[1] == "gamma":
                param_value_list.append("gam")
        param_dict[dict_index].append(param_value_list)
        return param_dict


class Prefitter(Fitter):

    def _generate_axis_list(self):
        spectrum_for_prefit = self.dataset.props.spectrum_for_prefit
        x_axis, y_axis = [], []
        x_axis.append(self.dataset.spectra[spectrum_for_prefit].x_axis)
        y_axis.append(self.dataset.spectra[spectrum_for_prefit].y_axis)
        return x_axis, y_axis


class GlobalFitter(Fitter):

    def _set_param_expr(self, params):
        for keys in params.keys():
            splitted_keys = keys.split("_")
            if splitted_keys[-1] != "0" and splitted_keys[-2] != "amp":
                splitted_keys[-1] = "0"
                params[keys].expr = "_".join(splitted_keys)
        return params


class SingleFitter(Fitter):
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
            for init_params in lhs_init_params:
                params = self._set_params(default_param_dict, init_params)
                result = self._start_minimize(params, peak._buildup_vals)
                best_result, best_chisqr = self._check_result_quality(
                    best_result, best_chisqr, result
                )
            result_list.append(best_result)

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
        return residual - intensity_sim

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
        return list(
            param[0] * (1 - np.exp(-np.asarray(tdel) / param[2]))
            + param[1] * (1 - np.exp(-np.asarray(tdel) / param[3]))
        )


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
        return list(
            param[0] * (1 - np.exp(-(np.asarray(tdel) - param[4]) / param[2]))
            + param[1]
            * (1 - np.exp(-(np.asarray(tdel) - param[4]) / param[3]))
        )


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
        return list(param[0] * (1 - np.exp(-np.asarray(tdel) / param[1])))


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
        return list(
            param[0] * (1 - np.exp(-(np.asarray(tdel) - param[2]) / param[1]))
        )


def voigt_profile(x, center, sigma, gamma, amplitude):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def gauss_profile(x, center, sigma, amplitude):
    return (
        amplitude
        * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        / (sigma * np.sqrt(2 * np.pi))
    )


def lorentz_profile(x, center, gamma, amplitude):
    return (
        amplitude
        * (gamma**2 / ((x - center) ** 2 + gamma**2))
        / (np.pi * gamma)
    )


def fwhm_gaussian(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def fwhm_lorentzian(gamma):
    return 2 * gamma


def fwhm_voigt(sigma, gamma):
    return 0.5346 * (2 * gamma) + np.sqrt(
        0.2166 * (2 * gamma) ** 2 + 4 * np.log(2) * sigma**2
    )
