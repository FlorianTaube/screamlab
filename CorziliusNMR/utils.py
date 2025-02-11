from scipy.special import wofz
import lmfit
import re
import numpy as np
import copy


class Fitter:

    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self):
        x_axis, y_axis = self._generate_axis_list()
        params = self._generate_params_list()
        params = (
            self._set_param_expr(params)
            if isinstance(self, GlobalFitter)
            else params
        )
        result = lmfit.minimize(
            self._spectral_fitting, params, args=(x_axis, y_axis)
        )
        return result

    def _generate_axis_list(self):
        x_axis, y_axis = [], []
        for spectrum in self.dataset.spectra:
            x_axis.append(spectrum.x_axis)
            y_axis.append(spectrum.y_axis)
        return x_axis, y_axis

    def _generate_params_list(self):
        params = lmfit.Parameters()
        spectra = self._get_spectra_list()
        for spectrum_nr, _ in enumerate(spectra):
            for peak in self.dataset.peak_list:
                params.add(**self._get_amplitude_dict(peak, spectrum_nr))
                params.add(**self._get_center_dict(peak, spectrum_nr))
                if peak.fitting_type == "voigt":
                    params.add(
                        **self._get_lw_dict(peak, spectrum_nr, "sigma")
                    )
                    params.add(
                        **self._get_lw_dict(peak, spectrum_nr, "gamma")
                    )
                elif peak.fitting_type == "gauss":
                    params.add(
                        **self._get_lw_dict(peak, spectrum_nr, "sigma")
                    )
                elif peak.fitting_type == "lorentz":
                    params.add(
                        **self._get_lw_dict(peak, spectrum_nr, "gamma")
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

    def _set_param_expr(self, params):
        pass


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
        pass


class BiexpFitter(BuildupFitter):
    """
    Class for fitting biexponential models to buildup data.
    """

    pass


class BiexpFitterWithOffset(BuildupFitter):
    """
    Class for fitting biexponential models with offsets to buildup data.
    """

    pass


class ExpFitter(BuildupFitter):
    """
    Class for fitting exponential models to buildup data.
    """

    pass


class ExpFitterWithOffset(BuildupFitter):
    """
    Class for fitting exponential models with offsets to buildup data.
    """

    pass


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
