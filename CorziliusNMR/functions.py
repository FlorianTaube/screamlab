import numpy as np
import re
from CorziliusNMR import utils
from scipy.special import wofz


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


def calc_exponential(time_vals, param):
    return list(param[0] * (1 - np.exp(-np.asarray(time_vals) / param[1])))


def calc_biexponential(time_vals, param):
    return list(
        param[0] * (1 - np.exp(-np.asarray(time_vals) / param[2]))
        + param[1] * (1 - np.exp(-np.asarray(time_vals) / param[3]))
    )


def calc_exponential_with_offset(time_vals, param):
    return list(
        param[0]
        * (1 - np.exp(-(np.asarray(time_vals) - param[2]) / param[1]))
    )


def calc_biexponential_with_offset(time_vals, param):
    return list(
        param[0]
        * (1 - np.exp(-(np.asarray(time_vals) - param[4]) / param[2]))
        + param[1]
        * (1 - np.exp(-(np.asarray(time_vals) - param[4]) / param[3]))
    )


def generate_spectra_param_dict(params):
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


def calc_peak(x_axis, simspec, val):
    if len(val) == 5:
        simspec += voigt_profile(x_axis, val[1], val[2], val[3], val[0])
    if len(val) == 3:
        simspec += gauss_profile(x_axis, val[1], val[2], val[0])
    if len(val) == 4:
        simspec += lorentz_profile(x_axis, val[1], val[2], val[0])
    return simspec
