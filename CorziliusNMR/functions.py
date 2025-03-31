import numpy as np
import re
from scipy.special import wofz


def voigt_profile(x, center, sigma, gamma, amplitude):
    """
        Compute the Voigt profile, a convolution of a Gaussian and Lorentzian function.
    a
        :param x: Array of x values.
        :param center: Center of the peak.
        :param sigma: Standard deviation of the Gaussian component.
        :param gamma: Half-width at half-maximum (HWHM) of the Lorentzian component.
        :param amplitude: Peak amplitude.
        :return: Voigt profile evaluated at x.
    """
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def gauss_profile(x, center, sigma, amplitude):
    """
    Compute a Gaussian profile.

    :param x: Array of x values.
    :param center: Center of the peak.
    :param sigma: Standard deviation of the Gaussian distribution.
    :param amplitude: Peak amplitude.
    :return: Gaussian profile evaluated at x.
    """
    return (
        amplitude
        * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        / (sigma * np.sqrt(2 * np.pi))
    )


def lorentz_profile(x, center, gamma, amplitude):
    """
    Compute a Lorentzian profile.

    :param x: Array of x values.
    :param center: Center of the peak.
    :param gamma: Half-width at half-maximum (HWHM) of the Lorentzian function.
    :param amplitude: Peak amplitude.
    :return: Lorentzian profile evaluated at x.
    """
    return (
        amplitude
        * (gamma**2 / ((x - center) ** 2 + gamma**2))
        / (np.pi * gamma)
    )


def fwhm_gaussian(sigma):
    """
    Compute the full width at half maximum (FWHM) for a Gaussian function.

    :param sigma: Standard deviation of the Gaussian.
    :return: FWHM value.
    """
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def fwhm_lorentzian(gamma):
    """
    Compute the full width at half maximum (FWHM) for a Lorentzian function.

    :param gamma: Half-width at half-maximum.
    :return: FWHM value.
    """
    return 2 * gamma


def fwhm_voigt(sigma, gamma):
    """
    Approximate the full width at half maximum (FWHM) for a Voigt function.

    :param sigma: Standard deviation of the Gaussian component.
    :param gamma: HWHM of the Lorentzian component.
    :return: Estimated FWHM value.
    """
    return 0.5346 * (2 * gamma) + np.sqrt(
        0.2166 * (2 * gamma) ** 2 + 4 * np.log(2) * sigma**2
    )


def calc_exponential(time_vals, param):
    """
    Compute an exponential decay function.

    :param time_vals: Time values.
    :param param: List containing amplitude and decay constant.
    :return: List of computed exponential values.
    """
    return list(param[0] * (1 - np.exp(-np.asarray(time_vals) / param[1])))


def calc_stretched_exponential(time_vals, param):
    """
    Compute a stretched exponential decay function.

    :param time_vals: Time values.
    :param param: List containing amplitude, time constant, and stretching exponent.
    :return: List of computed stretched exponential values.
    """
    return list(
        param[0]
        * (1 - np.exp(-((np.asarray(time_vals) / param[1]) ** param[2])))
    )


def calc_biexponential(time_vals, param):
    """
    Compute a biexponential decay function.

    :param time_vals: Time values.
    :param param: List containing two amplitudes and two decay constants.
    :return: List of computed biexponential values.
    """
    return list(
        param[0] * (1 - np.exp(-np.asarray(time_vals) / param[2]))
        + param[1] * (1 - np.exp(-np.asarray(time_vals) / param[3]))
    )


def calc_exponential_with_offset(time_vals, param):
    """
    Compute an exponential decay function with an offset.

    :param time_vals: Time values.
    :param param: List containing amplitude, decay constant, and offset.
    :return: List of computed exponential values.
    """
    return list(
        param[0]
        * (1 - np.exp(-(np.asarray(time_vals) - param[2]) / param[1]))
    )


def calc_biexponential_with_offset(time_vals, param):
    """
    Compute a biexponential decay function with an offset.

    :param time_vals: Time values.
    :param param: List containing two amplitudes, two decay constants, and an offset.
    :return: List of computed biexponential values.
    """
    return list(
        param[0]
        * (1 - np.exp(-(np.asarray(time_vals) - param[4]) / param[2]))
        + param[1]
        * (1 - np.exp(-(np.asarray(time_vals) - param[4]) / param[3]))
    )


def generate_spectra_param_dict(params):
    """
    Generate a dictionary of spectral parameters from a list of parameter names.

    :param params: Dictionary of parameter names and values.
    :return: Dictionary of structured parameter values.
    """
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
    """
    Compute and add spectral peaks based on given parameters.

    :param x_axis: X-axis values.
    :param simspec: Simulated spectrum array.
    :param val: List of peak parameters.
    :return: Updated simulated spectrum.
    """
    if len(val) == 5:
        simspec += voigt_profile(x_axis, val[1], val[2], val[3], val[0])
    if len(val) == 3:
        simspec += gauss_profile(x_axis, val[1], val[2], val[0])
    if len(val) == 4:
        simspec += lorentz_profile(x_axis, val[1], val[2], val[0])
    return simspec


def format_mapping():
    """
    Returns a dictionary mapping each buildup function type to its corresponding parameter names.

    The dictionary defines the parameters required for fitting different buildup functions such as
    'exponential', 'biexponential', etc. The parameters are returned as lists of strings.

    :return: Dictionary mapping function types to parameter names.
    :rtype: dict
    """
    return {
        "exponential": [
            "A1",
            "t1",
            "---",
            "---",
            "---",
            "R1",
            "---",
            "S1",
            "---",
            "---",
        ],
        "streched_exponential": [
            "A1",
            "t1",
            "---",
            "---",
            "---",
            "R1",
            "---",
            "S1",
            "---",
            "beta",
        ],
        "exponential_with_offset": [
            "A1",
            "t1",
            "---",
            "---",
            "x1",
            "R1",
            "---",
            "S1",
            "---",
            "---",
        ],
        "biexponential": [
            "A1",
            "t1",
            "A2",
            "t2",
            "---",
            "R1",
            "R2",
            "S1",
            "S2",
            "---",
        ],
        "biexponential_with_offset": [
            "A1",
            "t1",
            "A2",
            "t2",
            "x1",
            "R1",
            "R2",
            "S1",
            "S2",
            "---",
        ],
    }


def buildup_header():
    """
    Returns a list of headers for the buildup analysis table.

    This header list defines the labels for the columns in the buildup fit table, which includes
    parameters like amplitude, time constants, offsets, and sensitivities.

    :return: List of headers for the buildup table.
    :rtype: list
    """
    return [
        "Label",
        "A1 / a.u.",
        "t1 / s",
        "A2 / a.u.",
        "t2 / s",
        "t_off / s",
        "R1 / 1/s",
        "R2 / 1/s",
        "Sensitivity1 (A1/sqrt(t1))",
        "Sensitivity2 (A2/sqrt(t2))",
    ]


def spectrum_fit_header():
    """
    Returns a list of headers for the spectrum fitting results table.

    The header includes typical fitting parameters such as time, center, amplitude, sigma,
    gamma, and full width at half maximum (FWHM) for various line shapes.

    :return: List of headers for the spectrum fit table.
    :rtype: list
    """
    return [
        "Label",
        "Time / s",
        "Center / ppm",
        "Amplitude / a.u.",
        "Sigma / ppm",
        "Gamma / ppm",
        "FWHM Lorentz / ppm",
        "FWHM Gauss / ppm",
        "FWHM Voigt / ppm",
        "Integral / a.u.",
    ]


def return_func_map():
    """
    Returns a dictionary mapping function types to their corresponding fitting functions.

    The dictionary maps buildup function types (e.g., 'exponential', 'biexponential') to
    the respective function used for calculation (e.g., `calc_exponential`, `calc_biexponential`).

    :return: Dictionary mapping function types to fitting functions.
    :rtype: dict
    """
    return {
        "exponential": calc_exponential,
        "biexponential": calc_biexponential,
        "exponential_with_offset": calc_exponential_with_offset,
        "biexponential_with_offset": calc_biexponential_with_offset,
        "streched_exponential": calc_stretched_exponential,
    }


def generate_subspec(spectrum, subspec):
    start = np.argmax(spectrum.x_axis < max(subspec))
    stop = np.argmax(spectrum.x_axis < min(subspec))
    return spectrum.x_axis[start:stop], spectrum.y_axis[start:stop]
